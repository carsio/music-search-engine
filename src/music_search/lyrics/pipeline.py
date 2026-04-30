"""Orquestrador async: cascade de fontes, retries com backoff, semaforo, cache."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm as atqdm

from music_search.lyrics.cache import LyricsCache
from music_search.lyrics.normalize import normalize_artist, normalize_title
from music_search.lyrics.sources.base import LyricsResult, LyricsSource, Status

logger = logging.getLogger(__name__)

SourcesFactory = Callable[[httpx.AsyncClient], Sequence[LyricsSource]]


@dataclass
class PipelineConfig:
    parquet_path: Path
    cache_path: Path
    concurrency: int = 8
    request_timeout: float = 15.0
    max_retries: int = 2
    retry_initial_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_jitter: float = 0.4
    retry_errors: bool = False
    limit: int | None = None
    user_agent: str = (
        "music-search-engine-lyrics/0.1 "
        "(+https://github.com/carsio/music-search-engine; academic/UFAM ICC222)"
    )


async def _retry_fetch(
    source: LyricsSource,
    artist: str,
    title: str,
    cfg: PipelineConfig,
) -> tuple[LyricsResult, int]:
    """Tenta ate `cfg.max_retries + 1` vezes em caso de erro transitorio.

    Status terminais (HIT, MISS, BLOCKED) retornam imediatamente:
    - HIT/MISS: a fonte respondeu de forma definitiva.
    - BLOCKED: a propria fonte ja aplicou penalty no rate limiter / circuit breaker;
      retentar aqui so atrasa. O pipeline cascateia para a proxima fonte.
    Apenas ERROR (timeout, 5xx, JSON invalido) e re-tentado com backoff + jitter.
    """
    delay = cfg.retry_initial_delay
    last: LyricsResult | None = None
    attempts = 0
    for attempt in range(cfg.max_retries + 1):
        attempts += 1
        result = await source.fetch(artist, title)
        last = result
        if result.status in (Status.HIT, Status.MISS, Status.BLOCKED):
            return result, attempts
        # ERROR transitorio: backoff + jitter
        if attempt >= cfg.max_retries:
            break
        sleep_for = delay + random.uniform(0.0, cfg.retry_jitter)
        await asyncio.sleep(sleep_for)
        delay *= cfg.retry_backoff
    assert last is not None
    return last, attempts


async def _process_track(
    track: dict,
    sources: Sequence[LyricsSource],
    cache: LyricsCache,
    semaphore: asyncio.Semaphore,
    cfg: PipelineConfig,
    cache_lock: asyncio.Lock,
) -> str:
    async with semaphore:
        artist = normalize_artist(track.get("primary_artist_name") or "")
        title = normalize_title(track.get("track_name") or "")
        track_id = track["track_id"]
        isrc = track.get("isrc")

        if not artist or not title:
            async with cache_lock:
                cache.upsert(
                    track_id=track_id,
                    isrc=isrc,
                    artist=artist,
                    title=title,
                    status=Status.MISS.value,
                    error="empty artist or title",
                )
            return Status.MISS.value

        last: LyricsResult | None = None
        attempts_total = 0
        for source in sources:
            result, attempts = await _retry_fetch(source, artist, title, cfg)
            attempts_total += attempts
            last = result
            if result.status == Status.HIT:
                async with cache_lock:
                    cache.upsert(
                        track_id=track_id,
                        isrc=isrc,
                        artist=artist,
                        title=title,
                        status=Status.HIT.value,
                        source=result.source,
                        source_url=result.source_url,
                        lyrics=result.lyrics,
                        attempts=attempts_total,
                    )
                return Status.HIT.value

        final_status = (last.status if last else Status.ERROR).value
        async with cache_lock:
            cache.upsert(
                track_id=track_id,
                isrc=isrc,
                artist=artist,
                title=title,
                status=final_status,
                source=last.source if last else None,
                source_url=last.source_url if last else None,
                error=last.error if last else "no source",
                attempts=attempts_total,
            )
        return final_status


def read_tracks(parquet_path: Path, limit: int | None) -> list[dict]:
    columns = ["track_id", "isrc", "track_name", "primary_artist_name"]
    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pylist()
    if limit:
        rows = rows[:limit]
    return rows


def _filter_pending(rows: Iterable[dict], cache: LyricsCache, retry_errors: bool) -> list[dict]:
    pending: list[dict] = []
    for row in rows:
        if not cache.has_resolved(row["track_id"], retry_errors=retry_errors):
            pending.append(row)
    return pending


async def run_pipeline(cfg: PipelineConfig, sources_factory: SourcesFactory) -> dict[str, int]:
    cache = LyricsCache(cfg.cache_path)
    rows = read_tracks(cfg.parquet_path, cfg.limit)
    pending = _filter_pending(rows, cache, retry_errors=cfg.retry_errors)
    total = len(rows)

    if not pending:
        print(f"Nada a fazer — {total:,} faixas ja resolvidas no cache.")
        stats = cache.stats()
        cache.close()
        return stats

    print(
        f"Pendentes: {len(pending):,} / {total:,} (concurrency={cfg.concurrency}, "
        f"retries={cfg.max_retries})"
    )

    timeout = httpx.Timeout(cfg.request_timeout, connect=10.0)
    limits = httpx.Limits(
        max_connections=cfg.concurrency * 2, max_keepalive_connections=cfg.concurrency
    )
    headers = {"User-Agent": cfg.user_agent}

    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, headers=headers, follow_redirects=True, http2=False
    ) as client:
        sources = list(sources_factory(client))
        if not sources:
            cache.close()
            raise SystemExit("Nenhuma fonte de letras foi configurada.")
        print("Fontes:", " -> ".join(s.name for s in sources))

        semaphore = asyncio.Semaphore(cfg.concurrency)
        cache_lock = asyncio.Lock()
        tasks = [
            _process_track(track, sources, cache, semaphore, cfg, cache_lock) for track in pending
        ]
        # tqdm.gather mantem progresso enquanto tasks resolvem em ordem aleatoria
        await atqdm.gather(*tasks, desc="Letras", unit="trk")

    stats = cache.stats()
    by_source = cache.stats_by_source()
    cache.close()
    print("Status:", stats)
    print("Hits por fonte:", by_source)
    return stats
