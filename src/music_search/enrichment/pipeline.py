"""Pipeline de enrichment: source HTML (web) -> LLM -> KeyValueCache."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm as atqdm

from music_search._async_http.cache import KeyValueCache
from music_search._async_http.pipeline import (
    CascadeConfig,
    Status,
    cascade_fetch,
    serialize_trace,
)
from music_search.enrichment.models import EntityKind
from music_search.enrichment.sources.base import EnrichmentSource
from music_search.enrichment.sources.wikipedia_pt import WikipediaPTSource
from music_search.llm.cache import LLMCache
from music_search.llm.client import NimClient
from music_search.llm.tasks import (
    extract_album_json,
    extract_artist_json,
    extract_composer_json,
    extract_genre_json,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/derived/enrichment_cache.sqlite")
DEFAULT_CACHE_TABLE = "enrichment"

ExtractFn = Callable[..., dict]

_EXTRACTORS: dict[str, ExtractFn] = {
    "artist": extract_artist_json,
    "album": extract_album_json,
    "genre": extract_genre_json,
    "composer": extract_composer_json,
}


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:80] or "x"


def make_key(kind: EntityKind, query: str) -> str:
    return f"{kind}:{slugify(query)}"


@dataclass
class EnrichmentConfig:
    cache_path: Path = DEFAULT_CACHE_PATH
    cache_table: str = DEFAULT_CACHE_TABLE
    concurrency: int = 4
    request_timeout: float = 30.0
    retry_errors: bool = False
    limit: int | None = None


async def _process_seed(
    *,
    kind: EntityKind,
    query: str,
    sources: list[EnrichmentSource],
    cascade_cfg: CascadeConfig,
    cache: KeyValueCache,
    cache_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    llm_client: NimClient,
    llm_cache: LLMCache | None,
) -> str:
    async with semaphore:
        item = {"kind": kind, "query": query}
        result, trace, attempts = await cascade_fetch(item, sources, cascade_cfg)

        if result.status != Status.HIT or not result.payload:
            async with cache_lock:
                cache.upsert(
                    key=make_key(kind, query),
                    kind=kind,
                    status=result.status.value,
                    source=result.source,
                    source_url=result.source_url,
                    error=result.error,
                    attempts=attempts,
                    trace=serialize_trace(trace),
                )
            return result.status.value

        # HTML em mao -> LLM extrai JSON
        extractor = _EXTRACTORS[kind]
        try:
            payload = await extractor(
                result.payload,
                source_url=result.source_url,
                client=llm_client,
                cache=llm_cache,
            )
        except Exception as exc:
            logger.warning("LLM extract failed for %s/%s: %s", kind, query, exc)
            async with cache_lock:
                cache.upsert(
                    key=make_key(kind, query),
                    kind=kind,
                    status=Status.ERROR.value,
                    source=result.source,
                    source_url=result.source_url,
                    error=f"llm extract: {exc}",
                    attempts=attempts,
                    trace=serialize_trace(trace),
                )
            return Status.ERROR.value

        payload.setdefault("source", result.source)
        payload.setdefault("source_url", result.source_url)
        async with cache_lock:
            cache.upsert(
                key=make_key(kind, query),
                kind=kind,
                status=Status.HIT.value,
                source=result.source,
                source_url=result.source_url,
                payload=payload,
                attempts=attempts,
                trace=serialize_trace(trace),
            )
        return Status.HIT.value


async def run_enrichment(
    kind: EntityKind,
    seeds: Iterable[str],
    cfg: EnrichmentConfig,
    *,
    llm_client: NimClient | None = None,
    llm_cache: LLMCache | None = None,
) -> dict[str, int]:
    """Roda enrichment para uma entidade. `seeds` e iteravel de strings (queries)."""
    cache = KeyValueCache(cfg.cache_path, table=cfg.cache_table)
    seeds_list = list(seeds)
    if cfg.limit:
        seeds_list = seeds_list[: cfg.limit]
    pending = [
        s
        for s in seeds_list
        if not cache.has_resolved(make_key(kind, s), retry_errors=cfg.retry_errors)
    ]
    if not pending:
        print(f"Nada a fazer — todas as {len(seeds_list):,} sementes ja resolvidas.")
        stats = cache.stats()
        cache.close()
        return stats

    print(
        f"Enrichment {kind}: {len(pending):,} pendentes / {len(seeds_list):,} totais "
        f"(concurrency={cfg.concurrency})"
    )

    timeout = httpx.Timeout(cfg.request_timeout, connect=10.0)
    limits = httpx.Limits(max_connections=cfg.concurrency * 2)
    cascade_cfg = CascadeConfig(
        concurrency=cfg.concurrency,
        request_timeout=cfg.request_timeout,
        retry_errors=cfg.retry_errors,
    )

    own_llm_client = llm_client is None
    if llm_client is None:
        llm_client = NimClient()
    if llm_cache is None:
        llm_cache = LLMCache()

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as http:
            sources: list[EnrichmentSource] = [WikipediaPTSource(http)]
            semaphore = asyncio.Semaphore(cfg.concurrency)
            cache_lock = asyncio.Lock()
            tasks = [
                _process_seed(
                    kind=kind,
                    query=q,
                    sources=sources,
                    cascade_cfg=cascade_cfg,
                    cache=cache,
                    cache_lock=cache_lock,
                    semaphore=semaphore,
                    llm_client=llm_client,
                    llm_cache=llm_cache,
                )
                for q in pending
            ]
            await atqdm.gather(*tasks, desc=f"Enrichment {kind}", unit="seed")
    finally:
        if own_llm_client:
            await llm_client.aclose()

    stats = cache.stats()
    by_kind = cache.stats_by_kind()
    cache.close()
    print("Status global:", stats)
    print("Por entidade:", by_kind)
    return stats
