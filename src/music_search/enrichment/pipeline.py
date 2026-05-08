"""Pipeline de enrichment em duas fases: coleta web -> normalizacao via LLM."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

ExtractFn = Callable[..., Awaitable[dict]]
RAW_FETCH_STATUS = "fetched"
_RAW_CONTENT_FIELD = "_raw_content"

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
    fetch_documents: bool = True
    normalize_documents: bool = True


def _build_raw_payload(content: str) -> dict[str, str]:
    return {_RAW_CONTENT_FIELD: content}


def _raw_content_from_row(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    content = payload.get(_RAW_CONTENT_FIELD)
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def _needs_fetch(row: dict[str, Any] | None, *, retry_errors: bool) -> bool:
    if row is None:
        return True

    status = str(row.get("status") or "")
    if status in (Status.HIT.value, RAW_FETCH_STATUS, Status.MISS.value, Status.BLOCKED.value):
        return False
    if status == Status.ERROR.value:
        if _raw_content_from_row(row) is not None:
            return False
        return retry_errors
    return True


def _needs_normalize(row: dict[str, Any] | None, *, retry_errors: bool) -> bool:
    if _raw_content_from_row(row) is None:
        return False

    status = str(row.get("status") or "")
    if status == RAW_FETCH_STATUS:
        return True
    if status == Status.ERROR.value:
        return retry_errors
    return False


async def _fetch_seed(
    *,
    kind: EntityKind,
    query: str,
    sources: list[EnrichmentSource],
    cascade_cfg: CascadeConfig,
    cache: KeyValueCache,
    cache_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        item = {"kind": kind, "query": query}
        result, trace, attempts = await cascade_fetch(item, sources, cascade_cfg)

        status = result.status.value
        payload = None
        if result.status == Status.HIT and result.payload:
            status = RAW_FETCH_STATUS
            payload = _build_raw_payload(result.payload)

        async with cache_lock:
            cache.upsert(
                key=make_key(kind, query),
                kind=kind,
                status=status,
                source=result.source,
                source_url=result.source_url,
                payload=payload,
                error=result.error,
                attempts=attempts,
                trace=serialize_trace(trace),
            )
        return status


async def _normalize_seed(
    *,
    kind: EntityKind,
    query: str,
    cache: KeyValueCache,
    cache_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    llm_client: NimClient,
    llm_cache: LLMCache | None,
) -> str:
    async with semaphore:
        key = make_key(kind, query)
        async with cache_lock:
            row = cache.get(key)

        raw_content = _raw_content_from_row(row)
        if row is None or raw_content is None:
            return "skipped"

        source = row.get("source")
        source_url = row.get("source_url")
        trace = row.get("trace")
        extractor = _EXTRACTORS[kind]

        try:
            payload = await extractor(
                raw_content,
                source_url=source_url,
                client=llm_client,
                cache=llm_cache,
            )
        except Exception as exc:
            logger.warning("LLM extract failed for %s/%s: %s", kind, query, exc)
            async with cache_lock:
                cache.upsert(
                    key=key,
                    kind=kind,
                    status=Status.ERROR.value,
                    source=source,
                    source_url=source_url,
                    payload=row.get("payload"),
                    error=f"llm extract: {exc}",
                    attempts=1,
                    trace=trace,
                )
            return Status.ERROR.value

        payload.setdefault("source", source)
        payload.setdefault("source_url", source_url)
        async with cache_lock:
            cache.upsert(
                key=key,
                kind=kind,
                status=Status.HIT.value,
                source=source,
                source_url=source_url,
                payload=payload,
                attempts=1,
                trace=trace,
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
    """Roda enrichment por fase para uma entidade usando o cache como handoff."""
    cache = KeyValueCache(cfg.cache_path, table=cfg.cache_table)
    seeds_list = list(seeds)
    if cfg.limit:
        seeds_list = seeds_list[: cfg.limit]
    did_work = False
    try:
        if cfg.fetch_documents:
            pending_fetch = [
                s
                for s in seeds_list
                if _needs_fetch(cache.get(make_key(kind, s)), retry_errors=cfg.retry_errors)
            ]
            if pending_fetch:
                did_work = True
                print(
                    f"Coleta {kind}: {len(pending_fetch):,} pendentes / {len(seeds_list):,} totais "
                    f"(concurrency={cfg.concurrency})"
                )
                timeout = httpx.Timeout(cfg.request_timeout, connect=10.0)
                limits = httpx.Limits(max_connections=cfg.concurrency * 2)
                cascade_cfg = CascadeConfig(
                    concurrency=cfg.concurrency,
                    request_timeout=cfg.request_timeout,
                    retry_errors=cfg.retry_errors,
                )
                async with httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    follow_redirects=True,
                ) as http:
                    sources: list[EnrichmentSource] = [WikipediaPTSource(http)]
                    semaphore = asyncio.Semaphore(cfg.concurrency)
                    cache_lock = asyncio.Lock()
                    tasks = [
                        _fetch_seed(
                            kind=kind,
                            query=q,
                            sources=sources,
                            cascade_cfg=cascade_cfg,
                            cache=cache,
                            cache_lock=cache_lock,
                            semaphore=semaphore,
                        )
                        for q in pending_fetch
                    ]
                    await atqdm.gather(*tasks, desc=f"Coleta {kind}", unit="seed")

        if cfg.normalize_documents:
            pending_normalize = [
                s
                for s in seeds_list
                if _needs_normalize(cache.get(make_key(kind, s)), retry_errors=cfg.retry_errors)
            ]
            if pending_normalize:
                did_work = True
                print(
                    f"Normalizacao {kind}: {len(pending_normalize):,} pendentes / "
                    f"{len(seeds_list):,} sementes selecionadas"
                )
                own_llm_client = llm_client is None
                if llm_client is None:
                    llm_client = NimClient()
                if llm_cache is None:
                    llm_cache = LLMCache()

                try:
                    semaphore = asyncio.Semaphore(cfg.concurrency)
                    cache_lock = asyncio.Lock()
                    tasks = [
                        _normalize_seed(
                            kind=kind,
                            query=q,
                            cache=cache,
                            cache_lock=cache_lock,
                            semaphore=semaphore,
                            llm_client=llm_client,
                            llm_cache=llm_cache,
                        )
                        for q in pending_normalize
                    ]
                    await atqdm.gather(*tasks, desc=f"Normalizacao {kind}", unit="seed")
                finally:
                    if own_llm_client:
                        await llm_client.aclose()
    finally:
        if not did_work:
            print(f"Nada a fazer — todas as {len(seeds_list):,} sementes ja foram processadas.")

    stats = cache.stats()
    by_kind = cache.stats_by_kind()
    cache.close()
    print("Status global:", stats)
    print("Por entidade:", by_kind)
    return stats
