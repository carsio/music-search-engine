"""Pipeline de enrichment em duas fases: coleta web -> materializacao local."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

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

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/derived/enrichment_cache.sqlite")
DEFAULT_CACHE_TABLE = "enrichment"

RAW_FETCH_STATUS = "fetched"
_RAW_CONTENT_FIELD = "_raw_content"
_SEED_QUERY_FIELD = "_seed_query"
_RESOLVED_TITLE_FIELD = "_resolved_title"

MaterializeFn = Callable[[str, str | None, str | None], dict[str, Any]]


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


def _resolved_title_from_url(source_url: str | None) -> str | None:
    if not source_url:
        return None
    path = urlparse(source_url).path
    if not path:
        return None
    slug = path.split("/wiki/", 1)[-1].rsplit("/", 1)[-1]
    if not slug:
        return None
    return unquote(slug).replace("_", " ").strip() or None


def _build_raw_payload(
    content: str,
    *,
    query: str,
    source_url: str | None,
) -> dict[str, str]:
    payload = {
        _RAW_CONTENT_FIELD: content,
        _SEED_QUERY_FIELD: query,
    }
    resolved_title = _resolved_title_from_url(source_url)
    if resolved_title:
        payload[_RESOLVED_TITLE_FIELD] = resolved_title
    return payload


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


def _seed_query_from_row(row: dict[str, Any] | None, *, default: str) -> str:
    if row is None:
        return default
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return default
    query = payload.get(_SEED_QUERY_FIELD)
    if isinstance(query, str) and query.strip():
        return query.strip()
    return default


def _resolved_title_from_row(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    payload = row.get("payload")
    if isinstance(payload, dict):
        title = payload.get(_RESOLVED_TITLE_FIELD)
        if isinstance(title, str) and title.strip():
            return title.strip()
    source_url = row.get("source_url")
    if isinstance(source_url, str):
        return _resolved_title_from_url(source_url)
    return None


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_text_block(block: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[0].lower() == "resumo":
        return _collapse_whitespace(" ".join(lines[1:]))
    if lines[0].startswith("#"):
        heading = lines[0].lstrip("#").strip()
        body = _collapse_whitespace(" ".join(lines[1:]))
        if heading and body:
            return f"{heading}. {body}"
        return heading or body
    return _collapse_whitespace(" ".join(lines))


def _clean_raw_text(content: str) -> str:
    blocks = [_clean_text_block(block) for block in content.split("\n\n")]
    return "\n\n".join(block for block in blocks if block)


def _summary_from_raw_text(content: str) -> str:
    for block in content.split("\n\n"):
        cleaned = _clean_text_block(block)
        if cleaned:
            return cleaned
    return ""


def _first_sentence(text: str, *, limit: int = 160) -> str | None:
    text = _collapse_whitespace(text)
    if not text:
        return None
    match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    sentence = match.group(1) if match else text
    sentence = sentence.strip()
    if not sentence:
        return None
    if len(sentence) <= limit:
        return sentence
    truncated = sentence[:limit].rstrip(" ,.;:")
    return f"{truncated}..."


def _strip_disambiguation(title: str | None) -> str | None:
    if not title:
        return None
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return stripped or None


def _materialize_artist_document(
    raw_text: str,
    seed_query: str | None,
    resolved_title: str | None,
) -> dict[str, Any]:
    name = _strip_disambiguation(resolved_title) or (seed_query or "")
    summary = _summary_from_raw_text(raw_text)
    return {
        "name": name,
        "tagline": _first_sentence(summary),
        "bio": summary or None,
        "raw_text": raw_text,
        "genres": [],
        "origin": None,
        "year_started": None,
        "monthly_listeners": None,
        "popularity": None,
        "albums": [],
        "top_tracks": [],
    }


def _materialize_album_document(
    raw_text: str,
    seed_query: str | None,
    resolved_title: str | None,
) -> dict[str, Any]:
    title = _strip_disambiguation(resolved_title) or (seed_query or "")
    artist = ""
    if seed_query:
        for candidate in (resolved_title, _strip_disambiguation(resolved_title)):
            if not candidate:
                continue
            if seed_query.casefold().startswith(candidate.casefold()):
                artist = seed_query[len(candidate) :].strip(" -")
                break
    return {
        "title": title,
        "artist": artist,
        "description": _summary_from_raw_text(raw_text) or None,
        "raw_text": raw_text,
        "year": None,
        "tracks": [],
    }


def _materialize_genre_document(
    raw_text: str,
    seed_query: str | None,
    resolved_title: str | None,
) -> dict[str, Any]:
    name = _strip_disambiguation(resolved_title) or (seed_query or "")
    return {
        "name": name,
        "description": _summary_from_raw_text(raw_text) or None,
        "raw_text": raw_text,
        "origin": None,
        "decade": None,
        "representative_artists": [],
        "related_genres": [],
    }


def _materialize_composer_document(
    raw_text: str,
    seed_query: str | None,
    resolved_title: str | None,
) -> dict[str, Any]:
    name = _strip_disambiguation(resolved_title) or (seed_query or "")
    return {
        "name": name,
        "bio": _summary_from_raw_text(raw_text) or None,
        "raw_text": raw_text,
        "genres": [],
        "origin": None,
        "year_started": None,
        "notable_works": [],
    }


_MATERIALIZERS: dict[str, MaterializeFn] = {
    "artist": _materialize_artist_document,
    "album": _materialize_album_document,
    "genre": _materialize_genre_document,
    "composer": _materialize_composer_document,
}


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
    if row is None or _raw_content_from_row(row) is None:
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
            payload = _build_raw_payload(
                result.payload,
                query=query,
                source_url=result.source_url,
            )

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


async def _materialize_seed(
    *,
    kind: EntityKind,
    query: str,
    cache: KeyValueCache,
    cache_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
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
        materialize = _MATERIALIZERS[kind]
        seed_query = _seed_query_from_row(row, default=query)
        resolved_title = _resolved_title_from_row(row)
        cleaned_text = _clean_raw_text(raw_content)

        try:
            payload = materialize(
                cleaned_text,
                seed_query,
                resolved_title,
            )
        except Exception as exc:
            logger.warning("Materializacao local falhou para %s/%s: %s", kind, query, exc)
            async with cache_lock:
                cache.upsert(
                    key=key,
                    kind=kind,
                    status=Status.ERROR.value,
                    source=source,
                    source_url=source_url,
                    payload=row.get("payload"),
                    error=f"materialize: {exc}",
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
    llm_client: Any | None = None,
    llm_cache: Any | None = None,
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
                    f"Materializacao {kind}: {len(pending_normalize):,} pendentes / "
                    f"{len(seeds_list):,} sementes selecionadas"
                )
                semaphore = asyncio.Semaphore(cfg.concurrency)
                cache_lock = asyncio.Lock()
                tasks = [
                    _materialize_seed(
                        kind=kind,
                        query=q,
                        cache=cache,
                        cache_lock=cache_lock,
                        semaphore=semaphore,
                    )
                    for q in pending_normalize
                ]
                await atqdm.gather(*tasks, desc=f"Materializacao {kind}", unit="seed")
    finally:
        if not did_work:
            print(f"Nada a fazer — todas as {len(seeds_list):,} sementes ja foram processadas.")

    stats = cache.stats()
    by_kind = cache.stats_by_kind()
    cache.close()
    print("Status global:", stats)
    print("Por entidade:", by_kind)
    return stats
