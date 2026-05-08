"""Fonte Wikipedia PT: resolve titulo + baixa conteudo textual da pagina.

Fluxo:
- `GET /w/api.php?action=query&list=search&srsearch=...&format=json` resolve titulo.
- `wikipediaapi` baixa resumo/secoes em texto para alimentar a extracao da LLM.
- Fallback para `GET /api/rest_v1/page/html/{title}` caso o client Python falhe.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Any

import httpx
import wikipediaapi

from music_search._async_http.pipeline import FetchResult, Status
from music_search._async_http.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)

logger = logging.getLogger(__name__)

API_BASE = "https://pt.wikipedia.org/w/api.php"
REST_BASE = "https://pt.wikipedia.org/api/rest_v1/page/html"
WIKI_BASE = "https://pt.wikipedia.org/wiki"

# UA estavel/identificavel — Wikimedia espera contato; UA aleatorio e desencorajado.
USER_AGENT = (
    "music-search-engine/0.1 (UFAM-ICC222 academic research; "
    "+https://github.com/carsio/music-search-engine; carsio1256@gmail.com)"
)
_MAX_CONTENT_CHARS = 50_000
_MAX_SECTION_CHARS = 3_500
_MAX_SECTIONS = 24


def _intent_hint(kind: str) -> str:
    """Sufixo opcional pra desambiguar — Wikipedia tem muitos termos repetidos."""
    if kind == "artist":
        return " musico"
    if kind == "album":
        return " album"
    if kind == "genre":
        return " genero musical"
    if kind == "composer":
        return " compositor"
    return ""


class WikipediaPTSource:
    """Fonte Wikipedia em portugues. Usa search API + wikipediaapi (texto)."""

    name = "wikipedia_pt"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        prefer_python_api: bool = True,
        rate_per_sec: float = 1.0,
        max_failures: int = 3,
        cooldown: float = 60.0,
    ):
        self.client = client
        self.prefer_python_api = prefer_python_api
        self.limiter = AsyncRateLimiter(rate=rate_per_sec)
        self.breaker = CircuitBreaker(max_failures=max_failures, cooldown=cooldown)
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=USER_AGENT,
            language="pt",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

    async def fetch(self, item: dict) -> FetchResult[str]:
        if self.breaker.is_open:
            return FetchResult(status=Status.BLOCKED, source=self.name, error="circuit open")
        kind = str(item.get("kind") or "")
        query = str(item.get("query") or "").strip()
        if not query:
            return FetchResult(status=Status.MISS, source=self.name, error="empty query")

        try:
            title = await self._resolve_title(query, kind)
        except httpx.HTTPError as exc:
            self.breaker.record_failure()
            return FetchResult(status=Status.ERROR, source=self.name, error=str(exc))
        if not title:
            self.breaker.record_success()
            return FetchResult(status=Status.MISS, source=self.name, error="title not found")

        if self.prefer_python_api:
            try:
                text_payload, page_url = await self._fetch_page_text(title)
            except Exception as exc:
                logger.warning("wikipediaapi fetch failed for %s: %s", title, exc)
                text_payload, page_url = None, None
            if text_payload:
                self.breaker.record_success()
                return FetchResult(
                    status=Status.HIT,
                    payload=text_payload,
                    source=self.name,
                    source_url=page_url,
                )

        try:
            html = await self._fetch_html(title)
        except httpx.HTTPError as exc:
            self.breaker.record_failure()
            return FetchResult(status=Status.ERROR, source=self.name, error=str(exc))
        if html is None:
            self.breaker.record_success()
            return FetchResult(status=Status.MISS, source=self.name, error="html missing")

        self.breaker.record_success()
        url = f"https://pt.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        return FetchResult(status=Status.HIT, payload=html, source=self.name, source_url=url)

    async def _fetch_page_text(self, title: str) -> tuple[str | None, str | None]:
        await self.limiter.acquire()
        page = await asyncio.to_thread(self.wiki.page, title)
        if not page.exists():
            return None, None

        content = self._compose_page_text(page)
        page_url = str(getattr(page, "fullurl", "") or "")
        if not page_url:
            slug = urllib.parse.quote(title.replace(" ", "_"))
            page_url = f"{WIKI_BASE}/{slug}"
        return content or None, page_url

    def _compose_page_text(self, page: Any) -> str:
        parts: list[str] = []
        counters = {"chars": 0, "sections": 0}

        summary = str(getattr(page, "summary", "") or "").strip()
        if summary:
            block = f"Resumo\n{summary}"[:_MAX_CONTENT_CHARS]
            parts.append(block)
            counters["chars"] += len(block)

        self._append_sections(
            parts, getattr(page, "sections", []) or [], depth=1, counters=counters
        )
        if not parts:
            return ""
        return "\n\n".join(parts)[:_MAX_CONTENT_CHARS].strip()

    def _append_sections(
        self,
        parts: list[str],
        sections: list[Any],
        *,
        depth: int,
        counters: dict[str, int],
    ) -> None:
        for section in sections:
            if counters["sections"] >= _MAX_SECTIONS or counters["chars"] >= _MAX_CONTENT_CHARS:
                return

            title = str(getattr(section, "title", "") or "").strip()
            text = str(getattr(section, "text", "") or "").strip()
            if text:
                text = text[:_MAX_SECTION_CHARS]
                heading = "#" * min(depth + 1, 6)
                block = f"{heading} {title}\n{text}" if title else text
                remaining = _MAX_CONTENT_CHARS - counters["chars"]
                if remaining <= 0:
                    return
                block = block[:remaining]
                parts.append(block)
                counters["chars"] += len(block)
                counters["sections"] += 1

            nested = getattr(section, "sections", []) or []
            if nested:
                self._append_sections(parts, nested, depth=depth + 1, counters=counters)
                if counters["sections"] >= _MAX_SECTIONS or counters["chars"] >= _MAX_CONTENT_CHARS:
                    return

    async def _resolve_title(self, query: str, kind: str) -> str | None:
        await self.limiter.acquire()
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query + _intent_hint(kind),
            "srlimit": "1",
            "format": "json",
        }
        resp = await self.client.get(API_BASE, params=params, headers={"User-Agent": USER_AGENT})
        if resp.status_code in (429, 503):
            self.limiter.penalize(parse_retry_after(resp.headers.get("Retry-After")))
            resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("query", {}).get("search") or []
        if not hits:
            return None
        return hits[0].get("title")

    async def _fetch_html(self, title: str) -> str | None:
        await self.limiter.acquire()
        url = f"{REST_BASE}/{urllib.parse.quote(title.replace(' ', '_'))}"
        resp = await self.client.get(url, headers={"User-Agent": USER_AGENT})
        if resp.status_code == 404:
            return None
        if resp.status_code in (429, 503):
            self.limiter.penalize(parse_retry_after(resp.headers.get("Retry-After")))
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.text
