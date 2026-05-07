"""Fonte Wikipedia PT: usa REST API para resolver titulo e baixar HTML da pagina.

API:
- `GET /w/api.php?action=query&list=search&srsearch=...&format=json` -> resolve titulo
- `GET /api/rest_v1/page/html/{title}` -> HTML normalizado da pagina

Sem chave. Rate limit conservador (1 rps) + headers de User-Agent identificavel para
ficar dentro da etiqueta da Wikimedia.
"""

from __future__ import annotations

import logging
import urllib.parse

import httpx

from music_search._async_http.pipeline import FetchResult, Status
from music_search._async_http.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)

logger = logging.getLogger(__name__)

API_BASE = "https://pt.wikipedia.org/w/api.php"
REST_BASE = "https://pt.wikipedia.org/api/rest_v1/page/html"

# UA estavel/identificavel — Wikimedia espera contato; UA aleatorio e desencorajado.
USER_AGENT = (
    "music-search-engine/0.1 (UFAM-ICC222 academic research; "
    "+https://github.com/carsio/music-search-engine; carsio1256@gmail.com)"
)


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
    """Fonte Wikipedia em portugues. Usa search API + REST HTML."""

    name = "wikipedia_pt"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        rate_per_sec: float = 1.0,
        max_failures: int = 3,
        cooldown: float = 60.0,
    ):
        self.client = client
        self.limiter = AsyncRateLimiter(rate=rate_per_sec)
        self.breaker = CircuitBreaker(max_failures=max_failures, cooldown=cooldown)

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
