"""Fonte: Genius — busca via API + scraping da pagina HTML para extrair a letra.

Autenticacao: token de acesso da API gratuita em https://genius.com/api-clients.
A chave entra na variavel de ambiente GENIUS_TOKEN.

A pagina HTML do Genius e protegida por Cloudflare. Para reduzir blocking:
- requisicao da pagina usa pool de User-Agents realista (rotacionado por chamada);
- rate limit conservador (1 rps default) com penalty em `Retry-After`;
- circuit breaker abre apos falhas consecutivas para nao queimar a fonte.
"""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup, Tag

from music_search.lyrics.normalize import slugify
from music_search.lyrics.sources.base import LyricsResult, Status
from music_search.lyrics.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)
from music_search.lyrics.user_agents import random_browser_headers


class GeniusSource:
    name = "genius"
    API = "https://api.genius.com"

    def __init__(
        self,
        client: httpx.AsyncClient,
        token: str,
        timeout: float = 20.0,
        rate_limiter: AsyncRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.client = client
        self.token = token
        self.timeout = timeout
        # Pagina HTML e mais sensivel — 1 rps default para nao acordar o WAF.
        self.rate_limiter = rate_limiter or AsyncRateLimiter(rate=1.0, capacity=2.0)
        # CB mais agressivo: 3 falhas consecutivas = pausa de 2 minutos.
        self.circuit_breaker = circuit_breaker or CircuitBreaker(max_failures=3, cooldown=120.0)

    async def fetch(self, artist: str, title: str) -> LyricsResult:
        if self.circuit_breaker.is_open:
            return LyricsResult(Status.BLOCKED, source=self.name, error="circuit open")

        await self.rate_limiter.acquire()
        api_headers = {"Authorization": f"Bearer {self.token}"}
        try:
            search = await self.client.get(
                f"{self.API}/search",
                params={"q": f"{artist} {title}"},
                headers=api_headers,
                timeout=self.timeout,
            )
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="search timeout", source=self.name)
        except httpx.HTTPError as exc:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error=repr(exc), source=self.name)

        if search.status_code == 429:
            wait = parse_retry_after(search.headers.get("Retry-After"), default=15.0)
            self.rate_limiter.penalize(wait)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                error=f"search rate limited (retry-after={wait}s)",
            )
        if search.status_code >= 500:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, source=self.name, error=f"search http {search.status_code}"
            )
        if search.status_code != 200:
            self.circuit_breaker.record_success()
            return LyricsResult(
                Status.MISS, source=self.name, error=f"search http {search.status_code}"
            )

        hits = search.json().get("response", {}).get("hits", [])
        match = self._best_match(hits, artist, title)
        if not match:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name)

        url = match["result"]["url"]
        # Pagina HTML: usa headers de browser com UA rotacionado.
        page_headers = random_browser_headers(referer="https://genius.com/")
        await self.rate_limiter.acquire()
        try:
            page = await self.client.get(url, timeout=self.timeout, headers=page_headers)
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, error="page timeout", source=self.name, source_url=url
            )
        except httpx.HTTPError as exc:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error=repr(exc), source=self.name, source_url=url)

        if page.status_code == 429:
            wait = parse_retry_after(page.headers.get("Retry-After"), default=30.0)
            self.rate_limiter.penalize(wait)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                source_url=url,
                error=f"page rate limited (retry-after={wait}s)",
            )
        if page.status_code in (403, 503):
            # tipico do Cloudflare — drena bucket e nao tenta de novo
            self.rate_limiter.penalize(60.0)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                source_url=url,
                error=f"page http {page.status_code} (cloudflare?)",
            )
        if page.status_code != 200:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR,
                source=self.name,
                source_url=url,
                error=f"page http {page.status_code}",
            )

        lyrics = self._parse_lyrics(page.text)
        self.circuit_breaker.record_success()
        if not lyrics:
            return LyricsResult(Status.MISS, source=self.name, source_url=url)
        return LyricsResult(Status.HIT, lyrics=lyrics, source=self.name, source_url=url)

    @staticmethod
    def _best_match(hits: list[dict], artist: str, title: str) -> dict | None:
        if not hits:
            return None
        artist_slug = slugify(artist)
        title_slug = slugify(title)
        for hit in hits:
            result = hit.get("result", {})
            primary = result.get("primary_artist", {}).get("name", "")
            primary_slug = slugify(primary)
            if artist_slug and (artist_slug in primary_slug or primary_slug in artist_slug):
                return hit
            result_title_slug = slugify(result.get("title", ""))
            if title_slug and title_slug in result_title_slug:
                return hit
        return hits[0]

    @staticmethod
    def _parse_lyrics(html: str) -> str | None:
        soup = BeautifulSoup(html, "lxml")
        containers = soup.select('[data-lyrics-container="true"]')
        if not containers:
            container = soup.find("div", class_=re.compile("Lyrics__Container"))
            containers = [container] if isinstance(container, Tag) else []
        if not containers:
            return None
        parts: list[str] = []
        for container in containers:
            for br in container.find_all("br"):
                br.replace_with("\n")
            parts.append(container.get_text())
        text = "\n".join(parts).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text or None
