"""Fonte: Genius — busca + scraping da pagina HTML para extrair a letra.

Tem dois modos:

1. **API com token** (`GENIUS_TOKEN`): usa `api.genius.com/search`, com mais quota
   estavel. Token gratuito em https://genius.com/api-clients.
2. **Publico (sem token)**: cai no endpoint `genius.com/api/search/multi` que o
   proprio site usa para autocompletar — nao exige autenticacao. Cota menor e
   mais sensivel a Cloudflare, mas funciona como fallback.

A pagina HTML em si e a mesma nos dois modos, e e protegida por Cloudflare. Para
reduzir blocking:
- requisicao da pagina usa pool de User-Agents realista (rotacionado por chamada);
- rate limit conservador (1 rps default) com penalty em `Retry-After`;
- circuit breaker abre apos falhas consecutivas para nao queimar a fonte.
"""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup, Tag

from music_search._async_http.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)
from music_search._async_http.user_agents import random_browser_headers
from music_search.lyrics.normalize import slugify
from music_search.lyrics.sources.base import LyricsResult, Status


class GeniusSource:
    API = "https://api.genius.com"
    PUBLIC = "https://genius.com/api"

    def __init__(
        self,
        client: httpx.AsyncClient,
        token: str | None = None,
        timeout: float = 20.0,
        rate_limiter: AsyncRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.client = client
        self.token = (token or "").strip() or None
        self.timeout = timeout
        # Pagina HTML e mais sensivel — 1 rps default para nao acordar o WAF.
        self.rate_limiter = rate_limiter or AsyncRateLimiter(rate=1.0, capacity=2.0)
        # CB mais agressivo: 3 falhas consecutivas = pausa de 2 minutos.
        self.circuit_breaker = circuit_breaker or CircuitBreaker(max_failures=3, cooldown=120.0)

    @property
    def name(self) -> str:
        return "genius" if self.token else "genius_public"

    async def fetch(self, artist: str, title: str) -> LyricsResult:
        if self.circuit_breaker.is_open:
            return LyricsResult(Status.BLOCKED, source=self.name, error="circuit open")

        await self.rate_limiter.acquire()
        if self.token:
            search_url = f"{self.API}/search"
            search_headers = {"Authorization": f"Bearer {self.token}"}
        else:
            # endpoint publico usado pelo proprio site: dispensa token e tem schema
            # parecido (response.hits[*].result.url), mas envelope multi-secao.
            search_url = f"{self.PUBLIC}/search/multi"
            search_headers = random_browser_headers(referer="https://genius.com/")
        try:
            search = await self.client.get(
                search_url,
                params={"q": f"{artist} {title}", "per_page": "5"},
                headers=search_headers,
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
        if search.status_code in (403, 503):
            # Cloudflare na busca publica: drena bucket
            self.rate_limiter.penalize(60.0)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                error=f"search http {search.status_code} (cloudflare?)",
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

        try:
            payload = search.json()
        except ValueError:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="invalid search json", source=self.name)
        hits = self._extract_hits(payload)
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
    def _extract_hits(payload: dict) -> list[dict]:
        """Normaliza o payload das duas variantes de busca (API e publica).

        - API (`api.genius.com/search`): `response.hits[*]` com `type=song`.
        - Publica (`genius.com/api/search/multi`): `response.sections[*].hits[*]`
          contem varios tipos (song, album, artist...). So songs interessam.
        """
        response = payload.get("response", {}) or {}
        if "hits" in response:
            return [h for h in (response.get("hits") or []) if h.get("type") == "song"]
        sections = response.get("sections") or []
        out: list[dict] = []
        for section in sections:
            if section.get("type") != "song":
                continue
            for hit in section.get("hits") or []:
                if hit.get("type") == "song":
                    out.append(hit)
        return out

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
