"""Fonte: letras.mus.br — scraping HTML, sem necessidade de chave.

Estrategia em duas etapas:
1. busca via endpoint Solr publico (`solr.sscdn.co/letras/m1/`) que devolve JSONP
   `LetrasSug({...})` — usado pelo proprio site para autocomplete;
2. com a URL relativa do melhor match, faz GET na pagina HTML e extrai a letra
   de `<div class="lyric-original">` (com fallback para `.cnt-letra-trad` / `#js-lyric-cnt`).

Anti-blocking:
- token bucket conservador (1 rps, burst 2);
- circuit breaker (3 falhas consecutivas → 120s);
- pool de User-Agents realista para a pagina HTML;
- honra `Retry-After` em 429.
"""

from __future__ import annotations

import json
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

_JSONP_WRAPPER = re.compile(r"^[A-Za-z_]+\((.*)\)\s*$", re.DOTALL)


class LetrasMusBrSource:
    name = "letras_mus_br"
    SEARCH = "https://solr.sscdn.co/letras/m1/"
    BASE = "https://www.letras.mus.br"

    def __init__(
        self,
        client: httpx.AsyncClient,
        timeout: float = 20.0,
        rate_limiter: AsyncRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.client = client
        self.timeout = timeout
        self.rate_limiter = rate_limiter or AsyncRateLimiter(rate=1.0, capacity=2.0)
        self.circuit_breaker = circuit_breaker or CircuitBreaker(max_failures=3, cooldown=120.0)

    async def fetch(self, artist: str, title: str) -> LyricsResult:
        if self.circuit_breaker.is_open:
            return LyricsResult(Status.BLOCKED, source=self.name, error="circuit open")

        # 1. busca via Solr
        await self.rate_limiter.acquire()
        try:
            search = await self.client.get(
                self.SEARCH,
                params={"q": f"{artist} {title}", "wt": "json", "rows": 5},
                timeout=self.timeout,
                headers={"Accept": "text/plain", "Referer": f"{self.BASE}/"},
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
            self.rate_limiter.penalize(60.0)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED, source=self.name, error=f"search http {search.status_code}"
            )
        if search.status_code != 200:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, source=self.name, error=f"search http {search.status_code}"
            )

        match = self._extract_match(search.text, artist, title)
        if not match:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name)

        artist_slug = match.get("dns") or slugify(match.get("art", ""))
        song_slug = slugify(match.get("txt", ""))
        if not artist_slug or not song_slug:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name)
        url = f"{self.BASE}/{artist_slug}/{song_slug}/"

        # 2. busca a pagina HTML da musica
        await self.rate_limiter.acquire()
        page_headers = random_browser_headers(referer=f"{self.BASE}/")
        try:
            page = await self.client.get(url, timeout=self.timeout, headers=page_headers)
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, error="page timeout", source=self.name, source_url=url
            )
        except httpx.HTTPError as exc:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, error=repr(exc), source=self.name, source_url=url
            )

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
            self.rate_limiter.penalize(60.0)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                source_url=url,
                error=f"page http {page.status_code} (cloudflare?)",
            )
        if page.status_code == 404:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name, source_url=url)
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
    def _extract_match(jsonp_text: str, artist: str, title: str) -> dict | None:
        match = _JSONP_WRAPPER.match(jsonp_text.strip())
        body = match.group(1) if match else jsonp_text
        try:
            data = json.loads(body)
        except ValueError:
            return None
        docs = data.get("response", {}).get("docs", []) or []
        if not docs:
            return None

        artist_slug = slugify(artist)
        title_slug = slugify(title)

        scored: list[tuple[int, dict]] = []
        for doc in docs:
            doc_art = slugify(doc.get("art", "") or "")
            doc_txt = slugify(doc.get("txt", "") or "")
            if not doc_art or not doc_txt:
                continue
            artist_match = artist_slug and (artist_slug in doc_art or doc_art in artist_slug)
            if not artist_match:
                continue
            score = 0
            if title_slug == doc_txt:
                score = 100
            elif title_slug and (title_slug in doc_txt or doc_txt in title_slug):
                score = 50
            scored.append((score, doc))
        if not scored:
            return docs[0] if docs else None
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    @staticmethod
    def _parse_lyrics(html: str) -> str | None:
        soup = BeautifulSoup(html, "lxml")
        container: Tag | None = (
            soup.select_one(".lyric-original")
            or soup.select_one(".cnt-letra-trad")
            or soup.select_one("#js-lyric-cnt")
        )
        if not isinstance(container, Tag):
            return None
        for br in container.find_all("br"):
            br.replace_with("\n")
        paragraphs = [p.get_text() for p in container.find_all("p")]
        text = "\n\n".join(paragraphs) if paragraphs else container.get_text()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text or None
