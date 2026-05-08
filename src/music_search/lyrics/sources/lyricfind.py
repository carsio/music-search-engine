"""Fonte: LyricFind (lyrics.lyricfind.com) — scraping HTML, sem chave.

A LyricFind opera o player publico em `https://lyrics.lyricfind.com/`, alem do
portal corporativo `www.lyricfind.com`. O fluxo aqui e parecido com o de
`letras.mus.br`:

1. Busca via endpoint publico de search (`https://lyrics.lyricfind.com/api/v1/search`)
   que devolve JSON com candidatos (`{"tracks": {"docs": [{"slug": ..., "artist": {...}}]}}`).
   Quando esse endpoint nao responde, cai para a pagina HTML de busca em
   `https://lyrics.lyricfind.com/search?q=...` e extrai o primeiro link de letra.
2. Constroi a URL da pagina (`/lyrics/<slug>`) e parseia a letra do container
   `[data-testid="lyrics"]` / `.lyrics-body` / `.song-body` (com fallbacks).

Anti-blocking igual aos outros providers: token bucket conservador (1 rps, burst 2),
circuit breaker (3 falhas → 120s), pool de User-Agents reais e respeito a `Retry-After`.

Observacao: como a LyricFind nao publica spec dos endpoints publicos, os seletores
e o shape do JSON podem mudar. Se comecar a vir MISS em massa, rode
`python -m music_search.lyrics probe "<artista>" "<musica>"` e ajuste os seletores
em `_parse_lyrics` ou o caminho do JSON em `_extract_match`.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

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


class LyricFindSource:
    name = "lyricfind"
    BASE = "https://lyrics.lyricfind.com"
    SEARCH_API = "https://lyrics.lyricfind.com/api/v1/search"

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

        # 1. busca para descobrir o slug da musica
        slug = await self._find_slug(artist, title)
        if isinstance(slug, LyricsResult):
            # _find_slug ja registrou ERROR/BLOCKED apropriadamente
            return slug
        if not slug:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name)

        url = f"{self.BASE}/lyrics/{slug}"

        # 2. baixa pagina HTML
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

    async def _find_slug(self, artist: str, title: str) -> str | None | LyricsResult:
        """Tenta API JSON; em caso de falha cai para scraping da pagina de search.

        Devolve `str` (slug encontrado), `None` (nada bateu — MISS) ou um
        `LyricsResult` ja preenchido (ERROR/BLOCKED para o caller propagar).
        """
        await self.rate_limiter.acquire()
        api_headers = {
            "Accept": "application/json",
            "Referer": f"{self.BASE}/",
            **random_browser_headers(referer=f"{self.BASE}/"),
        }
        try:
            resp = await self.client.get(
                self.SEARCH_API,
                params={"q": f"{artist} {title}", "limit": 5, "type": "track"},
                timeout=self.timeout,
                headers=api_headers,
            )
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="search timeout", source=self.name)
        except httpx.HTTPError as exc:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error=repr(exc), source=self.name)

        if resp.status_code == 429:
            wait = parse_retry_after(resp.headers.get("Retry-After"), default=15.0)
            self.rate_limiter.penalize(wait)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                error=f"search rate limited (retry-after={wait}s)",
            )

        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                data = None
            if isinstance(data, dict):
                slug = self._extract_slug(data, artist, title)
                if slug:
                    return slug
            # API respondeu 200 mas nao bateu: cai para HTML de search

        # Fallback: scrape da pagina de busca HTML
        return await self._search_html(artist, title)

    async def _search_html(self, artist: str, title: str) -> str | None | LyricsResult:
        await self.rate_limiter.acquire()
        url = f"{self.BASE}/search?q={quote_plus(f'{artist} {title}')}"
        headers = random_browser_headers(referer=f"{self.BASE}/")
        try:
            resp = await self.client.get(url, timeout=self.timeout, headers=headers)
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="search-html timeout", source=self.name)
        except httpx.HTTPError as exc:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error=repr(exc), source=self.name)

        if resp.status_code == 429:
            wait = parse_retry_after(resp.headers.get("Retry-After"), default=30.0)
            self.rate_limiter.penalize(wait)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                error=f"search-html rate limited (retry-after={wait}s)",
            )
        if resp.status_code in (403, 503):
            self.rate_limiter.penalize(60.0)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED, source=self.name, error=f"search-html http {resp.status_code}"
            )
        if resp.status_code == 404:
            self.circuit_breaker.record_success()
            return None
        if resp.status_code != 200:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, source=self.name, error=f"search-html http {resp.status_code}"
            )

        return self._extract_slug_from_html(resp.text, artist, title)

    @staticmethod
    def _extract_slug(data: dict[str, Any], artist: str, title: str) -> str | None:
        """Encontra o melhor slug nos resultados JSON.

        O endpoint costuma devolver algo como
        `{"tracks": {"docs": [{"slug": "...", "title": "...", "artist": {"name": "..."}}, ...]}}`,
        mas variantes existem (`{"results": [...]}`, `{"data": {"tracks": [...]}}`). A gente
        tenta varios caminhos.
        """
        candidates: list[dict[str, Any]] = []
        for path in (
            ("tracks", "docs"),
            ("tracks",),
            ("results",),
            ("data", "tracks"),
            ("docs",),
            ("hits",),
        ):
            cur: Any = data
            ok = True
            for key in path:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, list):
                candidates = [c for c in cur if isinstance(c, dict)]
                if candidates:
                    break
        if not candidates:
            return None

        artist_slug = slugify(artist)
        title_slug = slugify(title)

        scored: list[tuple[int, str]] = []
        for doc in candidates:
            slug = doc.get("slug") or doc.get("track_slug") or doc.get("url")
            if isinstance(slug, str):
                slug = slug.strip("/").removeprefix("lyrics/")
            if not isinstance(slug, str) or not slug:
                continue

            doc_artist = ""
            artist_field = doc.get("artist")
            if isinstance(artist_field, dict):
                doc_artist = artist_field.get("name", "") or ""
            elif isinstance(artist_field, str):
                doc_artist = artist_field
            doc_title = doc.get("title") or doc.get("name") or ""

            doc_artist_slug = slugify(doc_artist)
            doc_title_slug = slugify(str(doc_title))

            score = 0
            if artist_slug and (artist_slug in doc_artist_slug or doc_artist_slug in artist_slug):
                score += 50
            if title_slug == doc_title_slug:
                score += 100
            elif title_slug and (title_slug in doc_title_slug or doc_title_slug in title_slug):
                score += 50
            scored.append((score, slug))

        if not scored:
            return None
        scored.sort(key=lambda x: -x[0])
        # Exige um casamento minimo (artista OR titulo) para nao pegar lixo do top do search.
        if scored[0][0] < 50:
            return None
        return scored[0][1]

    def _extract_slug_from_html(
        self, html: str, artist: str, title: str
    ) -> str | None | LyricsResult:
        soup = BeautifulSoup(html, "lxml")
        artist_slug = slugify(artist)
        title_slug = slugify(title)

        best: tuple[int, str] | None = None
        for a in soup.find_all("a", href=True):
            if not isinstance(a, Tag):
                continue
            href = a.get("href", "")
            if not isinstance(href, str) or "/lyrics/" not in href:
                continue
            slug = href.split("/lyrics/", 1)[1].split("?", 1)[0].split("#", 1)[0].strip("/")
            if not slug:
                continue
            link_text = slugify(a.get_text(" ", strip=True))
            score = 0
            if artist_slug and artist_slug in link_text:
                score += 50
            if title_slug and title_slug in link_text:
                score += 50
            if best is None or score > best[0]:
                best = (score, slug)
        if best is None:
            self.circuit_breaker.record_success()
            return None
        if best[0] < 50:
            self.circuit_breaker.record_success()
            return None
        return best[1]

    @staticmethod
    def _parse_lyrics(html: str) -> str | None:
        soup = BeautifulSoup(html, "lxml")
        container: Tag | None = None
        for selector in (
            '[data-testid="lyrics"]',
            ".lyrics-body",
            ".song-body",
            "#lyrics-body-text",
            ".lyrics",
            "pre.lyrics",
        ):
            found = soup.select_one(selector)
            if isinstance(found, Tag):
                container = found
                break
        if container is None:
            return None

        for br in container.find_all("br"):
            br.replace_with("\n")
        paragraphs = [p.get_text() for p in container.find_all("p")]
        text = "\n\n".join(paragraphs) if paragraphs else container.get_text()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text or None
