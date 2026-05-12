"""Fonte: LRCLib (lrclib.net) — API JSON publica, gratuita, sem chave.

LRCLib e uma base de letras (sincronizadas e simples) mantida pela comunidade,
usada por varios players (Spicetify, Lyrica, etc.). A API e direta:

    GET https://lrclib.net/api/get?artist_name=...&track_name=...

Retorna 200 com JSON {plainLyrics, syncedLyrics, ...} ou 404 quando nao acha.
Quando o /get nao acerta o casamento exato (artista/titulo escritos de forma
ligeiramente diferente da base), a gente cai para /api/search que e mais permissivo
e devolve uma lista ordenada por relevancia.

Sem rate limit publicado — usar moderado (4 rps default) e honrar 429.
"""

from __future__ import annotations

import re

import httpx

from music_search._async_http.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)
from music_search.lyrics.sources.base import LyricsResult, Status

_TIMESTAMP_RE = re.compile(r"^\s*\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]\s*")


class LrcLibSource:
    name = "lrclib"
    BASE = "https://lrclib.net/api"

    def __init__(
        self,
        client: httpx.AsyncClient,
        timeout: float = 15.0,
        rate_limiter: AsyncRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.client = client
        self.timeout = timeout
        self.rate_limiter = rate_limiter or AsyncRateLimiter(rate=4.0, capacity=8.0)
        self.circuit_breaker = circuit_breaker or CircuitBreaker(max_failures=5, cooldown=60.0)

    async def fetch(self, artist: str, title: str) -> LyricsResult:
        if self.circuit_breaker.is_open:
            return LyricsResult(Status.BLOCKED, source=self.name, error="circuit open")

        # 1. tentativa de match exato via /get
        await self.rate_limiter.acquire()
        get_url = f"{self.BASE}/get"
        params = {"artist_name": artist, "track_name": title}
        try:
            response = await self.client.get(get_url, params=params, timeout=self.timeout)
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="timeout", source=self.name)
        except httpx.HTTPError as exc:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error=repr(exc), source=self.name)

        if response.status_code == 429:
            wait = parse_retry_after(response.headers.get("Retry-After"), default=10.0)
            self.rate_limiter.penalize(wait)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                error=f"rate limited (retry-after={wait}s)",
            )
        if response.status_code >= 500:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, source=self.name, error=f"http {response.status_code}"
            )

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                self.circuit_breaker.record_failure()
                return LyricsResult(Status.ERROR, error="invalid json", source=self.name)
            lyrics = self._extract_lyrics(data)
            if lyrics:
                self.circuit_breaker.record_success()
                return LyricsResult(
                    Status.HIT,
                    lyrics=lyrics,
                    source=self.name,
                    source_url=self._track_url(data),
                )
            # /get bateu mas devolveu vazio (instrumental, sem letra) — tenta /search
        elif response.status_code != 404:
            # status estranho (403, etc.) — registra mas tenta /search assim mesmo
            self.circuit_breaker.record_failure()

        # 2. fallback /search (mais tolerante a variacoes)
        return await self._search(artist, title)

    async def _search(self, artist: str, title: str) -> LyricsResult:
        await self.rate_limiter.acquire()
        url = f"{self.BASE}/search"
        params = {"artist_name": artist, "track_name": title}
        try:
            response = await self.client.get(url, params=params, timeout=self.timeout)
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="search timeout", source=self.name)
        except httpx.HTTPError as exc:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error=repr(exc), source=self.name)

        if response.status_code == 429:
            wait = parse_retry_after(response.headers.get("Retry-After"), default=10.0)
            self.rate_limiter.penalize(wait)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                error=f"search rate limited (retry-after={wait}s)",
            )
        if response.status_code >= 500:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, source=self.name, error=f"search http {response.status_code}"
            )
        if response.status_code != 200:
            self.circuit_breaker.record_success()
            return LyricsResult(
                Status.MISS,
                source=self.name,
                error=f"search http {response.status_code}",
            )

        try:
            data = response.json()
        except ValueError:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="invalid search json", source=self.name)

        if not isinstance(data, list) or not data:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name)

        # primeiro resultado costuma ser o mais relevante; se tiver letra usavel, retorna.
        for entry in data[:5]:
            lyrics = self._extract_lyrics(entry)
            if lyrics:
                self.circuit_breaker.record_success()
                return LyricsResult(
                    Status.HIT,
                    lyrics=lyrics,
                    source=self.name,
                    source_url=self._track_url(entry),
                )

        self.circuit_breaker.record_success()
        return LyricsResult(Status.MISS, source=self.name)

    @classmethod
    def _extract_lyrics(cls, data: dict) -> str | None:
        if not isinstance(data, dict):
            return None
        if data.get("instrumental"):
            return None
        plain = (data.get("plainLyrics") or "").strip()
        if plain:
            return plain
        synced = (data.get("syncedLyrics") or "").strip()
        if synced:
            return cls._strip_timestamps(synced)
        return None

    @staticmethod
    def _strip_timestamps(synced: str) -> str:
        """Converte LRC sincronizado em texto puro removendo `[mm:ss.xx]`."""
        out: list[str] = []
        for raw in synced.splitlines():
            cleaned = _TIMESTAMP_RE.sub("", raw).strip()
            if cleaned:
                out.append(cleaned)
        return "\n".join(out).strip()

    @staticmethod
    def _track_url(data: dict) -> str | None:
        track_id = data.get("id")
        if track_id is None:
            return None
        return f"https://lrclib.net/api/get/{track_id}"
