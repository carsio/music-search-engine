"""Fonte: lyrics.ovh — API publica gratuita, sem autenticacao."""

from __future__ import annotations

from urllib.parse import quote

import httpx

from music_search.lyrics.sources.base import LyricsResult, Status
from music_search.lyrics.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)


class LyricsOvhSource:
    name = "lyrics_ovh"
    BASE = "https://api.lyrics.ovh/v1"

    def __init__(
        self,
        client: httpx.AsyncClient,
        timeout: float = 15.0,
        rate_limiter: AsyncRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.client = client
        self.timeout = timeout
        self.rate_limiter = rate_limiter or AsyncRateLimiter(rate=5.0)
        self.circuit_breaker = circuit_breaker or CircuitBreaker(max_failures=5, cooldown=60.0)

    async def fetch(self, artist: str, title: str) -> LyricsResult:
        if self.circuit_breaker.is_open:
            return LyricsResult(Status.BLOCKED, source=self.name, error="circuit open")

        await self.rate_limiter.acquire()
        url = f"{self.BASE}/{quote(artist)}/{quote(title)}"
        try:
            response = await self.client.get(url, timeout=self.timeout)
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="timeout", source=self.name)
        except httpx.HTTPError as exc:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error=repr(exc), source=self.name)

        if response.status_code == 404:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name, source_url=url)
        if response.status_code == 429:
            wait = parse_retry_after(response.headers.get("Retry-After"), default=5.0)
            self.rate_limiter.penalize(wait)
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.BLOCKED,
                source=self.name,
                source_url=url,
                error=f"rate limited (retry-after={wait}s)",
            )
        if response.status_code >= 500:
            self.circuit_breaker.record_failure()
            return LyricsResult(
                Status.ERROR, source=self.name, error=f"http {response.status_code}"
            )
        if response.status_code != 200:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name, error=f"http {response.status_code}")

        try:
            data = response.json()
        except ValueError:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="invalid json", source=self.name)

        lyrics = (data.get("lyrics") or "").strip()
        self.circuit_breaker.record_success()
        if not lyrics:
            return LyricsResult(Status.MISS, source=self.name, source_url=url)
        return LyricsResult(Status.HIT, lyrics=lyrics, source=self.name, source_url=url)
