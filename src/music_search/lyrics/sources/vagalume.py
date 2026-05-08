"""Fonte: api.vagalume.com.br — focado em letras de musicas brasileiras.

Autenticacao: chave gratuita obtida em https://auth.vagalume.com.br/.
A chave entra na variavel de ambiente VAGALUME_API_KEY.
"""

from __future__ import annotations

import httpx

from music_search._async_http.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)
from music_search.lyrics.sources.base import LyricsResult, Status


class VagalumeSource:
    name = "vagalume"
    BASE = "https://api.vagalume.com.br/search.php"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | None = None,
        timeout: float = 15.0,
        rate_limiter: AsyncRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.client = client
        self.api_key = api_key
        self.timeout = timeout
        # 2 rps e o limite "padrao" gratuito; capacity um pouco maior para burst inicial.
        self.rate_limiter = rate_limiter or AsyncRateLimiter(rate=2.0, capacity=4.0)
        self.circuit_breaker = circuit_breaker or CircuitBreaker(max_failures=5, cooldown=60.0)

    async def fetch(self, artist: str, title: str) -> LyricsResult:
        if self.circuit_breaker.is_open:
            return LyricsResult(Status.BLOCKED, source=self.name, error="circuit open")

        await self.rate_limiter.acquire()
        params: dict[str, str] = {"art": artist, "mus": title}
        if self.api_key:
            params["apikey"] = self.api_key
        try:
            response = await self.client.get(self.BASE, params=params, timeout=self.timeout)
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
        if response.status_code != 200:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name, error=f"http {response.status_code}")

        try:
            data = response.json()
        except ValueError:
            self.circuit_breaker.record_failure()
            return LyricsResult(Status.ERROR, error="invalid json", source=self.name)

        rtype = (data.get("type") or "").lower()
        if "notfound" in rtype:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name)

        mus_list = data.get("mus") or []
        if not mus_list:
            self.circuit_breaker.record_success()
            return LyricsResult(Status.MISS, source=self.name)

        first = mus_list[0]
        text = (first.get("text") or "").strip()
        url = first.get("url")
        self.circuit_breaker.record_success()
        if not text:
            return LyricsResult(Status.MISS, source=self.name, source_url=url)
        return LyricsResult(Status.HIT, lyrics=text, source=self.name, source_url=url)
