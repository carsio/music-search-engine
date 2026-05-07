"""Mecanismos anti-bloqueio: token bucket assincrono e circuit breaker.

Cada fonte tem um par (`AsyncRateLimiter`, `CircuitBreaker`) que limita a taxa de
requisicoes e desliga a fonte temporariamente apos falhas consecutivas. O objetivo
e nao apanhar 429 / WAF nem queimar nossa cota nas APIs gratuitas.
"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Token bucket assincrono. Ate `rate` requisicoes/seg com burst de `capacity`.

    `acquire()` espera (asyncio.sleep) quando o bucket esta vazio. Multiplas corotinas
    podem competir pelos tokens; o `_lock` garante consistencia.

    `penalize(seconds)` drena o bucket o suficiente para forcar uma pausa de pelo
    menos `seconds` antes da proxima requisicao bem sucedida — usado quando o servidor
    devolve `Retry-After`.
    """

    def __init__(self, rate: float, capacity: float | None = None):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = float(rate)
        self.capacity = float(capacity) if capacity else max(self.rate, 1.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            await asyncio.sleep(wait)

    def penalize(self, seconds: float) -> None:
        """Forca pausa de pelo menos `seconds` antes da proxima requisicao."""
        if seconds <= 0:
            return
        self._tokens = -seconds * self.rate
        self._last = time.monotonic()


class CircuitBreaker:
    """Abre apos `max_failures` falhas consecutivas; fecha apos `cooldown` segundos.

    Enquanto aberto, `is_open` retorna True e a fonte deve devolver BLOCKED sem bater
    na rede. Sucesso zera o contador.
    """

    def __init__(self, max_failures: int = 5, cooldown: float = 60.0):
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        self.max_failures = max_failures
        self.cooldown = float(cooldown)
        self._consecutive_failures = 0
        self._open_until: float | None = None

    @property
    def is_open(self) -> bool:
        if self._open_until is None:
            return False
        if time.monotonic() >= self._open_until:
            self._open_until = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._open_until = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_failures:
            self._open_until = time.monotonic() + self.cooldown


def parse_retry_after(value: str | None, default: float = 5.0, cap: float = 60.0) -> float:
    """Parseia o cabecalho HTTP `Retry-After`.

    Aceita segundos (int) ou data HTTP. Retorna `default` em caso de ausencia/parse falho,
    com teto em `cap` segundos para nao deixar workers parados muito tempo.
    """
    if not value:
        return default
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        return default
    return min(max(seconds, 0.0), cap)
