"""Re-export fino de `music_search._async_http.throttle`.

A logica original mora em `_async_http/throttle.py` para ser reusada por enrichment
(Wikipedia) e LLM. Este modulo permanece como ponto de import dos sources de letras
para nao quebrar o pipeline ja rodando.
"""

from music_search._async_http.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)

__all__ = ["AsyncRateLimiter", "CircuitBreaker", "parse_retry_after"]
