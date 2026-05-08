"""Infraestrutura assincrona compartilhada: rate limit, circuit breaker, cache KV, cascade.

Originalmente vivia em `music_search.lyrics.*` e foi promovida aqui para reuso
pelos pipelines de enrichment (Wikipedia) e LLM. O codigo de letras continua
funcionando via re-export — ver `lyrics/throttle.py` e `lyrics/user_agents.py`.
"""

from music_search._async_http.cache import KeyValueCache
from music_search._async_http.pipeline import (
    CascadeConfig,
    Fetcher,
    FetchResult,
    Status,
    cascade_fetch,
)
from music_search._async_http.throttle import (
    AsyncRateLimiter,
    CircuitBreaker,
    parse_retry_after,
)
from music_search._async_http.user_agents import random_browser_headers, random_browser_ua

__all__ = [
    "AsyncRateLimiter",
    "CascadeConfig",
    "CircuitBreaker",
    "FetchResult",
    "Fetcher",
    "KeyValueCache",
    "Status",
    "cascade_fetch",
    "parse_retry_after",
    "random_browser_headers",
    "random_browser_ua",
]
