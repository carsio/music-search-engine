"""Cascade fetcher generico: tenta uma sequencia de fontes ate uma devolver HIT.

Espelha o pipeline de letras (`lyrics/pipeline.py:_process_track`) sem o acoplamento
ao dominio de musicas (artist/title/variants). Aqui o item e generico — pode ser
um artista para enriquecer via Wikipedia, ou qualquer outra fonte externa.

Para preservar o comportamento existente do pipeline de letras (que tem logica
especifica de variantes de titulo), `lyrics/pipeline.py` nao foi modificado e
continua usando sua propria implementacao.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)


class Status(enum.StrEnum):
    """Status terminais de uma tentativa de fetch.

    HIT     — fonte respondeu com dados validos.
    MISS    — fonte respondeu definitivamente "nao tenho".
    BLOCKED — circuit breaker / rate limit / 429-503; nao reten ta nesta fonte agora.
    ERROR   — transitorio (timeout, 5xx, parse). Pode ser re-tentado.
    """

    HIT = "hit"
    MISS = "miss"
    BLOCKED = "blocked"
    ERROR = "error"


T = TypeVar("T")


@dataclass
class FetchResult[T]:
    status: Status
    payload: T | None = None
    source: str | None = None
    source_url: str | None = None
    error: str | None = None


@runtime_checkable
class Fetcher(Protocol[T]):
    """Contrato de uma fonte cascateavel.

    `fetch(item)` recebe o item generico (ex.: dict com seed) e retorna o resultado.
    Implementacoes concretas devem internalizar rate limit / circuit breaker.
    """

    name: str

    async def fetch(self, item: Any) -> FetchResult[T]: ...


@dataclass
class CascadeConfig:
    concurrency: int = 8
    request_timeout: float = 15.0
    max_retries: int = 2
    retry_initial_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_jitter: float = 0.4
    retry_errors: bool = False
    retry_misses: bool = False
    retry_blocked: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


async def _retry_one[T](
    source: Fetcher[T],
    item: Any,
    cfg: CascadeConfig,
) -> tuple[FetchResult[T], int]:
    """Reten ta `cfg.max_retries + 1` vezes em caso de ERROR transitorio.

    Status terminais (HIT, MISS, BLOCKED) retornam imediatamente.
    """
    delay = cfg.retry_initial_delay
    last: FetchResult[T] | None = None
    attempts = 0
    for attempt in range(cfg.max_retries + 1):
        attempts += 1
        result = await source.fetch(item)
        last = result
        if result.status in (Status.HIT, Status.MISS, Status.BLOCKED):
            return result, attempts
        if attempt >= cfg.max_retries:
            break
        sleep_for = delay + random.uniform(0.0, cfg.retry_jitter)
        await asyncio.sleep(sleep_for)
        delay *= cfg.retry_backoff
    assert last is not None
    return last, attempts


def serialize_trace(trace: list[dict]) -> str | None:
    """Serializa o trace como JSON compacto."""
    if not trace:
        return None
    try:
        return json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


async def cascade_fetch[T](
    item: Any,
    sources: list[Fetcher[T]],
    cfg: CascadeConfig,
    *,
    on_progress: Callable[[str, FetchResult[T], int], None] | None = None,
) -> tuple[FetchResult[T], list[dict], int]:
    """Tenta cada fonte em ordem ate uma devolver HIT.

    Retorna (resultado_final, trace, attempts_total). O trace lista cada tentativa
    para debug/visualizacao.

    Apenas HIT e BLOCKED interrompem o cascade dentro de uma fonte (BLOCKED so para
    aquela fonte; passa para a proxima). MISS / ERROR cascateiam.
    """
    trace: list[dict] = []
    last: FetchResult[T] | None = None
    attempts_total = 0

    for source in sources:
        started = time.time()
        result, attempts = await _retry_one(source, item, cfg)
        elapsed_ms = int((time.time() - started) * 1000)
        attempts_total += attempts
        last = result
        trace.append(
            {
                "ts": int(started),
                "source": source.name,
                "status": result.status.value,
                "source_url": result.source_url,
                "error": result.error,
                "attempts": attempts,
                "elapsed_ms": elapsed_ms,
            }
        )
        if on_progress:
            on_progress(source.name, result, attempts)
        if result.status == Status.HIT:
            return result, trace, attempts_total

    assert last is not None
    return last, trace, attempts_total
