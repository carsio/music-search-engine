"""Protocol e tipos para fontes de enrichment."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from music_search._async_http.pipeline import FetchResult, Status
from music_search.enrichment.models import EntityKind


class EnrichmentItem(dict):
    """Seed para uma fonte. Contem `kind`, `query` e opcional `context`.

    Mantido como dict (e nao TypedDict) para facilitar o uso com cascade_fetch
    que recebe `Any`.
    """

    @classmethod
    def make(cls, kind: EntityKind, query: str, **context: object) -> EnrichmentItem:
        return cls(kind=kind, query=query, context=context)


@runtime_checkable
class EnrichmentSource(Protocol):
    """Fonte que retorna conteudo cru para uma entidade.

    O conteudo (texto/HTML) e depois passado para o materializador local que
    extrai um payload estruturado (parsing + heuristicas, sem LLM).
    """

    name: str

    async def fetch(self, item: dict) -> FetchResult[str]: ...


__all__ = ["EnrichmentItem", "EnrichmentSource", "FetchResult", "Status"]
