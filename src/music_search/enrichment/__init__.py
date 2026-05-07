"""Enrichment de entidades (artistas, albuns, generos, compositores) via web + LLM.

Pipeline em duas etapas:
1. **Source** (ex.: Wikipedia PT) busca HTML cru por entidade.
2. **LLM** extrai um JSON estruturado (schema em `music_search.llm.prompts`).

Resultado vai para `data/derived/enrichment_cache.sqlite` (KeyValueCache).
Depois, `scripts/export_entities.py` consolida o cache em parquets versionados.
"""

from music_search.enrichment.models import (
    AlbumDocument,
    ArtistDocument,
    ComposerDocument,
    EntityKind,
    GenreDocument,
)

__all__ = [
    "AlbumDocument",
    "ArtistDocument",
    "ComposerDocument",
    "EntityKind",
    "GenreDocument",
]
