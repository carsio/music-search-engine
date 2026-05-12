"""Enrichment de entidades (artistas, albuns, generos, compositores) a partir da Wikipedia PT.

Pipeline em duas etapas:
1. **Fetch** baixa o conteudo cru da Wikipedia PT por entidade.
2. **Materialize** extrai o payload estruturado localmente (parsing do texto + heuristicas).

Resultado vai para `data/derived/enrichment_cache.sqlite` (KeyValueCache).
Depois, ``music_search.scripts.export_entities`` consolida o cache em parquets versionados.
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
