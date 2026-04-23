"""Busca vetorial sobre embeddings densos armazenados no Milvus.

Complementa os modelos esparsos (TF-IDF, BM25) com recuperação por similaridade
de cosseno sobre embeddings de texto.

Para usar, instale as dependências opcionais:

    uv sync --all-groups --extra vector

Pipeline resumido:
1. `python -m music_search.vector.indexing` — gera embeddings de cada track e
   popula a coleção `spotify_tracks` no Milvus.
2. `python -m music_search.vector.search "query"` — busca semântica via CLI.
3. `from music_search.vector import VectorSearch, search_tracks` — uso como
   biblioteca.
"""

from music_search.vector.search import VectorSearch, search_tracks

__all__ = ["VectorSearch", "search_tracks"]
