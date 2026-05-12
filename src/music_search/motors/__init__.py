"""Motores de busca que aplicam o core RI sobre o corpus curado.

`search` expõe o `SparseSearchEngine` (BM25/TF-IDF multi-campo com boosts);
`multi_index` orquestra busca por entidades (tracks, artistas, álbuns, gêneros)
com roteamento por intent; `tuning` define perfis de busca e pesos por campo.
"""
