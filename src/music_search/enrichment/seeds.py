"""Geradores de sementes (queries) por tipo de entidade, a partir do corpus curado."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb

from music_search.datasets import DEFAULT_CURATED_TRACKS_PATH

DEFAULT_TRACKS_PATH = DEFAULT_CURATED_TRACKS_PATH


def _connect_view(path: Path) -> duckdb.DuckDBPyConnection:
    if not path.exists():
        raise FileNotFoundError(
            f"corpus de tracks ausente em {path}. Rode `uv run python "
            "scripts/build_curated_corpus.py` (ou apenas o passo de tracks)."
        )
    con = duckdb.connect()
    con.execute(f"CREATE VIEW tracks AS SELECT * FROM '{path.as_posix()}'")
    return con


def artist_seeds(path: Path = DEFAULT_TRACKS_PATH, limit: int | None = None) -> Iterator[str]:
    con = _connect_view(path)
    try:
        sql = """
            SELECT primary_artist_name AS name, COUNT(*) AS n
            FROM tracks
            WHERE COALESCE(primary_artist_name, '') <> ''
            GROUP BY 1
            ORDER BY n DESC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        for row in con.execute(sql).fetchall():
            yield row[0]
    finally:
        con.close()


def album_seeds(path: Path = DEFAULT_TRACKS_PATH, limit: int | None = None) -> Iterator[str]:
    con = _connect_view(path)
    try:
        sql = """
            SELECT
                album_name || ' ' || primary_artist_name AS query,
                COUNT(*) AS n
            FROM tracks
            WHERE COALESCE(album_name, '') <> ''
              AND COALESCE(primary_artist_name, '') <> ''
            GROUP BY 1
            ORDER BY n DESC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        for row in con.execute(sql).fetchall():
            yield row[0]
    finally:
        con.close()


def genre_seeds(path: Path = DEFAULT_TRACKS_PATH, limit: int | None = None) -> Iterator[str]:
    con = _connect_view(path)
    try:
        # macro_genre eh o gargalo principal para queries de gênero.
        sql = """
            SELECT macro_genre AS genre, COUNT(*) AS n
            FROM tracks
            WHERE COALESCE(macro_genre, '') <> ''
            GROUP BY 1
            ORDER BY n DESC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        seen: set[str] = set()
        for row in con.execute(sql).fetchall():
            g = row[0]
            if g and g not in seen:
                seen.add(g)
                yield g
    finally:
        con.close()


def composer_seeds(path: Path = DEFAULT_TRACKS_PATH, limit: int | None = None) -> Iterator[str]:
    """Stub: compositores normalmente nao estao no parquet de tracks.

    Por enquanto retorna artistas que tambem sao compositores conhecidos. Quando
    tiver uma fonte de compositores (ex.: ECAD, Wikidata), expanda aqui.
    """
    yield from artist_seeds(path, limit)
