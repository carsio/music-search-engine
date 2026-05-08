"""Geradores de sementes (queries) por tipo de entidade, a partir do corpus curado."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import duckdb

from music_search.core.preprocessing import normalize
from music_search.data.datasets import DEFAULT_CURATED_TRACKS_PATH

DEFAULT_TRACKS_PATH = DEFAULT_CURATED_TRACKS_PATH

GenreSeedMode = Literal["expanded", "macro"]

_GENRE_SEED_SYNONYMS: dict[str, str] = {
    "agronejo": "agronejo",
    "arrocha": "arrocha",
    "axe": "axé",
    "bolero": "bolero",
    "boom bap": "boom bap",
    "bossa nova": "bossa nova",
    "brazilian evangelical music": "música gospel",
    "brazilian funk": "funk carioca",
    "brazilian hip hop": "hip hop brasileiro",
    "brazilian jazz": "jazz brasileiro",
    "brazilian phonk": "phonk brasileiro",
    "brazilian pop": "pop brasileiro",
    "brazilian rock": "rock brasileiro",
    "brazilian trap": "trap brasileiro",
    "brega": "brega",
    "brega funk": "brega funk",
    "calypso": "calypso",
    "forro": "forró",
    "forro tradicional": "forró",
    "funk": "funk",
    "funk carioca": "funk carioca",
    "funk consciente": "funk consciente",
    "funk de bh": "funk de bh",
    "funk melody": "funk melody",
    "gospel": "música gospel",
    "jazz": "jazz",
    "mpb": "mpb",
    "nova mpb": "nova mpb",
    "pagode": "pagode",
    "pagode baiano": "pagode baiano",
    "pentecostal": "música gospel",
    "phonk": "phonk",
    "piseiro": "piseiro",
    "pop": "pop",
    "pop rock": "pop rock",
    "rap": "rap",
    "reggae": "reggae",
    "roots reggae": "reggae",
    "rock": "rock",
    "samba": "samba",
    "seresta": "seresta",
    "sertanejo": "sertanejo",
    "sertanejo tradicional": "sertanejo tradicional",
    "sertanejo universitario": "sertanejo universitário",
    "tecnobrega": "tecnobrega",
    "trap": "trap",
    "trap funk": "trap funk",
    "worship": "música gospel",
}


def _connect_view(path: Path) -> duckdb.DuckDBPyConnection:
    if not path.exists():
        raise FileNotFoundError(
            f"corpus de tracks ausente em {path}. Rode `uv run python -m "
            "music_search.scripts.build_curated_corpus` (ou apenas o passo de tracks)."
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


def _iter_artist_genres(value: str) -> Iterator[str]:
    for genre in value.split(" | "):
        cleaned = genre.strip()
        if cleaned:
            yield cleaned


def _canonical_genre_seed(raw_genre: str) -> str | None:
    key = normalize(raw_genre)
    if not key:
        return None
    return _GENRE_SEED_SYNONYMS.get(key)


def _macro_genre_seeds(path: Path = DEFAULT_TRACKS_PATH, limit: int | None = None) -> Iterator[str]:
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


def _expanded_genre_seeds(path: Path = DEFAULT_TRACKS_PATH, limit: int | None = None) -> Iterator[str]:
    con = _connect_view(path)
    try:
        counts: Counter[str] = Counter()
        cursor = con.execute(
            """
            SELECT COALESCE(artist_genres, '') AS artist_genres
            FROM tracks
            WHERE COALESCE(artist_genres, '') <> ''
            """
        )
        while True:
            rows = cursor.fetchmany(5_000)
            if not rows:
                break
            for (artist_genres,) in rows:
                per_track = {
                    canonical
                    for raw_genre in _iter_artist_genres(artist_genres)
                    if (canonical := _canonical_genre_seed(raw_genre)) is not None
                }
                counts.update(per_track)

        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if limit is not None:
            ordered = ordered[: int(limit)]
        for genre, _ in ordered:
            yield genre
    finally:
        con.close()


def genre_seeds(
    path: Path = DEFAULT_TRACKS_PATH,
    limit: int | None = None,
    *,
    seed_mode: GenreSeedMode = "expanded",
) -> Iterator[str]:
    if seed_mode == "macro":
        yield from _macro_genre_seeds(path, limit)
        return
    yield from _expanded_genre_seeds(path, limit)


def composer_seeds(path: Path = DEFAULT_TRACKS_PATH, limit: int | None = None) -> Iterator[str]:
    """Stub: compositores normalmente nao estao no parquet de tracks.

    Por enquanto retorna artistas que tambem sao compositores conhecidos. Quando
    tiver uma fonte de compositores (ex.: ECAD, Wikidata), expanda aqui.
    """
    yield from artist_seeds(path, limit)
