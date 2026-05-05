"""Loaders de datasets para alimentar o indexer.

A única fonte oficial do projeto é o Spotify Metadata em
`data/spotify-metadata/spotify_clean_parquet/` (ver `CLAUDE.md`). Cada track
é um documento cujos campos textuais são título, álbum e artista(s).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import duckdb

DEFAULT_PARQUET_DIR = Path("data/spotify-metadata/spotify_clean_parquet")
DEFAULT_CURATED_TRACKS_PATH = Path("data/derived/br_curated_tracks.parquet")
DEFAULT_CURATED_LYRICS_PATH = Path("data/derived/lyrics.parquet")
DEFAULT_CURATED_CORPUS_PATH = Path("data/derived/br_curated_lyrics.parquet")

FIELDS: tuple[str, ...] = ("title", "album", "artist")
CURATED_FIELDS: tuple[str, ...] = (
    "track_name",
    "artist_names",
    "artist_genres",
    "macro_genre",
    "album_name",
    "lyrics",
)


class TrackDocument(TypedDict):
    """Documento de track pronto para indexação."""

    id: str
    title: str
    album: str
    artist: str


class RichTrackDocument(TypedDict, total=False):
    """Documento de track enriquecido, usado pela indexação vetorial.

    Inclui campos textuais e numéricos adicionais (gêneros, popularidades,
    duração, data de lançamento) usados para compor o texto de embedding e
    para metadados da base vetorial.
    """

    id: str
    track_name: str
    album_name: str
    artist_names: str
    artist_genres: str
    album_type: str
    label: str
    release_date: str
    track_popularity: int
    album_popularity: int
    duration_ms: int
    explicit: bool


class CuratedLyricsDocument(TypedDict, total=False):
    """Documento consolidado do corpus curado brasileiro com letras."""

    id: str
    track_name: str
    primary_artist_name: str
    artist_names: str
    artist_genres: str
    macro_genre: str
    album_name: str
    release_date: str
    release_year: int
    track_popularity: int
    duration_ms: int
    explicit: bool
    lyrics: str
    lyrics_source: str
    lyrics_source_url: str


def build_brazilian_lyrics_corpus(
    output_path: Path = DEFAULT_CURATED_CORPUS_PATH,
    *,
    tracks_path: Path = DEFAULT_CURATED_TRACKS_PATH,
    lyrics_path: Path = DEFAULT_CURATED_LYRICS_PATH,
    cache_path: Path | None = None,
) -> Path:
    """Consolida tracks curadas + letras em um único parquet versionável.

    A fonte de letras preferida é o cache SQLite quando ele existe, porque ele
    costuma estar mais atualizado do que o parquet exportado. Se o cache não
    estiver disponível, cai para `lyrics.parquet`.
    """

    if not tracks_path.exists():
        raise FileNotFoundError(f"dataset curado ausente: {tracks_path}")

    rows = _load_curated_lyrics_rows(cache_path=cache_path, lyrics_path=lyrics_path)
    if not rows:
        raise ValueError("nenhuma letra encontrada para consolidar o corpus curado")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(
            """
            CREATE TEMP TABLE lyrics_hits (
                track_id VARCHAR,
                isrc VARCHAR,
                artist VARCHAR,
                title VARCHAR,
                source VARCHAR,
                source_url VARCHAR,
                lyrics VARCHAR
            )
            """
        )
        con.executemany("INSERT INTO lyrics_hits VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        tracks = tracks_path.as_posix()
        target = output_path.as_posix()
        con.execute(
            f"""
            COPY (
                SELECT
                    t.track_id AS id,
                    COALESCE(t.track_name, '') AS track_name,
                    COALESCE(t.primary_artist_name, '') AS primary_artist_name,
                    COALESCE(t.artist_names, '') AS artist_names,
                    COALESCE(t.artist_genres, '') AS artist_genres,
                    COALESCE(t.macro_genre, '') AS macro_genre,
                    COALESCE(t.album_name, '') AS album_name,
                    COALESCE(t.release_date, '') AS release_date,
                    COALESCE(t.release_year, 0) AS release_year,
                    COALESCE(t.track_popularity, 0) AS track_popularity,
                    COALESCE(t.duration_ms, 0) AS duration_ms,
                    COALESCE(t.explicit, FALSE) AS explicit,
                    COALESCE(h.source, '') AS lyrics_source,
                    COALESCE(h.source_url, '') AS lyrics_source_url,
                    COALESCE(h.lyrics, '') AS lyrics
                FROM '{tracks}' t
                INNER JOIN lyrics_hits h ON h.track_id = t.track_id
                WHERE COALESCE(h.lyrics, '') <> ''
            )
            TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()

    return output_path


@dataclass(frozen=True)
class BrazilianLyricsLoader:
    """Lê o corpus curado brasileiro consolidado em um doc por música."""

    corpus_path: Path = DEFAULT_CURATED_CORPUS_PATH

    def _query(self, limit: int | None = None) -> str:
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        corpus = self.corpus_path.as_posix()
        return f"""
            SELECT
                id,
                COALESCE(track_name, '') AS track_name,
                COALESCE(primary_artist_name, '') AS primary_artist_name,
                COALESCE(artist_names, '') AS artist_names,
                COALESCE(artist_genres, '') AS artist_genres,
                COALESCE(macro_genre, '') AS macro_genre,
                COALESCE(album_name, '') AS album_name,
                COALESCE(release_date, '') AS release_date,
                COALESCE(release_year, 0) AS release_year,
                COALESCE(track_popularity, 0) AS track_popularity,
                COALESCE(duration_ms, 0) AS duration_ms,
                COALESCE(explicit, FALSE) AS explicit,
                COALESCE(lyrics, '') AS lyrics,
                COALESCE(lyrics_source, '') AS lyrics_source,
                COALESCE(lyrics_source_url, '') AS lyrics_source_url
            FROM '{corpus}'
            WHERE COALESCE(lyrics, '') <> ''
            {limit_clause}
        """

    def iter_docs(self, limit: int | None = None) -> Iterator[CuratedLyricsDocument]:
        self._ensure_file_exists()
        con = duckdb.connect()
        try:
            cursor = con.execute(self._query(limit))
            while True:
                rows = cursor.fetchmany(5_000)
                if not rows:
                    break
                for row in rows:
                    yield CuratedLyricsDocument(
                        id=str(row[0]),
                        track_name=row[1] or "",
                        primary_artist_name=row[2] or "",
                        artist_names=row[3] or "",
                        artist_genres=row[4] or "",
                        macro_genre=row[5] or "",
                        album_name=row[6] or "",
                        release_date=row[7] or "",
                        release_year=int(row[8] or 0),
                        track_popularity=int(row[9] or 0),
                        duration_ms=int(row[10] or 0),
                        explicit=bool(row[11]),
                        lyrics=row[12] or "",
                        lyrics_source=row[13] or "",
                        lyrics_source_url=row[14] or "",
                    )
        finally:
            con.close()

    def count(self) -> int:
        self._ensure_file_exists()
        con = duckdb.connect()
        try:
            corpus = self.corpus_path.as_posix()
            result = con.execute(
                f"SELECT COUNT(*) FROM '{corpus}' WHERE COALESCE(lyrics, '') <> ''"
            ).fetchone()
            return int(result[0]) if result else 0
        finally:
            con.close()

    def _ensure_file_exists(self) -> None:
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"corpus curado ausente em {self.corpus_path}. "
                "Gere-o com `uv run python scripts/build_curated_corpus.py`."
            )


def _load_curated_lyrics_rows(
    *,
    cache_path: Path | None,
    lyrics_path: Path,
) -> list[tuple[str, str | None, str, str, str | None, str | None, str]]:
    cache = Path(cache_path) if cache_path else Path("data/derived/lyrics_cache.sqlite")
    if cache.exists():
        return _load_curated_lyrics_rows_from_cache(cache)
    if lyrics_path.exists():
        return _load_curated_lyrics_rows_from_parquet(lyrics_path)
    raise FileNotFoundError(
        "nenhuma fonte de letras encontrada: esperado cache SQLite ou lyrics.parquet"
    )


def _load_curated_lyrics_rows_from_cache(
    cache_path: Path,
) -> list[tuple[str, str | None, str, str, str | None, str | None, str]]:
    con = sqlite3.connect(cache_path)
    try:
        rows = con.execute(
            """
            SELECT track_id, isrc, artist, title, source, source_url, lyrics
            FROM lyrics
            WHERE status = 'hit' AND COALESCE(lyrics, '') <> ''
            """
        ).fetchall()
        return [tuple(row) for row in rows]
    finally:
        con.close()


def _load_curated_lyrics_rows_from_parquet(
    lyrics_path: Path,
) -> list[tuple[str, str | None, str, str, str | None, str | None, str]]:
    con = duckdb.connect()
    try:
        query = f"""
            SELECT track_id, isrc, artist, title, source, source_url, lyrics
            FROM '{lyrics_path.as_posix()}'
            WHERE COALESCE(lyrics, '') <> ''
        """
        rows = con.execute(query).fetchall()
        return [tuple(row) for row in rows]
    finally:
        con.close()


@dataclass(frozen=True)
class SpotifyTracksLoader:
    """Lê tracks + albums + artists dos parquets e emite um doc por track."""

    parquet_dir: Path = DEFAULT_PARQUET_DIR

    def _query(self, limit: int | None = None) -> str:
        tracks = (self.parquet_dir / "tracks.parquet").as_posix()
        albums = (self.parquet_dir / "albums.parquet").as_posix()
        artists = (self.parquet_dir / "artists.parquet").as_posix()
        track_artists = (self.parquet_dir / "track_artists.parquet").as_posix()
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        # string_agg preserva a ordem dos artistas quando há `ORDER BY`; sem
        # ordenação confiável no join, aceitamos ordem arbitrária.
        return f"""
            SELECT
                t.id AS id,
                t.name AS title,
                COALESCE(a.name, '') AS album,
                COALESCE(string_agg(ar.name, ' '), '') AS artist
            FROM '{tracks}' t
            LEFT JOIN '{albums}' a ON t.album_rowid = a.rowid
            LEFT JOIN '{track_artists}' ta ON t.rowid = ta.track_rowid
            LEFT JOIN '{artists}' ar ON ta.artist_rowid = ar.rowid
            GROUP BY t.id, t.name, a.name
            {limit_clause}
        """

    def iter_docs(self, limit: int | None = None) -> Iterator[TrackDocument]:
        self._ensure_files_exist()
        con = duckdb.connect()
        try:
            cursor = con.execute(self._query(limit))
            while True:
                rows = cursor.fetchmany(10_000)
                if not rows:
                    break
                for track_id, title, album, artist in rows:
                    yield TrackDocument(
                        id=str(track_id),
                        title=title or "",
                        album=album or "",
                        artist=artist or "",
                    )
        finally:
            con.close()

    def count(self) -> int:
        self._ensure_files_exist()
        con = duckdb.connect()
        try:
            tracks = (self.parquet_dir / "tracks.parquet").as_posix()
            result = con.execute(f"SELECT COUNT(*) FROM '{tracks}'").fetchone()
            return int(result[0]) if result else 0
        finally:
            con.close()

    def _rich_query(self, limit: int | None = None) -> str:
        tracks = (self.parquet_dir / "tracks.parquet").as_posix()
        albums = (self.parquet_dir / "albums.parquet").as_posix()
        artists = (self.parquet_dir / "artists.parquet").as_posix()
        track_artists = (self.parquet_dir / "track_artists.parquet").as_posix()
        artist_genres = (self.parquet_dir / "artist_genres.parquet").as_posix()
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        return f"""
            WITH artistas_por_track AS (
                SELECT
                    ta.track_rowid,
                    string_agg(DISTINCT ar.name,  ' | ' ORDER BY ar.name)  AS artist_names,
                    string_agg(DISTINCT ag.genre, ' | ' ORDER BY ag.genre) AS genres
                FROM '{track_artists}' ta
                LEFT JOIN '{artists}' ar       ON ar.rowid = ta.artist_rowid
                LEFT JOIN '{artist_genres}' ag ON ag.artist_rowid = ta.artist_rowid
                GROUP BY ta.track_rowid
            )
            SELECT
                t.id                        AS id,
                COALESCE(t.name, '')        AS track_name,
                COALESCE(a.name, '')        AS album_name,
                COALESCE(apt.artist_names, '') AS artist_names,
                COALESCE(apt.genres, '')    AS artist_genres,
                COALESCE(a.album_type, '')  AS album_type,
                COALESCE(a.label, '')       AS label,
                COALESCE(CAST(a.release_date AS VARCHAR), '') AS release_date,
                COALESCE(t.popularity, 0)   AS track_popularity,
                COALESCE(a.popularity, 0)   AS album_popularity,
                COALESCE(t.duration_ms, 0)  AS duration_ms,
                COALESCE(t.explicit, FALSE) AS explicit
            FROM '{tracks}' t
            LEFT JOIN '{albums}' a      ON t.album_rowid = a.rowid
            LEFT JOIN artistas_por_track apt ON apt.track_rowid = t.rowid
            {limit_clause}
        """

    def iter_rich_docs(self, limit: int | None = None) -> Iterator[RichTrackDocument]:
        """Itera documentos enriquecidos para indexação vetorial.

        Diferente de `iter_docs`, inclui gêneros, popularidades, data de
        lançamento, tipo de álbum e label — campos úteis para compor texto
        de embedding e metadados na base vetorial.
        """
        self._ensure_files_exist(rich=True)
        con = duckdb.connect()
        try:
            cursor = con.execute(self._rich_query(limit))
            while True:
                rows = cursor.fetchmany(10_000)
                if not rows:
                    break
                for row in rows:
                    yield RichTrackDocument(
                        id=str(row[0]),
                        track_name=row[1] or "",
                        album_name=row[2] or "",
                        artist_names=row[3] or "",
                        artist_genres=row[4] or "",
                        album_type=row[5] or "",
                        label=row[6] or "",
                        release_date=row[7] or "",
                        track_popularity=int(row[8] or 0),
                        album_popularity=int(row[9] or 0),
                        duration_ms=int(row[10] or 0),
                        explicit=bool(row[11]),
                    )
        finally:
            con.close()

    def _ensure_files_exist(self, rich: bool = False) -> None:
        required = ["tracks.parquet", "albums.parquet", "artists.parquet", "track_artists.parquet"]
        if rich:
            required.append("artist_genres.parquet")
        missing = [f for f in required if not (self.parquet_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"parquets ausentes em {self.parquet_dir}: {missing}. "
                "Rode `./scripts/download_spotify_metadata.sh --truncated`."
            )
