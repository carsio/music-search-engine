"""Loaders de datasets para alimentar o indexer.

A única fonte oficial do projeto é o Spotify Metadata em
`data/spotify-metadata/spotify_clean_parquet/` (ver `CLAUDE.md`). Cada track
é um documento cujos campos textuais são título, álbum e artista(s).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import duckdb

DEFAULT_PARQUET_DIR = Path("data/spotify-metadata/spotify_clean_parquet")

FIELDS: tuple[str, ...] = ("title", "album", "artist")


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
