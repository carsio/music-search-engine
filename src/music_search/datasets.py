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

    def _ensure_files_exist(self) -> None:
        required = ["tracks.parquet", "albums.parquet", "artists.parquet", "track_artists.parquet"]
        missing = [f for f in required if not (self.parquet_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"parquets ausentes em {self.parquet_dir}: {missing}. "
                "Rode `./scripts/download_spotify_metadata.sh --truncated`."
            )
