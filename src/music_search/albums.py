"""Catalogo de albuns derivado do parquet curado de tracks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TypedDict
import unicodedata

import duckdb

from music_search.datasets import DEFAULT_CURATED_TRACKS_PATH


class AlbumTrackRef(TypedDict):
    """Faixa derivada do dataset de tracks para uso na pagina de album."""

    id: str
    title: str
    disc_number: int
    track_number: int
    duration_ms: int
    duration: str | None
    popularity: int | None
    preview_url: str | None
    explicit: bool


class ArtistTrackRef(TypedDict):
    """Resumo leve de musica para sidebar do artista."""

    id: str
    title: str
    album: str | None
    plays: str | None


class ArtistAlbumRef(TypedDict):
    """Resumo leve de album para sidebar do artista."""

    id: str
    title: str
    year: int | None
    tracks: int | None


class AlbumArtistSummary(TypedDict):
    """Resumo do artista principal derivado do mesmo dataset de tracks."""

    id: str
    name: str
    image_url: str | None
    genres: list[str]
    popularity: int | None
    followers_total: int | None
    top_tracks: list[ArtistTrackRef]
    albums: list[ArtistAlbumRef]


class AlbumDocument(TypedDict):
    """Documento agregado de album usado por busca e detalhe."""

    id: str
    title: str
    artist: str
    artist_id: str | None
    year: int | None
    description: str
    tags: list[str]
    tracks_count: int
    cover_url: str | None
    artist_image_url: str | None
    album_type: str | None
    label: str | None
    total_duration_ms: int
    duration: str | None
    tracks: list[AlbumTrackRef]
    artist_summary: AlbumArtistSummary


class _TrackRow(TypedDict):
    track_id: str
    track_name: str
    preview_url: str
    track_number: int
    disc_number: int
    primary_artist_id: str
    primary_artist_name: str
    primary_artist_followers_total: int
    primary_artist_popularity: int
    artist_genres: str
    macro_genre: str
    album_id: str
    album_name: str
    album_type: str
    album_label: str
    album_total_tracks: int
    release_year: int
    track_popularity: int
    duration_ms: int
    explicit: bool
    album_image_url: str
    primary_artist_image_url: str


@dataclass
class _ArtistAccumulator:
    id: str
    name: str
    image_url: str | None = None
    genres: set[str] = field(default_factory=set)
    popularity: int = 0
    followers_total: int = 0
    tracks: dict[str, ArtistTrackRef] = field(default_factory=dict)
    albums: dict[str, ArtistAlbumRef] = field(default_factory=dict)


def load_album_catalog_from_tracks(
    tracks_path: Path = DEFAULT_CURATED_TRACKS_PATH,
) -> dict[str, AlbumDocument]:
    """Agrupa o parquet curado por album e devolve um catalogo navegavel."""
    rows = _load_track_rows(tracks_path)
    if not rows:
        return {}

    rows_by_album: dict[str, list[_TrackRow]] = defaultdict(list)
    artists: dict[str, _ArtistAccumulator] = {}

    for row in rows:
        artist_key = _artist_key(row)
        artist = artists.setdefault(
            artist_key,
            _ArtistAccumulator(
                id=row["primary_artist_id"] or artist_key,
                name=row["primary_artist_name"] or "Artista desconhecido",
            ),
        )
        artist.image_url = artist.image_url or _none_if_blank(row["primary_artist_image_url"])
        artist.popularity = max(artist.popularity, row["primary_artist_popularity"])
        artist.followers_total = max(artist.followers_total, row["primary_artist_followers_total"])
        artist.genres.update(_collect_tags((row,)))
        track_ref: ArtistTrackRef = {
            "id": row["track_id"],
            "title": row["track_name"],
            "album": _none_if_blank(row["album_name"]),
            "plays": _format_play_count(row["track_popularity"]),
        }
        current_track = artist.tracks.get(row["track_id"])
        if current_track is None or _track_ref_sort_key(track_ref) < _track_ref_sort_key(current_track):
            artist.tracks[row["track_id"]] = track_ref

        album_id = _album_id(row)
        rows_by_album[album_id].append(row)

    catalog: dict[str, AlbumDocument] = {}
    for album_id, group in rows_by_album.items():
        exemplar = _choose_album_exemplar(group)
        tracks = [_album_track_from_row(row) for row in _sort_album_tracks(group)]
        tags = sorted(_collect_tags(group))
        year = _first_positive_int(row["release_year"] for row in group)
        tracks_count = _album_tracks_count(group, tracks)
        total_duration_ms = sum(track["duration_ms"] for track in tracks)
        artist_key = _artist_key(exemplar)
        artist = artists[artist_key]
        artist.albums[album_id] = {
            "id": album_id,
            "title": exemplar["album_name"],
            "year": year,
            "tracks": tracks_count,
        }
        catalog[album_id] = {
            "id": album_id,
            "title": exemplar["album_name"],
            "artist": exemplar["primary_artist_name"],
            "artist_id": _none_if_blank(exemplar["primary_artist_id"]),
            "year": year,
            "description": _build_album_description(
                title=exemplar["album_name"],
                artist=exemplar["primary_artist_name"],
                year=year,
                tracks_count=tracks_count,
                tags=tags,
            ),
            "tags": tags,
            "tracks_count": tracks_count,
            "cover_url": _none_if_blank(exemplar["album_image_url"]),
            "artist_image_url": _none_if_blank(exemplar["primary_artist_image_url"]),
            "album_type": _none_if_blank(exemplar["album_type"]),
            "label": _none_if_blank(exemplar["album_label"]),
            "total_duration_ms": total_duration_ms,
            "duration": _format_duration(total_duration_ms),
            "tracks": tracks,
            "artist_summary": _empty_artist_summary(),
        }

    for album in catalog.values():
        artist = artists[_artist_key_from_values(album.get("artist_id"), album["artist"])]
        album["artist_summary"] = _artist_summary_from_accumulator(artist)

    return catalog


def build_album_search_records(catalog: Iterable[AlbumDocument]) -> list[dict[str, object]]:
    """Reduz o catalogo para o payload enxuto usado no indice de busca."""
    records: list[dict[str, object]] = []
    for album in catalog:
        records.append(
            {
                "id": album["id"],
                "title": album["title"],
                "artist": album["artist"],
                "artist_id": album.get("artist_id"),
                "year": album.get("year"),
                "description": album.get("description"),
                "tracks_count": album.get("tracks_count"),
                "tags": album.get("tags", []),
                "cover_url": album.get("cover_url"),
                "artist_image_url": album.get("artist_image_url"),
                "album_type": album.get("album_type"),
                "label": album.get("label"),
            }
        )
    return records


def _load_track_rows(tracks_path: Path) -> list[_TrackRow]:
    if not tracks_path.exists():
        return []
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
            SELECT
                COALESCE(track_id, '') AS track_id,
                COALESCE(track_name, '') AS track_name,
                COALESCE(preview_url, '') AS preview_url,
                COALESCE(track_number, 0) AS track_number,
                COALESCE(disc_number, 0) AS disc_number,
                COALESCE(primary_artist_id, '') AS primary_artist_id,
                COALESCE(primary_artist_name, '') AS primary_artist_name,
                COALESCE(primary_artist_followers_total, 0) AS primary_artist_followers_total,
                COALESCE(primary_artist_popularity, 0) AS primary_artist_popularity,
                COALESCE(artist_genres, '') AS artist_genres,
                COALESCE(macro_genre, '') AS macro_genre,
                COALESCE(album_id, '') AS album_id,
                COALESCE(album_name, '') AS album_name,
                COALESCE(album_type, '') AS album_type,
                COALESCE(album_label, '') AS album_label,
                COALESCE(album_total_tracks, 0) AS album_total_tracks,
                COALESCE(release_year, 0) AS release_year,
                COALESCE(track_popularity, 0) AS track_popularity,
                COALESCE(duration_ms, 0) AS duration_ms,
                COALESCE(explicit, FALSE) AS explicit,
                COALESCE(album_image_url, '') AS album_image_url,
                COALESCE(primary_artist_image_url, '') AS primary_artist_image_url
            FROM '{tracks_path.as_posix()}'
            WHERE COALESCE(album_name, '') <> ''
              AND COALESCE(primary_artist_name, '') <> ''
            """
        ).fetchall()
    finally:
        con.close()

    loaded: list[_TrackRow] = []
    for row in rows:
        loaded.append(
            _TrackRow(
                track_id=str(row[0]),
                track_name=str(row[1]),
                preview_url=str(row[2]),
                track_number=int(row[3] or 0),
                disc_number=int(row[4] or 0),
                primary_artist_id=str(row[5]),
                primary_artist_name=str(row[6]),
                primary_artist_followers_total=int(row[7] or 0),
                primary_artist_popularity=int(row[8] or 0),
                artist_genres=str(row[9]),
                macro_genre=str(row[10]),
                album_id=str(row[11]),
                album_name=str(row[12]),
                album_type=str(row[13]),
                album_label=str(row[14]),
                album_total_tracks=int(row[15] or 0),
                release_year=int(row[16] or 0),
                track_popularity=int(row[17] or 0),
                duration_ms=int(row[18] or 0),
                explicit=bool(row[19]),
                album_image_url=str(row[20]),
                primary_artist_image_url=str(row[21]),
            )
        )
    return loaded


def _choose_album_exemplar(rows: list[_TrackRow]) -> _TrackRow:
    return max(
        rows,
        key=lambda row: (
            1 if row["album_image_url"] else 0,
            row["album_total_tracks"],
            row["release_year"],
            row["track_popularity"],
            row["track_name"],
        ),
    )


def _sort_album_tracks(rows: Iterable[_TrackRow]) -> list[_TrackRow]:
    return sorted(
        rows,
        key=lambda row: (
            _sort_pos(row["disc_number"]),
            _sort_pos(row["track_number"]),
            -row["track_popularity"],
            row["track_name"],
            row["track_id"],
        ),
    )


def _album_track_from_row(row: _TrackRow) -> AlbumTrackRef:
    duration_ms = row["duration_ms"]
    popularity = row["track_popularity"] or None
    return {
        "id": row["track_id"],
        "title": row["track_name"],
        "disc_number": row["disc_number"],
        "track_number": row["track_number"],
        "duration_ms": duration_ms,
        "duration": _format_duration(duration_ms),
        "popularity": popularity,
        "preview_url": _none_if_blank(row["preview_url"]),
        "explicit": row["explicit"],
    }


def _album_tracks_count(rows: Iterable[_TrackRow], tracks: list[AlbumTrackRef]) -> int:
    declared = max((row["album_total_tracks"] for row in rows), default=0)
    return declared or len(tracks)


def _collect_tags(rows: Iterable[_TrackRow]) -> set[str]:
    tags: set[str] = set()
    for row in rows:
        tags.update(_split_terms(row["artist_genres"]))
        tags.update(_split_terms(row["macro_genre"]))
    return {tag for tag in tags if tag}


def _build_album_description(
    *,
    title: str,
    artist: str,
    year: int | None,
    tracks_count: int,
    tags: list[str],
) -> str:
    parts = [f"Album de {artist}"]
    if year is not None:
        parts.append(f"lançado em {year}")
    if tracks_count > 0:
        parts.append(f"com {tracks_count} faixas")
    if tags:
        parts.append(f"em torno de {', '.join(tags[:3])}")
    return f"{title}: {'; '.join(parts)}."


def _artist_summary_from_accumulator(artist: _ArtistAccumulator) -> AlbumArtistSummary:
    albums = sorted(
        artist.albums.values(),
        key=lambda album: (-(album.get("year") or 0), album["title"].lower(), album["id"]),
    )
    top_tracks = sorted(artist.tracks.values(), key=_track_ref_sort_key)[:5]
    return {
        "id": artist.id,
        "name": artist.name,
        "image_url": artist.image_url,
        "genres": sorted(artist.genres),
        "popularity": artist.popularity or None,
        "followers_total": artist.followers_total or None,
        "top_tracks": top_tracks,
        "albums": albums[:6],
    }


def _empty_artist_summary() -> AlbumArtistSummary:
    return {
        "id": "",
        "name": "",
        "image_url": None,
        "genres": [],
        "popularity": None,
        "followers_total": None,
        "top_tracks": [],
        "albums": [],
    }


def _album_id(row: _TrackRow) -> str:
    if row["album_id"]:
        return row["album_id"]
    return _fallback_album_id(row["album_name"], row["primary_artist_name"], row["release_year"])


def _artist_key(row: _TrackRow) -> str:
    return _artist_key_from_values(row["primary_artist_id"], row["primary_artist_name"])


def _artist_key_from_values(artist_id: str | None, artist_name: str) -> str:
    if artist_id:
        return artist_id
    return f"artist-{_slug(artist_name)}"


def _fallback_album_id(album_name: str, artist_name: str, year: int) -> str:
    suffix = str(year) if year > 0 else "unknown"
    return f"album-{_slug(album_name)}-{_slug(artist_name)}-{suffix}"


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
    return collapsed or "unknown"


def _split_terms(value: str) -> list[str]:
    pieces = [part.strip() for part in re.split(r"[|,]", value) if part.strip()]
    return pieces


def _first_positive_int(values: Iterable[int]) -> int | None:
    positives = [value for value in values if value > 0]
    return min(positives) if positives else None


def _none_if_blank(value: str) -> str | None:
    return value or None


def _format_duration(duration_ms: int) -> str | None:
    if duration_ms <= 0:
        return None
    seconds = duration_ms // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


def _format_play_count(popularity: int) -> str | None:
    if popularity <= 0:
        return None
    return f"pop {popularity}"


def _track_ref_sort_key(track: ArtistTrackRef) -> tuple[int, str, str]:
    plays = track.get("plays") or ""
    popularity = int(plays.removeprefix("pop ")) if plays.startswith("pop ") else 0
    return (-popularity, track["title"].lower(), track["id"])


def _sort_pos(value: int) -> int:
    return value if value > 0 else 10_000