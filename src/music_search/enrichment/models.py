"""Schemas de entidades enriquecidas. Espelham o contrato da UI mockup (`music search/data.jsx`).

Mantemos como `TypedDict` (forma plana, serializavel) e nao como dataclass — a saida
da LLM e dict, e os parquets gerados por `export_entities.py` tambem.
"""

from __future__ import annotations

from typing import Literal, TypedDict

EntityKind = Literal["artist", "album", "genre", "composer"]


class AlbumRef(TypedDict, total=False):
    title: str
    year: int | None
    tracks: int | None


class TrackRef(TypedDict, total=False):
    title: str
    album: str | None
    plays: str | None


class ArtistDocument(TypedDict, total=False):
    id: str
    name: str
    tagline: str | None
    bio: str | None
    genres: list[str]
    origin: str | None
    year_started: int | None
    monthly_listeners: str | None
    popularity: int | None
    color: str | None
    albums: list[AlbumRef]
    top_tracks: list[TrackRef]
    source: str
    source_url: str | None


class AlbumTrack(TypedDict, total=False):
    position: int | None
    title: str
    duration: str | None


class AlbumDocument(TypedDict, total=False):
    id: str
    title: str
    artist: str
    year: int | None
    description: str | None
    tracks: list[AlbumTrack]
    source: str
    source_url: str | None


class GenreDocument(TypedDict, total=False):
    id: str
    name: str
    description: str | None
    origin: str | None
    decade: str | None
    representative_artists: list[str]
    related_genres: list[str]
    source: str
    source_url: str | None


class NotableWork(TypedDict, total=False):
    title: str
    year: int | None
    performer: str | None


class ComposerDocument(TypedDict, total=False):
    id: str
    name: str
    bio: str | None
    genres: list[str]
    origin: str | None
    year_started: int | None
    notable_works: list[NotableWork]
    source: str
    source_url: str | None
