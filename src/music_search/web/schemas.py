"""Pydantic schemas para a API. Espelham o contrato da UI mockup em `music search/data.jsx`."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal["artist", "album", "song", "lyric", "genre", "none", "track"]
SearchAlgorithm = Literal["bm25", "tfidf"]


class AlbumRef(BaseModel):
    title: str
    year: int | None = None
    tracks: int | None = None


class TrackRef(BaseModel):
    title: str
    album: str | None = None
    plays: str | None = None


class AlbumTrack(BaseModel):
    id: str
    title: str
    disc_number: int | None = None
    track_number: int | None = None
    duration: str | None = None
    duration_ms: int | None = None
    popularity: int | None = None
    preview_url: str | None = None
    explicit: bool = False


class AlbumArtistSummary(BaseModel):
    id: str
    name: str
    image_url: str | None = None
    genres: list[str] = Field(default_factory=list)
    popularity: int | None = None
    followers_total: int | None = None
    top_tracks: list[TrackRef] = Field(default_factory=list)
    albums: list[AlbumRef] = Field(default_factory=list)


class AlbumResponse(BaseModel):
    id: str
    title: str
    artist: str
    artist_id: str | None = None
    year: int | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    tracks_count: int | None = None
    cover_url: str | None = None
    artist_image_url: str | None = None
    album_type: str | None = None
    label: str | None = None
    duration: str | None = None
    total_duration_ms: int | None = None
    tracks: list[AlbumTrack] = Field(default_factory=list)
    artist_summary: AlbumArtistSummary


class ArtistResponse(BaseModel):
    id: str
    name: str
    tagline: str | None = None
    bio: str | None = None
    genres: list[str] = Field(default_factory=list)
    origin: str | None = None
    year_started: int | None = None
    monthly_listeners: str | None = None
    popularity: int | None = None
    albums: list[AlbumRef] = Field(default_factory=list)
    top_tracks: list[TrackRef] = Field(default_factory=list)
    source: str | None = None
    source_url: str | None = None


class SongResponse(BaseModel):
    id: str
    title: str
    artist: str
    artist_id: str | None = None
    album: str | None = None
    year: int | None = None
    duration: str | None = None
    plays: str | None = None
    composers: list[str] = Field(default_factory=list)
    lyrics: str | None = None
    lyrics_source: str | None = None
    lyrics_source_url: str | None = None
    genres: list[str] = Field(default_factory=list)
    macro_genre: str | None = None


class LyricSnippet(BaseModel):
    line: int
    text: str


class LyricMatch(BaseModel):
    song_id: str
    title: str
    artist: str
    score: float
    snippets: list[LyricSnippet] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    """Item generico no payload de /search. Forma depende do intent_used."""

    id: str
    rank: int
    score: float
    intent: str
    title: str
    subtitle: str | None = None
    snippet: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    intent_requested: Intent
    intent_used: Intent
    algorithm: SearchAlgorithm
    items: list[SearchResultItem]
    rerank_used: bool = False
    elapsed_ms: int


class LyricSearchResponse(BaseModel):
    query: str
    algorithm: SearchAlgorithm
    matches: list[LyricMatch]
    elapsed_ms: int
