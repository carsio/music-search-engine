from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from music_search.albums import AlbumDocument
from music_search.multi_index import EntityIndex, MultiEntityIndex
from music_search.web.app import _album_response_from_payload, app, get_album, search, search_lyric


class _TrackEngineStub:
    def search(self, *_args, **_kwargs):
        return []


def _album_catalog() -> dict[str, AlbumDocument]:
    return {
        "album-1": {
            "id": "album-1",
            "title": "Raizes do Norte",
            "artist": "Artista A",
            "artist_id": "artist-a",
            "year": 2024,
            "description": "Resumo do album.",
            "tags": ["mpb", "samba"],
            "tracks_count": 2,
            "cover_url": "https://example.test/album.jpg",
            "artist_image_url": "https://example.test/artist.jpg",
            "album_type": "album",
            "label": "Selo Azul",
            "total_duration_ms": 380000,
            "duration": "6:20",
            "tracks": [
                {
                    "id": "t1",
                    "title": "Faixa Um",
                    "disc_number": 1,
                    "track_number": 1,
                    "duration_ms": 200000,
                    "duration": "3:20",
                    "popularity": 90,
                    "preview_url": "https://example.test/preview.mp3",
                    "explicit": False,
                }
            ],
            "artist_summary": {
                "id": "artist-a",
                "name": "Artista A",
                "image_url": "https://example.test/artist.jpg",
                "genres": ["mpb", "samba"],
                "popularity": 84,
                "followers_total": 5000,
                "top_tracks": [
                    {
                        "id": "t1",
                        "title": "Faixa Um",
                        "album": "Raizes do Norte",
                        "plays": "pop 90",
                    }
                ],
                "albums": [
                    {"id": "album-1", "title": "Raizes do Norte", "year": 2024, "tracks": 2}
                ],
            },
        }
    }


def test_album_response_from_payload_converte_sidebar_e_tracklist() -> None:
    response = _album_response_from_payload(_album_catalog()["album-1"])

    assert response.id == "album-1"
    assert response.artist_summary.name == "Artista A"
    assert response.tracks[0].title == "Faixa Um"
    assert response.artist_summary.top_tracks[0].plays == "pop 90"


def test_get_album_usa_catalogo_agregado() -> None:
    app.state.multi = MultiEntityIndex(
        track_engine=None,
        entity_indexes={},
        album_catalog=_album_catalog(),
    )

    response = get_album("album-1")

    assert response.title == "Raizes do Norte"
    assert response.cover_url == "https://example.test/album.jpg"


def test_get_album_retorna_404_quando_ausente() -> None:
    app.state.multi = MultiEntityIndex(track_engine=None, entity_indexes={}, album_catalog={})

    with pytest.raises(HTTPException) as exc:
        get_album("missing")

    assert exc.value.status_code == 404


def test_get_artist_aceita_payload_minimo_wikipedia_only() -> None:
    track_engine = _TrackEngineStub()
    artist_index = EntityIndex.build(
        "artist",
        [
            {
                "id": "gilberto-gil",
                "name": "Gilberto Gil",
                "tagline": "Cantor e compositor brasileiro.",
                "bio": "Cantor e compositor brasileiro.",
                "raw_text": "Cantor e compositor brasileiro.\n\nCarreira. Gravou discos marcantes.",
                "genres": [],
                "origin": None,
                "albums": [],
                "source": "wikipedia_pt",
                "source_url": "https://pt.wikipedia.org/wiki/Gilberto_Gil",
            }
        ],
    )
    app.state.track_engine = track_engine
    app.state.multi = MultiEntityIndex(
        track_engine=track_engine,
        entity_indexes={"artist": artist_index},
        album_catalog={},
    )

    from music_search.web.app import get_artist

    response = get_artist("gilberto-gil")

    assert response.name == "Gilberto Gil"
    assert response.bio == "Cantor e compositor brasileiro."
    assert response.top_tracks == []
    assert response.source_url == "https://pt.wikipedia.org/wiki/Gilberto_Gil"


@pytest.mark.anyio
async def test_search_encaminha_parametros_avancados_ao_multi_index() -> None:
    class _MultiStub:
        def __init__(self) -> None:
            self.kwargs = {}

        def search_routed(self, *_args, **kwargs):
            self.kwargs = kwargs
            return {
                "intent_used": "track",
                "hits": [
                    {
                        "id": "track-1",
                        "kind": "track",
                        "rank": 1,
                        "score": 1.0,
                        "track_name": "Neblina Azul",
                        "artist_names": "Duo Mar",
                        "lyrics_preview": "neblina no cais",
                    }
                ],
            }

    app.state.multi = _MultiStub()
    app.state.nim_client = None
    app.state.llm_cache = None

    response = await search(
        q="neblina",
        top=7,
        algorithm="tfidf",
        rerank=False,
        profile="metadata",
        bm25_k1=2.2,
        bm25_b=0.4,
        tf_scheme="raw",
    )

    assert response.algorithm == "tfidf"
    assert response.items[0].id == "track-1"
    assert app.state.multi.kwargs == {
        "algorithm": "tfidf",
        "top_k": 7,
        "profile": "metadata",
        "bm25_k1": 2.2,
        "bm25_b": 0.4,
        "tf_scheme": "raw",
    }


def test_search_lyric_respeita_max_snippets_e_parametros_avancados() -> None:
    class _TrackEngineStub:
        def __init__(self) -> None:
            self.kwargs = {}

        def search(self, *_args, **kwargs):
            self.kwargs = kwargs
            return [
                SimpleNamespace(
                    id="track-1",
                    track_name="Neblina Azul",
                    artist_names="Duo Mar",
                    primary_artist_name="Duo Mar",
                    score=1.0,
                    lyrics="Neblina no cais\nNoite sem farol\nNeblina volta cedo\nCidade acorda lenta",
                )
            ]

    app.state.track_engine = _TrackEngineStub()

    response = search_lyric(
        q="neblina",
        top=5,
        algorithm="bm25",
        profile="lyrics",
        bm25_k1=2.0,
        bm25_b=0.3,
        tf_scheme="augmented",
        max_snippets=2,
    )

    assert len(response.matches[0].snippets) == 2
    assert app.state.track_engine.kwargs == {
        "algorithm": "bm25",
        "top_k": 5,
        "profile": "lyrics",
        "bm25_k1": 2.0,
        "bm25_b": 0.3,
        "tf_scheme": "augmented",
    }
