from __future__ import annotations

import duckdb

from music_search.albums import AlbumDocument
from music_search.datasets import CuratedLyricsDocument
from music_search.multi_index import EntityIndex, MultiEntityIndex, _load_or_build_entity_index
from music_search.search import SparseSearchEngine


def _make_track_engine() -> SparseSearchEngine:
    docs = [
        CuratedLyricsDocument(
            id="t1",
            track_name="Faixa do Amor",
            primary_artist_name="Artista X",
            artist_names="Artista X",
            artist_genres="mpb",
            macro_genre="mpb",
            album_name="Paisagens",
            release_date="2024-01-01",
            release_year=2024,
            track_popularity=88,
            duration_ms=200000,
            explicit=False,
            lyrics_source="fixture",
            lyrics_source_url="",
            lyrics="amor amor amor na areia",
        ),
        CuratedLyricsDocument(
            id="t2",
            track_name="Noite no Sertao",
            primary_artist_name="Artista Y",
            artist_names="Artista Y",
            artist_genres="forro",
            macro_genre="forro",
            album_name="Lua do Norte",
            release_date="2023-01-01",
            release_year=2023,
            track_popularity=55,
            duration_ms=180000,
            explicit=False,
            lyrics_source="fixture",
            lyrics_source_url="",
            lyrics="noite fria no sertao",
        ),
    ]
    return SparseSearchEngine.build(docs)


def _make_album_catalog() -> dict[str, AlbumDocument]:
    return {
        "album-1": {
            "id": "album-1",
            "title": "Raizes do Norte",
            "artist": "Artista A",
            "artist_id": "artist-a",
            "year": 2024,
            "description": (
                "Raizes do Norte: Album de Artista A; lançado em 2024; "
                "com 2 faixas; em torno de mpb, samba."
            ),
            "tags": ["mpb", "samba"],
            "tracks_count": 2,
            "cover_url": "https://example.test/album.jpg",
            "artist_image_url": "https://example.test/artist.jpg",
            "album_type": "album",
            "label": "Selo Azul",
            "total_duration_ms": 380000,
            "duration": "6:20",
            "tracks": [],
            "artist_summary": {
                "id": "artist-a",
                "name": "Artista A",
                "image_url": "https://example.test/artist.jpg",
                "genres": ["mpb", "samba"],
                "popularity": 84,
                "followers_total": 5000,
                "top_tracks": [],
                "albums": [],
            },
        }
    }


def test_search_routed_prefere_album_quando_album_e_melhor_candidato() -> None:
    multi = MultiEntityIndex.from_parquets(
        track_engine=_make_track_engine(),
        parquets={},
        album_catalog=_make_album_catalog(),
    )

    result = multi.search_routed("Raizes do Norte", "artist", algorithm="bm25", top_k=3)

    assert result["intent_used"] == "album"
    assert result["hits"][0]["id"] == "album-1"


def test_search_routed_mantem_track_quando_track_domina() -> None:
    multi = MultiEntityIndex.from_parquets(
        track_engine=_make_track_engine(),
        parquets={},
        album_catalog=_make_album_catalog(),
    )

    result = multi.search_routed("amor na areia", "lyric", algorithm="bm25", top_k=3)

    assert result["intent_used"] == "track"
    assert result["hits"][0]["id"] == "t1"


def test_search_entity_considera_raw_text_com_peso_baixo() -> None:
    genre_index = EntityIndex.build(
        "genre",
        [
            {
                "id": "g1",
                "name": "Tropicália",
                "description": "",
                "raw_text": "Movimento psicodelico da musica brasileira.",
                "origin": None,
                "representative_artists": [],
            }
        ],
    )
    multi = MultiEntityIndex(track_engine=None, entity_indexes={"genre": genre_index})

    hits = multi.search_entity("genre", "psicodelico", algorithm="bm25", top_k=3)

    assert hits[0].id == "g1"


def test_load_or_build_entity_index_reusa_cache_persistido(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "artists.parquet"
    cache_path = tmp_path / "entity_artist.pkl"
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT
                    'artist-1' AS id,
                    'Gilberto Gil' AS name,
                    'Cantor brasileiro' AS tagline,
                    'Cantor brasileiro' AS bio,
                    'Cantor brasileiro' AS raw_text,
                    ['mpb'] AS genres,
                    'Salvador' AS origin
            ) TO '{source_path.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()

    cached = EntityIndex.build(
        "artist",
        [
            {
                "id": "artist-1",
                "name": "Gilberto Gil",
                "tagline": "Cantor brasileiro",
                "bio": "Cantor brasileiro",
                "raw_text": "Cantor brasileiro",
                "genres": ["mpb"],
                "origin": "Salvador",
            }
        ],
    )
    cached.save(cache_path)

    def _fail_build(_cls, kind, records, **kwargs):
        raise AssertionError(f"não deveria reconstruir cache para {kind}")

    monkeypatch.setattr(EntityIndex, "build", classmethod(_fail_build))

    loaded = _load_or_build_entity_index(
        "artist",
        [{"id": "artist-1", "name": "Outro nome"}],
        source_path=source_path,
        index_path=cache_path,
    )

    assert loaded.documents["artist-1"]["name"] == "Gilberto Gil"
