"""Testes do motor de busca esparsa sobre o corpus curado."""

from __future__ import annotations

from music_search.datasets import CuratedLyricsDocument
from music_search.search import DEFAULT_FIELD_WEIGHTS, SparseSearchEngine


def _make_docs() -> list[CuratedLyricsDocument]:
    return [
        CuratedLyricsDocument(
            id="t1",
            track_name="Amor na Praia",
            primary_artist_name="Banda Sol",
            artist_names="Banda Sol",
            artist_genres="mpb | pop",
            macro_genre="mpb",
            album_name="Verão",
            release_date="2024-01-01",
            release_year=2024,
            track_popularity=75,
            duration_ms=180000,
            explicit=False,
            lyrics_source="fixture",
            lyrics_source_url="",
            lyrics="amor, calor e mar na areia",
        ),
        CuratedLyricsDocument(
            id="t2",
            track_name="Noite de Saudade",
            primary_artist_name="Dupla X",
            artist_names="Dupla X",
            artist_genres="sertanejo universitário",
            macro_genre="sertanejo",
            album_name="Madrugada",
            release_date="2023-01-01",
            release_year=2023,
            track_popularity=82,
            duration_ms=190000,
            explicit=False,
            lyrics_source="fixture",
            lyrics_source_url="",
            lyrics="saudade e madrugada no interior",
        ),
        CuratedLyricsDocument(
            id="t3",
            track_name="Canção do Interior",
            primary_artist_name="Voz do Campo",
            artist_names="Voz do Campo",
            artist_genres="folk brasileiro",
            macro_genre="folk",
            album_name="Raízes",
            release_date="2022-01-01",
            release_year=2022,
            track_popularity=60,
            duration_ms=200000,
            explicit=False,
            lyrics_source="fixture",
            lyrics_source_url="",
            lyrics="amor sertanejo na estrada de terra",
        ),
    ]


def test_sparse_search_combina_campos_e_retorna_contribuicoes() -> None:
    engine = SparseSearchEngine.build(_make_docs())

    hits = engine.search("amor", algorithm="bm25", top_k=2)

    assert [hit.id for hit in hits] == ["t1", "t3"]
    assert hits[0].field_scores
    assert hits[0].field_scores[0].weighted_score > 0


def test_sparse_search_permite_restringir_busca_a_genero() -> None:
    engine = SparseSearchEngine.build(_make_docs())

    hits = engine.search(
        "sertanejo",
        algorithm="tfidf",
        top_k=3,
        field_weights={"artist_genres": DEFAULT_FIELD_WEIGHTS["artist_genres"]},
    )

    assert [hit.id for hit in hits] == ["t2"]


def test_sparse_search_both_retorna_bm25_e_tfidf() -> None:
    engine = SparseSearchEngine.build(_make_docs())

    results = engine.search_both("saudade", top_k=2)

    assert set(results) == {"bm25", "tfidf"}
    assert results["bm25"][0].id == "t2"
    assert results["tfidf"][0].id == "t2"


def test_sparse_search_profile_muda_o_campo_priorizado() -> None:
    engine = SparseSearchEngine.build(
        [
            CuratedLyricsDocument(
                id="meta-first",
                track_name="Neblina Azul",
                primary_artist_name="Duo Mar",
                artist_names="Duo Mar",
                artist_genres="mpb",
                macro_genre="mpb",
                album_name="Bruma",
                release_date="2024-01-01",
                release_year=2024,
                track_popularity=70,
                duration_ms=180000,
                explicit=False,
                lyrics_source="fixture",
                lyrics_source_url="",
                lyrics="sol e cais ao amanhecer",
            ),
            CuratedLyricsDocument(
                id="lyrics-first",
                track_name="Horizonte",
                primary_artist_name="Duo Mar",
                artist_names="Duo Mar",
                artist_genres="mpb",
                macro_genre="mpb",
                album_name="Bruma",
                release_date="2024-01-01",
                release_year=2024,
                track_popularity=70,
                duration_ms=180000,
                explicit=False,
                lyrics_source="fixture",
                lyrics_source_url="",
                lyrics="neblina sobe no cais quando a cidade acorda",
            ),
        ]
    )

    metadata_hits = engine.search("neblina", algorithm="bm25", top_k=2, profile="metadata")
    lyrics_hits = engine.search("neblina", algorithm="bm25", top_k=2, profile="lyrics")

    assert metadata_hits[0].id == "meta-first"
    assert lyrics_hits[0].id == "lyrics-first"
