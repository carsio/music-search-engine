from __future__ import annotations

import asyncio
from pathlib import Path

import duckdb

from music_search._async_http.cache import KeyValueCache
from music_search.enrichment.pipeline import (
    DEFAULT_CACHE_TABLE,
    RAW_FETCH_STATUS,
    EnrichmentConfig,
    _build_raw_payload,
    make_key,
    run_enrichment,
)


def test_run_enrichment_materializes_artist_payload_without_llm(tmp_path: Path) -> None:
    cache_path = tmp_path / "enrichment.sqlite"
    query = "Gilberto Gil"
    source_url = "https://pt.wikipedia.org/wiki/Gilberto_Gil"

    cache = KeyValueCache(cache_path, table=DEFAULT_CACHE_TABLE)
    cache.upsert(
        key=make_key("artist", query),
        kind="artist",
        status=RAW_FETCH_STATUS,
        source="wikipedia_pt",
        source_url=source_url,
        payload=_build_raw_payload(
            (
                "Resumo\nGilberto Gil e cantor e compositor brasileiro.\n\n"
                "## Carreira\nGravou discos marcantes."
            ),
            query=query,
            source_url=source_url,
        ),
        attempts=1,
    )
    cache.close()

    stats = asyncio.run(
        run_enrichment(
            "artist",
            [query],
            EnrichmentConfig(
                cache_path=cache_path,
                fetch_documents=False,
                normalize_documents=True,
            ),
        )
    )

    assert stats == {"hit": 1}

    cache = KeyValueCache(cache_path, table=DEFAULT_CACHE_TABLE)
    try:
        row = cache.get(make_key("artist", query))
        assert row is not None
        assert row["status"] == "hit"

        payload = row["payload"]
        assert payload["name"] == "Gilberto Gil"
        assert payload["tagline"] == "Gilberto Gil e cantor e compositor brasileiro."
        assert payload["bio"] == "Gilberto Gil e cantor e compositor brasileiro."
        assert payload["raw_text"] == (
            "Gilberto Gil e cantor e compositor brasileiro.\n\n"
            "Carreira. Gravou discos marcantes."
        )
        assert payload["source"] == "wikipedia_pt"
        assert payload["source_url"] == source_url
    finally:
        cache.close()


def test_export_kind_preserves_raw_text_column(tmp_path: Path) -> None:
    from music_search.scripts.export_entities import export_kind

    cache_path = tmp_path / "enrichment.sqlite"
    output = tmp_path / "br_genres.parquet"

    cache = KeyValueCache(cache_path, table=DEFAULT_CACHE_TABLE)
    cache.upsert(
        key=make_key("genre", "tropicalia"),
        kind="genre",
        status="hit",
        source="wikipedia_pt",
        source_url="https://pt.wikipedia.org/wiki/Tropic%C3%A1lia",
        payload={
            "id": "tropicalia",
            "name": "Tropicália",
            "description": "Movimento cultural brasileiro.",
            "raw_text": (
                "Movimento cultural brasileiro.\n\n"
                "Contexto. Mistura musica e artes visuais."
            ),
            "origin": None,
            "decade": None,
            "representative_artists": [],
            "related_genres": [],
            "source": "wikipedia_pt",
            "source_url": "https://pt.wikipedia.org/wiki/Tropic%C3%A1lia",
        },
        attempts=1,
    )
    cache.close()

    export_kind("genre", cache_path=cache_path, output=output)

    con = duckdb.connect()
    try:
        row = con.execute(
            f"SELECT id, name, description, raw_text, source, source_url FROM '{output.as_posix()}'"
        ).fetchone()
    finally:
        con.close()

    assert row == (
        "tropicalia",
        "Tropicália",
        "Movimento cultural brasileiro.",
        "Movimento cultural brasileiro.\n\nContexto. Mistura musica e artes visuais.",
        "wikipedia_pt",
        "https://pt.wikipedia.org/wiki/Tropic%C3%A1lia",
    )
