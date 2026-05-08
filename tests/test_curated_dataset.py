from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb

from music_search.datasets import BrazilianLyricsLoader, build_brazilian_lyrics_corpus


def _write_tracks_parquet(path: Path) -> None:
    con = duckdb.connect()
    try:
        target = path.as_posix()
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM (
                    VALUES
                        (
                            'trk-1', 'BR001', TRUE, 'Coração Brasileiro', 'Artista A', 'Artista A',
                            'mpb | samba', 'mpb', 'alb-1', 'Canções do Brasil', '2024-01-01',
                            2024, 2020, 80, 210000, FALSE
                        ),
                        (
                            'trk-2', 'BR002', TRUE, 'Rock do Sul', 'Artista B', 'Artista B',
                            'rock gaucho', 'rock', 'alb-2', 'Noites Frias', '2023-02-02',
                            2023, 2020, 70, 180000, FALSE
                        )
                ) AS t(
                    track_id, isrc, isrc_br, track_name, primary_artist_name, artist_names,
                    artist_genres, macro_genre, album_id, album_name, release_date,
                    release_year, decade, track_popularity, duration_ms, explicit
                )
            )
            TO '{target}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def _write_cache(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            CREATE TABLE lyrics (
                track_id TEXT PRIMARY KEY,
                isrc TEXT,
                artist TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT,
                source_url TEXT,
                lyrics TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 1,
                fetched_at INTEGER NOT NULL
            );
            """
        )
        con.execute(
            """
            INSERT INTO lyrics (
                track_id, isrc, artist, title, status, source, source_url,
                lyrics, error, attempts, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "trk-1",
                "BR001",
                "Artista A",
                "Coração Brasileiro",
                "hit",
                "fixture",
                "https://example.test/letra",
                "Coração brasileiro no compasso do samba",
                None,
                1,
                1,
            ),
        )
        con.commit()
    finally:
        con.close()


def test_build_brazilian_lyrics_corpus_join_tracks_e_cache(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.parquet"
    cache = tmp_path / "lyrics.sqlite"
    output = tmp_path / "curated.parquet"
    _write_tracks_parquet(tracks)
    _write_cache(cache)

    result = build_brazilian_lyrics_corpus(
        output_path=output,
        tracks_path=tracks,
        cache_path=cache,
        lyrics_path=tmp_path / "unused.parquet",
    )

    assert result == output
    loader = BrazilianLyricsLoader(output)
    docs = list(loader.iter_docs())
    assert loader.count() == 1
    assert docs[0]["id"] == "trk-1"
    assert docs[0]["track_name"] == "Coração Brasileiro"
    assert docs[0]["artist_genres"] == "mpb | samba"
    assert docs[0]["lyrics_source"] == "fixture"
    assert "samba" in docs[0]["lyrics"]
