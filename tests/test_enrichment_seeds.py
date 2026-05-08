"""Testes das seeds do pipeline de enrichment."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

from music_search.enrichment import __main__ as enrichment_cli
from music_search.enrichment.seeds import genre_seeds


def _write_tracks_parquet(path: Path) -> None:
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM (
                    VALUES
                        ('mpb_bossa_choro', 'mpb | bossa nova | jazz'),
                        ('mpb_bossa_choro', 'mpb | nova mpb | bossa nova'),
                        ('gospel', 'gospel | brazilian evangelical music | worship'),
                        ('gospel', 'pentecostal | gospel'),
                        ('gospel', 'gospel'),
                        ('funk', 'brazilian funk | funk carioca | funk')
                ) AS t(macro_genre, artist_genres)
            )
            TO '{path.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def test_genre_seeds_expandidas_canonicalizam_artist_genres(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.parquet"
    _write_tracks_parquet(tracks_path)

    assert list(genre_seeds(tracks_path, seed_mode="expanded")) == [
        "música gospel",
        "bossa nova",
        "mpb",
        "funk",
        "funk carioca",
        "jazz",
        "nova mpb",
    ]


def test_genre_seeds_expandidas_aplicam_limit_apos_ordenar(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.parquet"
    _write_tracks_parquet(tracks_path)

    assert list(genre_seeds(tracks_path, limit=3, seed_mode="expanded")) == [
        "música gospel",
        "bossa nova",
        "mpb",
    ]


def test_genre_seeds_macro_preservam_taxonomia_curada(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.parquet"
    _write_tracks_parquet(tracks_path)

    assert list(genre_seeds(tracks_path, seed_mode="macro")) == [
        "gospel",
        "mpb_bossa_choro",
        "funk",
    ]


def test_cli_usa_seed_mode_expanded_por_padrao_para_genres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    async def fake_run(args) -> None:
        captured["kind"] = args.kind
        captured["seed_mode"] = args.seed_mode

    monkeypatch.setattr(enrichment_cli, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["music_search.enrichment", "genres"])

    enrichment_cli.main()

    assert captured == {"kind": "genres", "seed_mode": "expanded"}


def test_cli_permte_forcar_seed_mode_macro_para_genres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    async def fake_run(args) -> None:
        captured["kind"] = args.kind
        captured["seed_mode"] = args.seed_mode

    monkeypatch.setattr(enrichment_cli, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["music_search.enrichment", "genres", "--seed-mode", "macro"],
    )

    enrichment_cli.main()

    assert captured == {"kind": "genres", "seed_mode": "macro"}


def test_cli_aceita_phase_materialize(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    async def fake_run(args) -> None:
        captured["kind"] = args.kind
        captured["phase"] = args.phase

    monkeypatch.setattr(enrichment_cli, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["music_search.enrichment", "artists", "--phase", "materialize"],
    )

    enrichment_cli.main()

    assert captured == {"kind": "artists", "phase": "materialize"}
