"""Build do dataset final commitavel.

Orquestra (todos relativos a ``data/derived/final/``):

1. (opcional) ``music_search.scripts.build_curated_corpus`` ->
   ``br_curated_lyrics.parquet``
2. ``music_search.scripts.export_entities`` ->
   ``br_{artist,album,genre,composer}s.parquet``
3. Manifest com versao, contagens e hashes em ``br_dataset_manifest.json``.

Uso tipico (enquanto o download de letras ainda corre):
    uv run python -m music_search.scripts.build_dataset --skip-lyrics

Quando as letras terminarem:
    uv run python -m music_search.scripts.build_dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from music_search.data.datasets import (
    DEFAULT_CURATED_CORPUS_PATH,
    DEFAULT_CURATED_TRACKS_PATH,
    DEFAULT_FINAL_DATASET_DIR,
    BrazilianLyricsLoader,
)
from music_search.scripts.export_entities import KINDS, export_kind

DATASET_VERSION = "0.3.0"
OUTPUT_DIR = DEFAULT_FINAL_DATASET_DIR


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path) -> dict | None:
    if not path.exists():
        return None
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha1": _sha1(path),
    }


def _run_lyrics_corpus() -> None:
    import sys

    print(">>> music_search.scripts.build_curated_corpus")
    res = subprocess.run(
        [sys.executable, "-m", "music_search.scripts.build_curated_corpus"],
        check=False,
    )
    if res.returncode != 0:
        raise SystemExit("build_curated_corpus falhou")


def _count_records(parquet: Path) -> int:
    if not parquet.exists():
        return 0
    import duckdb

    con = duckdb.connect()
    try:
        row = con.execute(f"SELECT COUNT(*) FROM '{parquet.as_posix()}'").fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Builda o dataset final commitavel.")
    parser.add_argument(
        "--skip-lyrics",
        action="store_true",
        help="Nao re-roda build_curated_corpus.py (use enquanto letras ainda baixam).",
    )
    parser.add_argument(
        "--skip-entities",
        action="store_true",
        help="Nao re-exporta entidades do enrichment_cache.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_lyrics:
        _run_lyrics_corpus()
    else:
        print("[skip] build_curated_corpus (--skip-lyrics)")

    if not args.skip_entities:
        for k in KINDS:
            try:
                export_kind(k)
            except FileNotFoundError as exc:
                print(f"[{k}] {exc}")
    else:
        print("[skip] export_entities (--skip-entities)")

    # Coleta de info para o manifesto
    files: dict[str, dict | None] = {
        "curated_tracks": _file_info(DEFAULT_CURATED_TRACKS_PATH),
        "curated_lyrics": _file_info(DEFAULT_CURATED_CORPUS_PATH),
    }
    for k in KINDS:
        files[f"{k}s"] = _file_info(args.output_dir / f"br_{k}s.parquet")

    counts = {
        "curated_tracks": _count_records(DEFAULT_CURATED_TRACKS_PATH),
        "curated_lyrics": (
            BrazilianLyricsLoader(DEFAULT_CURATED_CORPUS_PATH).count()
            if DEFAULT_CURATED_CORPUS_PATH.exists()
            else 0
        ),
    }
    for k in KINDS:
        counts[f"{k}s"] = _count_records(args.output_dir / f"br_{k}s.parquet")

    manifest = {
        "version": DATASET_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "files": files,
        "counts": counts,
    }
    manifest_path = args.output_dir / "br_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
