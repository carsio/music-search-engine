"""Build do dataset final commitavel. Orquestra:

1. (opcional) `scripts/build_curated_corpus.py` -> `data/derived/final/br_curated_lyrics.parquet`
2. `scripts/export_entities.py` -> `data/derived/final/br_{artist,album,genre,composer}s.parquet`
3. Manifest com versao, contagens, hashes em `data/derived/final/br_dataset_manifest.json`

Uso tipico (enquanto o download de letras ainda corre):
    uv run python scripts/build_dataset.py --skip-lyrics

Quando as letras terminarem:
    uv run python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from music_search.datasets import (
    DEFAULT_CURATED_CORPUS_PATH,
    DEFAULT_CURATED_TRACKS_PATH,
    DEFAULT_FINAL_DATASET_DIR,
    BrazilianLyricsLoader,
)

# scripts/ nao e um pacote Python: importa via path-hack para reusar export_kind.
sys.path.insert(0, str(Path(__file__).parent))
from export_entities import KINDS, export_kind  # type: ignore[import-not-found]

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
    print(">>> build_curated_corpus.py")
    res = subprocess.run(
        [sys.executable, "scripts/build_curated_corpus.py"],
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
        return int(con.execute(f"SELECT COUNT(*) FROM '{parquet.as_posix()}'").fetchone()[0])
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
