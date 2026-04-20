"""Constrói o índice invertido dos tracks do Spotify e persiste em disco.

Uso:
    uv run python scripts/build_index.py                       # todos os tracks
    uv run python scripts/build_index.py --limit 10000         # amostra
    uv run python scripts/build_index.py --output data/indexes/dev.pkl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from music_search.datasets import DEFAULT_PARQUET_DIR, FIELDS, SpotifyTracksLoader
from music_search.indexer import IndexBuilder

DEFAULT_OUTPUT = Path("data/indexes/spotify.pkl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build inverted index from Spotify parquets.")
    p.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help=f"diretório dos parquets (default: {DEFAULT_PARQUET_DIR})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"arquivo de saída do índice (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limita a quantidade de tracks indexados (útil para dev)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="imprime progresso a cada N docs (0 desativa)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loader = SpotifyTracksLoader(parquet_dir=args.parquet_dir)
    builder = IndexBuilder(fields=FIELDS)

    print(f"[build_index] lendo de {args.parquet_dir}")
    total = args.limit or loader.count()
    print(f"[build_index] indexando {total} documento(s) nos campos {FIELDS}")

    start = time.perf_counter()
    for i, doc in enumerate(loader.iter_docs(limit=args.limit), start=1):
        builder.add(doc["id"], doc)
        if args.progress_every and i % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = i / elapsed if elapsed else 0.0
            print(f"[build_index] {i}/{total} ({rate:,.0f} docs/s)")

    print("[build_index] compactando posting lists...")
    index = builder.build()
    build_time = time.perf_counter() - start

    print(f"[build_index] salvando em {args.output}")
    index.save(args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)

    print(
        f"[build_index] pronto em {build_time:.1f}s — "
        f"{index.num_docs} docs, {size_mb:.1f} MB, "
        f"vocab: " + ", ".join(f"{f}={sum(1 for _ in index.vocabulary(f))}" for f in index.fields)
    )


if __name__ == "__main__":
    main()
