"""Constrói o índice invertido e persiste em disco.

Uso:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --dataset spotify
    uv run python scripts/build_index.py --limit 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from music_search.datasets import (
    CURATED_FIELDS,
    DEFAULT_CURATED_CORPUS_PATH,
    DEFAULT_PARQUET_DIR,
    FIELDS,
    BrazilianLyricsLoader,
    SpotifyTracksLoader,
)
from music_search.indexer import IndexBuilder
from music_search.search import DEFAULT_INDEX_PATH

DEFAULT_SPOTIFY_OUTPUT = Path("data/indexes/spotify.pkl")
DATASET_CHOICES = ("curated-br", "spotify")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build inverted index from supported datasets.")
    p.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="curated-br",
        help="dataset a indexar (default: curated-br)",
    )
    p.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help=f"diretório dos parquets do Spotify (default: {DEFAULT_PARQUET_DIR})",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CURATED_CORPUS_PATH,
        help=f"parquet consolidado do corpus curado (default: {DEFAULT_CURATED_CORPUS_PATH})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="arquivo de saída do índice (default depende do dataset)",
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
    if args.dataset == "spotify":
        loader = SpotifyTracksLoader(parquet_dir=args.parquet_dir)
        fields = FIELDS
        source_label = str(args.parquet_dir)
        output = args.output or DEFAULT_SPOTIFY_OUTPUT
    else:
        loader = BrazilianLyricsLoader(corpus_path=args.corpus)
        fields = CURATED_FIELDS
        source_label = str(args.corpus)
        output = args.output or DEFAULT_INDEX_PATH

    builder = IndexBuilder(fields=fields)

    print(f"[build_index] dataset={args.dataset}")
    print(f"[build_index] lendo de {source_label}")
    total = args.limit or loader.count()
    print(f"[build_index] indexando {total} documento(s) nos campos {fields}")

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

    print(f"[build_index] salvando em {output}")
    index.save(output)
    size_mb = output.stat().st_size / (1024 * 1024)

    print(
        f"[build_index] pronto em {build_time:.1f}s — "
        f"{index.num_docs} docs, {size_mb:.1f} MB, "
        f"vocab: " + ", ".join(f"{f}={sum(1 for _ in index.vocabulary(f))}" for f in index.fields)
    )


if __name__ == "__main__":
    main()
