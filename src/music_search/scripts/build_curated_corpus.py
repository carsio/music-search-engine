"""Gera o parquet consolidado do corpus curado brasileiro com letras.

Uso:
    uv run python -m music_search.scripts.build_curated_corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

from music_search.data.datasets import (
    DEFAULT_CURATED_CORPUS_PATH,
    DEFAULT_CURATED_LYRICS_PATH,
    DEFAULT_CURATED_TRACKS_PATH,
    BrazilianLyricsLoader,
    build_brazilian_lyrics_corpus,
)

DEFAULT_CACHE = Path("data/derived/lyrics_cache.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida tracks curadas + letras em um único parquet versionável."
    )
    parser.add_argument("--tracks", type=Path, default=DEFAULT_CURATED_TRACKS_PATH)
    parser.add_argument("--lyrics", type=Path, default=DEFAULT_CURATED_LYRICS_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_CURATED_CORPUS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_brazilian_lyrics_corpus(
        output_path=args.output,
        tracks_path=args.tracks,
        lyrics_path=args.lyrics,
        cache_path=args.cache,
    )
    loader = BrazilianLyricsLoader(output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Corpus consolidado: {output}")
    print(f"Documentos com letra: {loader.count():,}")
    print(f"Tamanho: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
