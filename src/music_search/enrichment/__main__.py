"""CLI: `python -m music_search.enrichment <kind> [--phase fetch|normalize|all]`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Iterable

from music_search.enrichment.models import EntityKind
from music_search.enrichment.pipeline import EnrichmentConfig, run_enrichment
from music_search.enrichment.seeds import (
    album_seeds,
    artist_seeds,
    composer_seeds,
    genre_seeds,
)

SeedGenerator = Callable[..., Iterable[str]]

_SEED_GENS: dict[str, tuple[EntityKind, SeedGenerator]] = {
    "artists": ("artist", artist_seeds),
    "albums": ("album", album_seeds),
    "genres": ("genre", genre_seeds),
    "composers": ("composer", composer_seeds),
}


async def _run(args: argparse.Namespace) -> None:
    kind, gen = _SEED_GENS[args.kind]
    if args.count_only:
        print(sum(1 for _ in gen(limit=None)))
        return

    cfg = EnrichmentConfig(
        concurrency=args.concurrency,
        retry_errors=args.retry_errors,
        limit=args.limit,
        fetch_documents=args.phase in {"all", "fetch"},
        normalize_documents=args.phase in {"all", "normalize"},
    )
    seeds = list(gen(limit=args.limit))
    if not seeds:
        print("Sem sementes — corpus de tracks vazio ou ausente.", file=sys.stderr)
        sys.exit(1)
    print(f"Sementes geradas: {len(seeds):,}")
    await run_enrichment(kind, seeds, cfg)


def main() -> None:
    parser = argparse.ArgumentParser("music_search.enrichment")
    parser.add_argument("kind", choices=list(_SEED_GENS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("all", "fetch", "normalize"),
        default="all",
        help="fase do pipeline: baixar documentos, normalizar com LLM ou ambas",
    )
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
