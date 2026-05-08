"""CLI do enrichment offline a partir da Wikipedia PT."""

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


def _build_seeds(args: argparse.Namespace, gen: SeedGenerator, *, limit: int | None) -> list[str]:
    if args.kind == "genres":
        return list(gen(limit=limit, seed_mode=args.seed_mode))
    return list(gen(limit=limit))


async def _run(args: argparse.Namespace) -> None:
    kind, gen = _SEED_GENS[args.kind]
    if args.count_only:
        print(len(_build_seeds(args, gen, limit=None)))
        return

    cfg = EnrichmentConfig(
        concurrency=args.concurrency,
        retry_errors=args.retry_errors,
        limit=args.limit,
        fetch_documents=args.phase in {"all", "fetch"},
        normalize_documents=args.phase in {"all", "normalize", "materialize"},
    )
    seeds = _build_seeds(args, gen, limit=args.limit)
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
        "--seed-mode",
        choices=("expanded", "macro"),
        default="expanded",
        help="modo das seeds de genero: detalhado (artist_genres) ou macro (taxonomia curada)",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "fetch", "materialize", "normalize"),
        default="all",
        help=(
            "fase do pipeline: baixar documentos, materializar payload local da Wikipedia "
            "ou ambas (`normalize` permanece como alias por compatibilidade)"
        ),
    )
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
