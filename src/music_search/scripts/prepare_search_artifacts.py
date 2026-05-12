"""Prepara os artefatos de busca persistidos usados pela API.

Uso:
    uv run python -m music_search.scripts.prepare_search_artifacts
"""

from __future__ import annotations

import time

from music_search.motors.multi_index import MultiEntityIndex
from music_search.motors.search import load_or_build_default_engine


def main() -> None:
    start = time.perf_counter()

    track_engine = load_or_build_default_engine()
    after_tracks = time.perf_counter()

    multi = MultiEntityIndex.from_parquets(track_engine=None)
    after_entities = time.perf_counter()

    entity_counts = {kind: index.num_docs for kind, index in sorted(multi.entity_indexes.items())}
    print(
        f"[prepare_search_artifacts] tracks: {track_engine.num_docs} docs "
        f"({after_tracks - start:.1f}s)"
    )
    print(
        f"[prepare_search_artifacts] entities: {entity_counts} "
        f"({after_entities - after_tracks:.1f}s)"
    )
    print(f"[prepare_search_artifacts] pronto em {after_entities - start:.1f}s")


if __name__ == "__main__":
    main()
