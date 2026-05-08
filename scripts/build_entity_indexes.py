"""Pré-constrói os índices persistidos de entidades não-track.

Uso:
    uv run python scripts/build_entity_indexes.py
"""

from __future__ import annotations

import time

from music_search.multi_index import DEFAULT_ENTITY_INDEX_PATHS, MultiEntityIndex


def main() -> None:
    start = time.perf_counter()
    multi = MultiEntityIndex.from_parquets(track_engine=None)
    elapsed = time.perf_counter() - start

    print(f"[build_entity_indexes] pronto em {elapsed:.1f}s")
    for kind, index in sorted(multi.entity_indexes.items()):
        target = DEFAULT_ENTITY_INDEX_PATHS.get(kind)
        location = str(target) if target is not None else "(sem cache)"
        print(f"[build_entity_indexes] {kind}: {index.num_docs} docs -> {location}")


if __name__ == "__main__":
    main()
