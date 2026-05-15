"""Prepara os artefatos de busca persistidos usados pela API.

Uso:
    uv run python -m music_search.scripts.prepare_search_artifacts
"""

from __future__ import annotations

import time

from music_search.motors.multi_index import MultiEntityIndex
from music_search.motors.search import load_or_build_default_engine


def _build_dense(track_engine: object) -> None:
    """Constrói e persiste o índice FAISS. Ignora se deps não instaladas."""
    try:
        from music_search.motors.dense_search import (
            DEFAULT_DENSE_INDEX_PATH,
            DEFAULT_DENSE_META_PATH,
            DenseSearchEngine,
        )
    except ImportError:
        print("[prepare_search_artifacts] sentence-transformers/faiss não instalados — pulando índice denso")
        return

    records = list(getattr(track_engine, "documents", {}).values())
    if not records:
        print("[prepare_search_artifacts] nenhum documento para indexar no motor denso")
        return

    t0 = time.perf_counter()
    print(f"[prepare_search_artifacts] construindo índice denso ({len(records)} docs)...")
    engine = DenseSearchEngine.build(records)
    engine.save(DEFAULT_DENSE_INDEX_PATH, DEFAULT_DENSE_META_PATH)
    print(f"[prepare_search_artifacts] índice denso: {engine.num_docs} docs ({time.perf_counter() - t0:.1f}s)")


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

    _build_dense(track_engine)

    print(f"[prepare_search_artifacts] pronto em {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
