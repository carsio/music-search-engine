"""Indice multi-entidade: tracks (existente) + artistas/albuns/generos/compositores.

Tracks continuam usando `SparseSearchEngine` (search.py) sem mudancas. Para as outras
entidades, este modulo define `EntityIndex`, uma versao mais leve do search engine
que indexa registros simples (dict) e retorna hits generrcos contendo o payload.

`MultiEntityIndex` e a fachada que junta tudo, expondo `search_routed(query, intent)`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb

from music_search.datasets import DEFAULT_FINAL_DATASET_DIR
from music_search.indexer import IndexBuilder, InvertedIndex
from music_search.ranking import BM25, TFIDF

EntityKind = Literal["track", "artist", "album", "genre", "composer"]
NonTrackEntity = Literal["artist", "album", "genre", "composer"]
SearchAlgorithm = Literal["bm25", "tfidf"]

DEFAULT_INDEX_DIR = Path("data/indexes")
DEFAULT_PARQUETS: dict[NonTrackEntity, Path] = {
    "artist": DEFAULT_FINAL_DATASET_DIR / "br_artists.parquet",
    "album": DEFAULT_FINAL_DATASET_DIR / "br_albums.parquet",
    "genre": DEFAULT_FINAL_DATASET_DIR / "br_genres.parquet",
    "composer": DEFAULT_FINAL_DATASET_DIR / "br_composers.parquet",
}

# Campos a indexar por entidade. Pesos sao defaults; podem ser sobrescritos por param.
ENTITY_FIELDS: dict[NonTrackEntity, tuple[str, ...]] = {
    "artist": ("name", "tagline", "bio", "genres", "origin"),
    "album": ("title", "artist", "description"),
    "genre": ("name", "description", "origin", "representative_artists"),
    "composer": ("name", "bio", "genres"),
}

ENTITY_WEIGHTS: dict[NonTrackEntity, dict[str, float]] = {
    "artist": {"name": 5.0, "tagline": 1.5, "bio": 1.0, "genres": 2.0, "origin": 0.5},
    "album": {"title": 5.0, "artist": 2.0, "description": 1.0},
    "genre": {"name": 5.0, "description": 1.0, "origin": 0.5, "representative_artists": 2.0},
    "composer": {"name": 5.0, "bio": 1.0, "genres": 1.0},
}


@dataclass(frozen=True)
class EntityHit:
    """Hit generico de busca em uma entidade nao-track."""

    id: str
    kind: EntityKind
    rank: int
    algorithm: SearchAlgorithm
    score: float
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "rank": self.rank,
            "algorithm": self.algorithm,
            "score": self.score,
            "payload": self.payload,
        }


def _stringify(value: Any) -> str:
    """Converte dict/list/None em string indexavel."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_stringify(v) for v in value)
    return str(value)


@dataclass
class EntityIndex:
    """Indice invertido + payloads para uma entidade nao-track."""

    kind: NonTrackEntity
    index: InvertedIndex
    documents: dict[str, dict[str, Any]]
    field_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        kind: NonTrackEntity,
        records: Iterable[Mapping[str, Any]],
        *,
        fields: Sequence[str] | None = None,
        field_weights: Mapping[str, float] | None = None,
    ) -> EntityIndex:
        fields = tuple(fields or ENTITY_FIELDS[kind])
        weights = dict(field_weights or ENTITY_WEIGHTS[kind])
        builder = IndexBuilder(fields=fields)
        docs: dict[str, dict[str, Any]] = {}
        for record in records:
            doc_id = str(record.get("id") or "")
            if not doc_id:
                continue
            text_by_field = {f: _stringify(record.get(f, "")) for f in fields}
            builder.add(doc_id, text_by_field)
            docs[doc_id] = dict(record)
        return cls(kind=kind, index=builder.build(), documents=docs, field_weights=weights)

    @property
    def num_docs(self) -> int:
        return self.index.num_docs

    def search(
        self,
        query: str,
        *,
        algorithm: SearchAlgorithm = "bm25",
        top_k: int = 10,
    ) -> list[EntityHit]:
        if top_k <= 0:
            raise ValueError("top_k deve ser > 0")
        query = query.strip()
        if not query or self.num_docs == 0:
            return []
        rankers = (
            {f: BM25(self.index, field=f) for f in self.index.fields}
            if algorithm == "bm25"
            else {f: TFIDF(self.index, field=f) for f in self.index.fields}
        )
        combined: dict[str, float] = defaultdict(float)
        for f, weight in self.field_weights.items():
            if f not in rankers or weight <= 0:
                continue
            results = rankers[f].rank(query, top_k=self.num_docs)
            if not results:
                continue
            max_score = results[0][1] or 1.0
            for doc_id, raw_score in results:
                combined[doc_id] += (raw_score / max_score) * weight
        ranked = sorted(combined.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [
            EntityHit(
                id=doc_id,
                kind=self.kind,
                rank=i + 1,
                algorithm=algorithm,
                score=score,
                payload=self.documents[doc_id],
            )
            for i, (doc_id, score) in enumerate(ranked)
        ]


def load_records_from_parquet(path: Path) -> list[dict[str, Any]]:
    """Le um parquet e devolve lista de dicts. Lista vazia se nao existe."""
    if not path.exists():
        return []
    con = duckdb.connect()
    try:
        rows = con.execute(f"SELECT * FROM '{path.as_posix()}'").fetch_arrow_table().to_pylist()
        return list(rows)
    finally:
        con.close()


@dataclass
class MultiEntityIndex:
    """Fachada juntando indice de tracks (SparseSearchEngine) + entidades nao-track."""

    track_engine: Any | None = None  # SparseSearchEngine — opcional
    entity_indexes: dict[NonTrackEntity, EntityIndex] = field(default_factory=dict)

    @classmethod
    def from_parquets(
        cls,
        *,
        track_engine: Any | None = None,
        parquets: Mapping[NonTrackEntity, Path] | None = None,
    ) -> MultiEntityIndex:
        parquets = dict(parquets or DEFAULT_PARQUETS)
        entity_indexes: dict[NonTrackEntity, EntityIndex] = {}
        for kind, path in parquets.items():
            records = load_records_from_parquet(path)
            if not records:
                continue
            entity_indexes[kind] = EntityIndex.build(kind, records)
        return cls(track_engine=track_engine, entity_indexes=entity_indexes)

    def has(self, kind: EntityKind) -> bool:
        if kind == "track":
            return self.track_engine is not None
        return kind in self.entity_indexes

    def search_entity(
        self,
        kind: NonTrackEntity,
        query: str,
        *,
        algorithm: SearchAlgorithm = "bm25",
        top_k: int = 10,
    ) -> list[EntityHit]:
        idx = self.entity_indexes.get(kind)
        if idx is None:
            return []
        return idx.search(query, algorithm=algorithm, top_k=top_k)

    def search_routed(
        self,
        query: str,
        intent: str,
        *,
        algorithm: SearchAlgorithm = "bm25",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Roteia a busca pelo intent. Sempre devolve `{intent_used, hits}`.

        Fallback para `track` quando intent eh `lyric|song|none` ou indisponivel.
        """
        normalized = intent.lower().strip()
        for entity in ("artist", "album", "genre"):
            if normalized == entity and self.has(entity):  # type: ignore[arg-type]
                hits = self.search_entity(
                    entity,  # type: ignore[arg-type]
                    query,
                    algorithm=algorithm,
                    top_k=top_k,
                )
                return {
                    "intent_used": entity,
                    "hits": [h.to_dict() for h in hits],
                }
        # Caso geral: cai em tracks (cobre lyric, song, none).
        if self.track_engine is None:
            return {"intent_used": "none", "hits": []}
        track_algorithm: SearchAlgorithm = algorithm if algorithm in ("bm25", "tfidf") else "bm25"
        track_hits = self.track_engine.search(query, algorithm=track_algorithm, top_k=top_k)
        return {
            "intent_used": "track",
            "hits": [h.to_dict() for h in track_hits],
        }
