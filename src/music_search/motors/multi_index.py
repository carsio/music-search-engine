"""Indice multi-entidade: tracks + catalogo de albuns + entidades opcionais.

Tracks continuam usando `SparseSearchEngine` (search.py) sem mudancas. Para as outras
entidades, este modulo define `EntityIndex`, uma versao mais leve do search engine
que indexa registros simples (dict) e retorna hits generrcos contendo o payload.

`MultiEntityIndex` e a fachada que junta tudo, expondo `search_routed(query, intent)`.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb

from music_search.core.indexer import IndexBuilder, InvertedIndex
from music_search.core.ranking import BM25, TFIDF, TfScheme
from music_search.data.albums import (
    AlbumDocument,
    build_album_search_records,
    load_album_catalog_from_tracks,
)
from music_search.data.datasets import DEFAULT_CURATED_TRACKS_PATH, DEFAULT_FINAL_DATASET_DIR
from music_search.motors.tuning import (
    SearchProfile,
    entity_weights_for_profile,
    track_weights_for_profile,
)

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
DEFAULT_ENTITY_INDEX_PATHS: dict[NonTrackEntity, Path] = {
    kind: DEFAULT_INDEX_DIR / f"entity_{kind}.pkl" for kind in DEFAULT_PARQUETS
}

# Campos a indexar por entidade. Pesos sao defaults; podem ser sobrescritos por param.
ENTITY_FIELDS: dict[NonTrackEntity, tuple[str, ...]] = {
    "artist": ("name", "tagline", "bio", "raw_text", "genres", "origin"),
    "album": ("title", "artist", "description", "raw_text"),
    "genre": ("name", "description", "raw_text", "origin", "representative_artists"),
    "composer": ("name", "bio", "raw_text", "genres"),
}

ENTITY_WEIGHTS: dict[NonTrackEntity, dict[str, float]] = {
    "artist": {
        "name": 5.0,
        "tagline": 1.5,
        "bio": 1.0,
        "raw_text": 0.75,
        "genres": 2.0,
        "origin": 0.5,
    },
    "album": {"title": 5.0, "artist": 2.0, "description": 1.0, "raw_text": 0.5},
    "genre": {
        "name": 5.0,
        "description": 1.0,
        "raw_text": 0.75,
        "origin": 0.5,
        "representative_artists": 2.0,
    },
    "composer": {"name": 5.0, "bio": 1.0, "raw_text": 0.75, "genres": 1.0},
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

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> EntityIndex:
        with Path(path).open("rb") as file:
            obj = pickle.load(file)
        if not isinstance(obj, cls):
            raise TypeError(f"arquivo não contém {cls.__name__}: {type(obj).__name__}")
        return obj

    def search(
        self,
        query: str,
        *,
        algorithm: SearchAlgorithm = "bm25",
        top_k: int = 10,
        field_weights: Mapping[str, float] | None = None,
        profile: SearchProfile = "balanced",
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        tf_scheme: TfScheme | None = None,
    ) -> list[EntityHit]:
        if top_k <= 0:
            raise ValueError("top_k deve ser > 0")
        query = query.strip()
        if not query or self.num_docs == 0:
            return []
        rankers = self._rankers_for(
            algorithm,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            tf_scheme=tf_scheme,
        )
        weights = self._resolve_field_weights(field_weights, profile=profile)
        combined: dict[str, float] = defaultdict(float)
        for f, weight in weights.items():
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

    def _resolve_field_weights(
        self,
        field_weights: Mapping[str, float] | None,
        *,
        profile: SearchProfile = "balanced",
    ) -> dict[str, float]:
        weights = dict(self.field_weights)
        if field_weights is not None:
            weights = {
                field: float(weight)
                for field, weight in field_weights.items()
                if field in self.index.fields and weight > 0
            }
        weights = entity_weights_for_profile(self.kind, weights, profile)
        if not weights:
            raise ValueError("ao menos um campo com boost positivo é necessário")
        return weights

    def _rankers_for(
        self,
        algorithm: SearchAlgorithm,
        *,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        tf_scheme: TfScheme | None = None,
    ) -> dict[str, BM25 | TFIDF]:
        if algorithm == "bm25":
            return {
                f: BM25(
                    self.index,
                    field=f,
                    k1=1.5 if bm25_k1 is None else bm25_k1,
                    b=0.75 if bm25_b is None else bm25_b,
                )
                for f in self.index.fields
            }
        if algorithm == "tfidf":
            return {
                f: TFIDF(self.index, field=f, tf_scheme="log" if tf_scheme is None else tf_scheme)
                for f in self.index.fields
            }
        raise ValueError(f"algoritmo desconhecido: {algorithm!r}")


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


def _can_reuse_index(*, index_path: Path, source_path: Path) -> bool:
    if not index_path.exists() or not source_path.exists():
        return False
    return index_path.stat().st_mtime >= source_path.stat().st_mtime


def _load_persisted_entity_index(path: Path) -> EntityIndex | None:
    try:
        return EntityIndex.load(path)
    except (
        AttributeError,
        EOFError,
        ModuleNotFoundError,
        OSError,
        TypeError,
        pickle.UnpicklingError,
    ):
        return None


def _load_or_build_entity_index(
    kind: NonTrackEntity,
    records: Iterable[Mapping[str, Any]],
    *,
    source_path: Path,
    index_path: Path,
) -> EntityIndex:
    if _can_reuse_index(index_path=index_path, source_path=source_path):
        cached = _load_persisted_entity_index(index_path)
        if cached is not None:
            return cached

    entity_index = EntityIndex.build(kind, records)
    entity_index.save(index_path)
    return entity_index


@dataclass
class MultiEntityIndex:
    """Fachada juntando indice de tracks (SparseSearchEngine) + entidades nao-track."""

    track_engine: Any | None = None  # SparseSearchEngine — opcional
    entity_indexes: dict[NonTrackEntity, EntityIndex] = field(default_factory=dict)
    album_catalog: dict[str, AlbumDocument] = field(default_factory=dict)

    @classmethod
    def from_parquets(
        cls,
        *,
        track_engine: Any | None = None,
        parquets: Mapping[NonTrackEntity, Path] | None = None,
        index_paths: Mapping[NonTrackEntity, Path] | None = None,
        album_catalog: Mapping[str, AlbumDocument] | None = None,
    ) -> MultiEntityIndex:
        parquets = dict(DEFAULT_PARQUETS if parquets is None else parquets)
        resolved_index_paths = dict(DEFAULT_ENTITY_INDEX_PATHS)
        if index_paths is not None:
            resolved_index_paths.update(index_paths)
        entity_indexes: dict[NonTrackEntity, EntityIndex] = {}
        resolved_album_catalog = dict(
            load_album_catalog_from_tracks() if album_catalog is None else album_catalog
        )
        if resolved_album_catalog:
            album_records = build_album_search_records(resolved_album_catalog.values())
            if album_catalog is None:
                entity_indexes["album"] = _load_or_build_entity_index(
                    "album",
                    album_records,
                    source_path=DEFAULT_CURATED_TRACKS_PATH,
                    index_path=resolved_index_paths["album"],
                )
            else:
                entity_indexes["album"] = EntityIndex.build("album", album_records)
        for kind, path in parquets.items():
            if kind == "album" and resolved_album_catalog:
                continue
            records = load_records_from_parquet(path)
            if not records:
                continue
            entity_indexes[kind] = _load_or_build_entity_index(
                kind,
                records,
                source_path=path,
                index_path=resolved_index_paths[kind],
            )
        return cls(
            track_engine=track_engine,
            entity_indexes=entity_indexes,
            album_catalog=resolved_album_catalog,
        )

    def has(self, kind: EntityKind) -> bool:
        if kind == "track":
            return self.track_engine is not None
        return kind in self.entity_indexes

    def get_album(self, album_id: str) -> AlbumDocument | None:
        return self.album_catalog.get(album_id)

    def search_entity(
        self,
        kind: NonTrackEntity,
        query: str,
        *,
        algorithm: SearchAlgorithm = "bm25",
        top_k: int = 10,
        profile: SearchProfile = "balanced",
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        tf_scheme: TfScheme | None = None,
    ) -> list[EntityHit]:
        idx = self.entity_indexes.get(kind)
        if idx is None:
            return []
        return idx.search(
            query,
            algorithm=algorithm,
            top_k=top_k,
            profile=profile,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            tf_scheme=tf_scheme,
        )

    def search_routed(
        self,
        query: str,
        intent: str,
        *,
        algorithm: SearchAlgorithm = "bm25",
        top_k: int = 10,
        profile: SearchProfile = "balanced",
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        tf_scheme: TfScheme | None = None,
    ) -> dict[str, Any]:
        """Seleciona o tipo vencedor comparando candidatos disponiveis."""
        normalized = intent.lower().strip()
        candidates: list[tuple[EntityKind, float, float, list[Any]]] = []

        track_algorithm: SearchAlgorithm = algorithm if algorithm in ("bm25", "tfidf") else "bm25"
        if self.track_engine is not None:
            track_weights = track_weights_for_profile(self.track_engine.field_weights, profile)
            track_hits = self.track_engine.search(
                query,
                algorithm=track_algorithm,
                top_k=top_k,
                profile=profile,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                tf_scheme=tf_scheme,
            )
            if track_hits:
                candidates.append(
                    (
                        "track",
                        self._normalized_track_score(track_hits[0], track_weights)
                        + self._intent_bonus(normalized, "track"),
                        track_hits[0].score,
                        track_hits,
                    )
                )

        for entity in ("album", "artist", "genre"):
            idx = self.entity_indexes.get(entity)  # type: ignore[arg-type]
            if idx is None:
                continue
            entity_weights = idx._resolve_field_weights(None, profile=profile)
            entity_hits = idx.search(
                query,
                algorithm=algorithm,
                top_k=top_k,
                profile=profile,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                tf_scheme=tf_scheme,
            )
            if not entity_hits:
                continue
            candidates.append(
                (
                    entity,
                    self._normalized_entity_score(entity_hits[0], entity_weights)
                    + self._intent_bonus(normalized, entity),
                    entity_hits[0].score,
                    entity_hits,
                )
            )

        if not candidates:
            return {"intent_used": "none", "hits": []}

        winner, _score, _raw, hits = max(
            candidates,
            key=lambda item: (
                item[1],
                item[2],
                self._intent_bonus(normalized, item[0]),
                self._kind_priority(item[0]),
            ),
        )
        return {
            "intent_used": winner,
            "hits": [h.to_dict() for h in hits],
        }

    def _normalized_track_score(
        self,
        hit: Any,
        weights: Mapping[str, float],
    ) -> float:
        total = sum(float(weight) for weight in weights.values()) or 1.0
        return float(hit.score) / total

    def _normalized_entity_score(
        self,
        hit: EntityHit,
        weights: Mapping[str, float],
    ) -> float:
        total = sum(float(weight) for weight in weights.values()) or 1.0
        return float(hit.score) / total

    @staticmethod
    def _intent_bonus(intent: str, kind: EntityKind) -> float:
        if kind == "track" and intent in {"track", "song", "lyric"}:
            return 0.12
        if intent == kind:
            return 0.12
        return 0.0

    @staticmethod
    def _kind_priority(kind: EntityKind) -> int:
        priorities: dict[EntityKind, int] = {
            "album": 4,
            "artist": 3,
            "genre": 2,
            "track": 1,
            "composer": 0,
        }
        return priorities.get(kind, 0)
