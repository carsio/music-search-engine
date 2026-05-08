"""Motor de busca esparsa sobre o corpus curado brasileiro com letras.

Reaproveita o índice invertido multi-campo e os rankers já existentes
(`TFIDF`, `BM25`) e adiciona apenas a orquestração necessária para:

- carregar o corpus curado consolidado;
- construir ou reaproveitar o índice persistido;
- combinar scores de múltiplos campos com boosts por domínio.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from music_search.datasets import (
    CURATED_FIELDS,
    DEFAULT_CURATED_CORPUS_PATH,
    BrazilianLyricsLoader,
    CuratedLyricsDocument,
)
from music_search.indexer import IndexBuilder, InvertedIndex
from music_search.ranking import BM25, TFIDF, TfScheme
from music_search.search_tuning import SearchProfile, track_weights_for_profile

SearchAlgorithm = Literal["bm25", "tfidf"]

DEFAULT_INDEX_PATH = Path("data/indexes/br_curated_lyrics.pkl")
DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "track_name": 2.5,
    "artist_names": 1.0,
    "artist_genres": 1.5,
    "macro_genre": 1.0,
    "album_name": 0.75,
    "lyrics": 4.0,
}
FIELD_LABELS: dict[str, str] = {
    "track_name": "Título",
    "artist_names": "Artistas",
    "artist_genres": "Gêneros",
    "macro_genre": "Macro-gênero",
    "album_name": "Álbum",
    "lyrics": "Letra",
}


@dataclass(frozen=True)
class FieldContribution:
    """Detalha como um campo contribuiu para o score final."""

    field: str
    raw_score: float
    normalized_score: float
    weight: float
    weighted_score: float

    @property
    def label(self) -> str:
        return FIELD_LABELS.get(self.field, self.field)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "label": self.label,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
        }


@dataclass(frozen=True)
class SearchHit:
    """Representa um hit pronto para UI/CLI."""

    id: str
    rank: int
    algorithm: SearchAlgorithm
    score: float
    track_name: str
    primary_artist_name: str
    artist_names: str
    artist_genres: str
    macro_genre: str
    album_name: str
    release_date: str
    release_year: int
    track_popularity: int
    duration_ms: int
    explicit: bool
    lyrics_source: str
    lyrics_source_url: str
    lyrics_preview: str
    lyrics: str
    field_scores: tuple[FieldContribution, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rank": self.rank,
            "algorithm": self.algorithm,
            "score": self.score,
            "track_name": self.track_name,
            "primary_artist_name": self.primary_artist_name,
            "artist_names": self.artist_names,
            "artist_genres": self.artist_genres,
            "macro_genre": self.macro_genre,
            "album_name": self.album_name,
            "release_date": self.release_date,
            "release_year": self.release_year,
            "track_popularity": self.track_popularity,
            "duration_ms": self.duration_ms,
            "explicit": self.explicit,
            "lyrics_source": self.lyrics_source,
            "lyrics_source_url": self.lyrics_source_url,
            "lyrics_preview": self.lyrics_preview,
            "lyrics": self.lyrics,
            "field_scores": [item.to_dict() for item in self.field_scores],
            "data_completa": {
                "id": self.id,
                "track_name": self.track_name,
                "primary_artist_name": self.primary_artist_name,
                "artist_names": self.artist_names,
                "artist_genres": self.artist_genres,
                "macro_genre": self.macro_genre,
                "album_name": self.album_name,
                "release_date": self.release_date,
                "release_year": self.release_year,
                "track_popularity": self.track_popularity,
                "duration_ms": self.duration_ms,
                "explicit": self.explicit,
                "lyrics_source": self.lyrics_source,
                "lyrics_source_url": self.lyrics_source_url,
            },
        }


@dataclass
class SparseSearchEngine:
    """Busca multi-campo com BM25 e TF-IDF sobre o corpus curado."""

    index: InvertedIndex
    documents: dict[str, CuratedLyricsDocument]
    field_weights: dict[str, float]

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("documents não pode estar vazio")
        invalid_fields = set(self.field_weights) - set(self.index.fields)
        if invalid_fields:
            raise KeyError(
                f"boosts referenciam campos ausentes no índice: {sorted(invalid_fields)}"
            )
        self.field_weights = {
            field: float(weight)
            for field, weight in self.field_weights.items()
            if field in self.index.fields and weight > 0
        }
        if not self.field_weights:
            raise ValueError("ao menos um campo com boost positivo é necessário")
        self._bm25_rankers = {field: BM25(self.index, field=field) for field in self.index.fields}
        self._tfidf_rankers = {field: TFIDF(self.index, field=field) for field in self.index.fields}

    @classmethod
    def build(
        cls,
        docs: Iterable[Mapping[str, object]],
        *,
        fields: Sequence[str] = CURATED_FIELDS,
        field_weights: Mapping[str, float] | None = None,
    ) -> SparseSearchEngine:
        builder = IndexBuilder(fields=tuple(fields))
        documents: dict[str, CuratedLyricsDocument] = {}
        for doc in docs:
            document = _coerce_document(doc)
            doc_id = document["id"]
            builder.add(doc_id, _extract_index_values(document, fields))
            documents[doc_id] = document
        return cls(
            index=builder.build(),
            documents=documents,
            field_weights=dict(field_weights or DEFAULT_FIELD_WEIGHTS),
        )

    @property
    def num_docs(self) -> int:
        return self.index.num_docs

    def search(
        self,
        query: str,
        *,
        algorithm: SearchAlgorithm,
        top_k: int = 10,
        field_weights: Mapping[str, float] | None = None,
        profile: SearchProfile = "balanced",
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        tf_scheme: TfScheme | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError(f"top_k deve ser > 0 (recebido {top_k})")
        query = query.strip()
        if not query:
            return []

        weights = self._resolve_field_weights(field_weights, profile=profile)
        combined_scores: dict[str, float] = defaultdict(float)
        contributions_by_doc: dict[str, list[FieldContribution]] = defaultdict(list)
        rankers = self._rankers_for(
            algorithm,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            tf_scheme=tf_scheme,
        )

        for field, weight in weights.items():
            results = rankers[field].rank(query, top_k=self.index.num_docs)
            if not results:
                continue
            max_score = results[0][1] or 1.0
            for doc_id, raw_score in results:
                normalized = raw_score / max_score if max_score else 0.0
                weighted = normalized * weight
                combined_scores[doc_id] += weighted
                contributions_by_doc[doc_id].append(
                    FieldContribution(
                        field=field,
                        raw_score=raw_score,
                        normalized_score=normalized,
                        weight=weight,
                        weighted_score=weighted,
                    )
                )

        ranked_ids = sorted(combined_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        hits: list[SearchHit] = []
        for rank, (doc_id, score) in enumerate(ranked_ids, start=1):
            doc = self.documents[doc_id]
            field_scores = tuple(
                sorted(
                    contributions_by_doc[doc_id],
                    key=lambda item: (-item.weighted_score, item.field),
                )
            )
            hits.append(
                SearchHit(
                    id=doc_id,
                    rank=rank,
                    algorithm=algorithm,
                    score=score,
                    track_name=_text_value(doc.get("track_name")),
                    primary_artist_name=_text_value(doc.get("primary_artist_name")),
                    artist_names=_text_value(doc.get("artist_names")),
                    artist_genres=_text_value(doc.get("artist_genres")),
                    macro_genre=_text_value(doc.get("macro_genre")),
                    album_name=_text_value(doc.get("album_name")),
                    release_date=_text_value(doc.get("release_date")),
                    release_year=_int_value(doc.get("release_year")),
                    track_popularity=_int_value(doc.get("track_popularity")),
                    duration_ms=_int_value(doc.get("duration_ms")),
                    explicit=bool(doc.get("explicit") or False),
                    lyrics_source=_text_value(doc.get("lyrics_source")),
                    lyrics_source_url=_text_value(doc.get("lyrics_source_url")),
                    lyrics_preview=_make_preview(_text_value(doc.get("lyrics"))),
                    lyrics=_text_value(doc.get("lyrics")),
                    field_scores=field_scores,
                )
            )
        return hits

    def search_both(
        self,
        query: str,
        *,
        top_k: int = 10,
        field_weights: Mapping[str, float] | None = None,
        profile: SearchProfile = "balanced",
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        tf_scheme: TfScheme | None = None,
    ) -> dict[SearchAlgorithm, list[SearchHit]]:
        return {
            "bm25": self.search(
                query,
                algorithm="bm25",
                top_k=top_k,
                field_weights=field_weights,
                profile=profile,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
            ),
            "tfidf": self.search(
                query,
                algorithm="tfidf",
                top_k=top_k,
                field_weights=field_weights,
                profile=profile,
                tf_scheme=tf_scheme,
            ),
        }

    def save_index(self, path: Path = DEFAULT_INDEX_PATH) -> Path:
        target = Path(path)
        self.index.save(target)
        return target

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
        weights = track_weights_for_profile(weights, profile)
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
    ) -> Mapping[str, BM25 | TFIDF]:
        if algorithm == "bm25":
            if (bm25_k1 is None or bm25_k1 == 1.5) and (bm25_b is None or bm25_b == 0.75):
                return self._bm25_rankers
            return {
                field: BM25(
                    self.index,
                    field=field,
                    k1=1.5 if bm25_k1 is None else bm25_k1,
                    b=0.75 if bm25_b is None else bm25_b,
                )
                for field in self.index.fields
            }
        if algorithm == "tfidf":
            if tf_scheme is None or tf_scheme == "log":
                return self._tfidf_rankers
            return {
                field: TFIDF(self.index, field=field, tf_scheme=tf_scheme)
                for field in self.index.fields
            }
        raise ValueError(f"algoritmo desconhecido: {algorithm!r}")


def load_or_build_default_engine(
    *,
    corpus_path: Path = DEFAULT_CURATED_CORPUS_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    rebuild_index: bool = False,
    field_weights: Mapping[str, float] | None = None,
) -> SparseSearchEngine:
    """Carrega o índice persistido ou o reconstrói quando necessário."""

    loader = BrazilianLyricsLoader(corpus_path=corpus_path)
    documents = {str(doc["id"]): doc for doc in loader.iter_docs()}
    weights = dict(field_weights or DEFAULT_FIELD_WEIGHTS)

    if not rebuild_index and _can_reuse_index(index_path=index_path, corpus_path=corpus_path):
        index = InvertedIndex.load(index_path)
        return SparseSearchEngine(index=index, documents=documents, field_weights=weights)

    engine = SparseSearchEngine.build(documents.values(), field_weights=weights)
    engine.save_index(index_path)
    return engine


def _coerce_document(doc: Mapping[str, object]) -> CuratedLyricsDocument:
    return CuratedLyricsDocument(
        id=_text_value(doc.get("id")),
        track_name=_text_value(doc.get("track_name")),
        primary_artist_name=_text_value(doc.get("primary_artist_name")),
        artist_names=_text_value(doc.get("artist_names")),
        artist_genres=_text_value(doc.get("artist_genres")),
        macro_genre=_text_value(doc.get("macro_genre")),
        album_name=_text_value(doc.get("album_name")),
        release_date=_text_value(doc.get("release_date")),
        release_year=_int_value(doc.get("release_year")),
        track_popularity=_int_value(doc.get("track_popularity")),
        duration_ms=_int_value(doc.get("duration_ms")),
        explicit=bool(doc.get("explicit") or False),
        lyrics_source=_text_value(doc.get("lyrics_source")),
        lyrics_source_url=_text_value(doc.get("lyrics_source_url")),
        lyrics=_text_value(doc.get("lyrics")),
    )


def _extract_index_values(
    doc: CuratedLyricsDocument,
    fields: Sequence[str],
) -> dict[str, str | None]:
    return {field: _text_value(doc.get(field)) for field in fields}


def _can_reuse_index(*, index_path: Path, corpus_path: Path) -> bool:
    if not index_path.exists() or not corpus_path.exists():
        return False
    return index_path.stat().st_mtime >= corpus_path.stat().st_mtime


def _text_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _int_value(value: object) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return int(str(value))


def _make_preview(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _print_hits(hits: Sequence[SearchHit]) -> None:
    if not hits:
        print("(sem resultados)")
        return
    for hit in hits:
        print(
            f"#{hit.rank:>2}  {hit.score:>7.4f}  {hit.track_name}"
            f"  |  {hit.artist_names or hit.primary_artist_name}"
            f"  |  {hit.artist_genres or hit.macro_genre or '—'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca esparsa sobre o corpus curado brasileiro.")
    parser.add_argument("query", help="texto da consulta")
    parser.add_argument("--top", type=int, default=10, help="quantidade de resultados")
    parser.add_argument(
        "--algorithm",
        choices=["bm25", "tfidf", "both"],
        default="both",
        help="algoritmo de ranking",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CURATED_CORPUS_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="força reconstrução do índice a partir do corpus",
    )
    args = parser.parse_args()

    engine = load_or_build_default_engine(
        corpus_path=args.corpus,
        index_path=args.index,
        rebuild_index=args.rebuild_index,
    )

    if args.algorithm == "both":
        for algorithm, hits in engine.search_both(args.query, top_k=args.top).items():
            print(f"\n[{algorithm.upper()}]")
            _print_hits(hits)
        return

    _print_hits(engine.search(args.query, algorithm=args.algorithm, top_k=args.top))


if __name__ == "__main__":
    main()
