"""Motor de busca vetorial densa (sentence-transformers + FAISS).

Usa paraphrase-multilingual-MiniLM-L12-v2 (384 dims, suporte a PT)
e IndexFlatIP do FAISS sobre vetores normalizados (= cosine similarity).
Embeddings são pré-computados no build; busca é puramente local.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DENSE_INDEX_PATH = Path("data/indexes/dense_index.faiss")
DEFAULT_DENSE_META_PATH = Path("data/indexes/dense_meta.pkl")

_SLIM_KEYS = (
    "track_name",
    "primary_artist_name",
    "artist_names",
    "artist_genres",
    "macro_genre",
    "album_name",
    "release_year",
    "lyrics_preview",
    "lyrics_source",
    "lyrics_source_url",
)


def _load_faiss() -> Any:
    return import_module("faiss")


def _load_sentence_transformer() -> Any:
    return import_module("sentence_transformers").SentenceTransformer


def _track_text(r: dict[str, Any]) -> str:
    parts = [
        r.get("track_name") or "",
        r.get("primary_artist_name") or r.get("artist_names") or "",
        r.get("artist_genres") or "",
        (r.get("lyrics") or "")[:300],
    ]
    return " ".join(p for p in parts if p).strip()


@dataclass
class DenseHit:
    id: str
    rank: int
    score: float
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "track",
            "rank": self.rank,
            "algorithm": "dense",
            "score": float(self.score),
            "payload": self.payload,
        }


@dataclass
class DenseSearchEngine:
    """Busca vetorial densa sobre tracks usando FAISS + sentence-transformers."""

    model_name: str
    index: Any  # faiss.Index
    doc_ids: list[str]
    documents: dict[str, dict[str, Any]]
    _model: Any = field(default=None, repr=False)

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    def _get_model(self) -> Any:
        if self._model is None:
            SentenceTransformer = _load_sentence_transformer()
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def warmup(self) -> None:
        """Carrega o modelo em memória antecipadamente para eliminar cold start."""
        self._get_model()

    def search(self, query: str, *, top_k: int = 10) -> list[DenseHit]:
        if not query.strip() or self.num_docs == 0:
            return []
        model = self._get_model()
        vec = model.encode([query.strip()], normalize_embeddings=True)
        k = min(top_k, self.num_docs)
        scores, indices = self.index.search(vec.astype("float32"), k)
        hits: list[DenseHit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1):
            if idx < 0:
                continue
            doc_id = self.doc_ids[int(idx)]
            hits.append(
                DenseHit(
                    id=doc_id,
                    rank=rank,
                    score=float(score),
                    payload=self.documents.get(doc_id, {}),
                )
            )
        return hits

    def save(
        self,
        index_path: Path = DEFAULT_DENSE_INDEX_PATH,
        meta_path: Path = DEFAULT_DENSE_META_PATH,
    ) -> None:
        faiss = _load_faiss()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with meta_path.open("wb") as f:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "doc_ids": self.doc_ids,
                    "documents": self.documents,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(
        cls,
        index_path: Path = DEFAULT_DENSE_INDEX_PATH,
        meta_path: Path = DEFAULT_DENSE_META_PATH,
    ) -> DenseSearchEngine:
        faiss = _load_faiss()
        index = faiss.read_index(str(index_path))
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        return cls(
            model_name=meta["model_name"],
            index=index,
            doc_ids=meta["doc_ids"],
            documents=meta["documents"],
        )

    @classmethod
    def build(
        cls,
        records: list[dict[str, Any]],
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 256,
    ) -> DenseSearchEngine:
        """Constrói índice FAISS a partir de dicts de track (com campo 'lyrics')."""
        faiss = _load_faiss()
        SentenceTransformer = _load_sentence_transformer()
        model = SentenceTransformer(model_name)
        doc_ids: list[str] = []
        texts: list[str] = []
        documents: dict[str, dict[str, Any]] = {}

        for r in records:
            doc_id = str(r.get("id") or "")
            if not doc_id:
                continue
            doc_ids.append(doc_id)
            texts.append(_track_text(r))
            documents[doc_id] = {k: r[k] for k in _SLIM_KEYS if k in r}

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return cls(
            model_name=model_name,
            index=index,
            doc_ids=doc_ids,
            documents=documents,
            _model=model,
        )
