"""
src/music_search/indexer.py
===========================
Construção e persistência do índice BM25 sobre metadados de músicas.

Responsabilidades:
    - Tokenizar o campo text_field gerado pelo pipeline de limpeza
    - Construir o índice BM25Okapi (rank_bm25)
    - Serializar / desserializar o índice em disco (pickle)
    - Expor MusicIndex como objeto reutilizável pelas Partes 2, 3 e 4

Dependências:
    pip install rank-bm25 pandas nltk
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from music_search.preprocessing import preprocessar_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_INDEX_PATH = Path("data") / "bm25_index.pkl"
DEFAULT_CORPUS_PATH = Path("data") / "songs_clean.csv"

# Parâmetros BM25Okapi (valores padrão da literatura)
BM25_K1 = 1.5   # controla saturação da frequência do termo
BM25_B  = 0.75  # controla normalização pelo comprimento do documento


# ---------------------------------------------------------------------------
# Estrutura de resultado
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Um resultado de busca com metadados e score BM25."""
    rank:        int
    score:       float
    track_id:    str
    track_name:  str
    artist_name: str
    album_name:  str
    popularity:  float
    energy:      float | None = None
    danceability: float | None = None
    extra:       dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"[#{self.rank}] {self.track_name} — {self.artist_name}"
            f"  (score={self.score:.4f}, pop={self.popularity})"
        )


# ---------------------------------------------------------------------------
# Índice BM25
# ---------------------------------------------------------------------------

class MusicIndex:
    """
    Índice BM25 sobre metadados textuais de músicas.

    Uso básico
    ----------
    >>> idx = MusicIndex.build(df)          # constrói a partir de um DataFrame
    >>> idx.save()                          # persiste em disco
    >>> idx = MusicIndex.load()             # carrega do disco
    >>> results = idx.search("sad indie")   # busca top-10
    """

    def __init__(
        self,
        bm25:   BM25Okapi,
        corpus: list[list[str]],
        df:     pd.DataFrame,
        k1:     float = BM25_K1,
        b:      float = BM25_B,
    ) -> None:
        self._bm25   = bm25
        self._corpus = corpus
        self._df     = df.reset_index(drop=True)
        self.k1      = k1
        self.b       = b
        self.n_docs  = len(corpus)

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        df:          pd.DataFrame,
        text_col:    str = "text_field",
        k1:          float = BM25_K1,
        b:           float = BM25_B,
        use_stemming: bool = True,
    ) -> "MusicIndex":
        """
        Constrói o índice BM25 a partir de um DataFrame limpo.

        Parâmetros
        ----------
        df           : DataFrame com ao menos as colunas text_field,
                       track_name, artist_name, album_name
        text_col     : coluna que contém o texto pré-processado
        k1           : parâmetro BM25 de saturação de frequência
        b            : parâmetro BM25 de normalização por comprimento
        use_stemming : aplica PorterStemmer nos tokens (recomendado)
        """
        if text_col not in df.columns:
            raise ValueError(
                f"Coluna '{text_col}' não encontrada. "
                f"Execute o pipeline de limpeza (02_limpeza_dataset.py) primeiro."
            )

        logger.info("Construindo índice BM25 | docs=%d k1=%.2f b=%.2f", len(df), k1, b)
        t0 = time.perf_counter()

        # Tokenização de todo o corpus
        corpus: list[list[str]] = []
        for texto in df[text_col].fillna(""):
            tokens = preprocessar_tokens(str(texto), use_stemming=use_stemming)
            corpus.append(tokens)

        # Documentos sem nenhum token viram lista vazia — BM25Okapi tolera isso
        n_vazios = sum(1 for t in corpus if not t)
        if n_vazios:
            logger.warning("%d documentos sem tokens após pré-processamento.", n_vazios)

        bm25 = BM25Okapi(corpus, k1=k1, b=b)

        elapsed = time.perf_counter() - t0
        logger.info("Índice construído em %.2fs | vocab=%d", elapsed, len(bm25.idf))

        return cls(bm25=bm25, corpus=corpus, df=df, k1=k1, b=b)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path: Path | str = DEFAULT_INDEX_PATH) -> Path:
        """Serializa o índice completo em disco via pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "bm25":    self._bm25,
            "corpus":  self._corpus,
            "df":      self._df,
            "k1":      self.k1,
            "b":       self.b,
            "n_docs":  self.n_docs,
            "version": "1.0",
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = path.stat().st_size / 1e6
        logger.info("Índice salvo em %s (%.1f MB)", path, size_mb)
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX_PATH) -> "MusicIndex":
        """Carrega um índice previamente salvo do disco."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Índice não encontrado: {path}\n"
                "Execute MusicIndex.build(df).save() primeiro."
            )

        with open(path, "rb") as f:
            payload = pickle.load(f)

        logger.info("Índice carregado de %s | docs=%d", path, payload["n_docs"])
        return cls(
            bm25=payload["bm25"],
            corpus=payload["corpus"],
            df=payload["df"],
            k1=payload["k1"],
            b=payload["b"],
        )

    # ------------------------------------------------------------------
    # Busca
    # ------------------------------------------------------------------

    def search(
        self,
        query:        str,
        k:            int = 10,
        use_stemming: bool = True,
        min_score:    float = 0.0,
    ) -> list[SearchResult]:
        """
        Busca as K músicas mais relevantes para a query.

        Parâmetros
        ----------
        query        : texto livre (ex: "energetic dance pop")
        k            : número de resultados a retornar
        use_stemming : deve ser igual ao usado na indexação
        min_score    : filtra resultados com score abaixo deste valor

        Retorno
        -------
        Lista de SearchResult ordenada por score decrescente.
        """
        if not query or not query.strip():
            logger.warning("Query vazia recebida.")
            return []

        query_tokens = preprocessar_tokens(query, use_stemming=use_stemming)
        if not query_tokens:
            logger.warning("Query '%s' resultou em 0 tokens após pré-processamento.", query)
            return []

        # Scores BM25 para todos os documentos
        scores: np.ndarray = self._bm25.get_scores(query_tokens)

        # Índices dos top-K por score decrescente
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_k_idx, start=1):
            score = float(scores[idx])
            if score < min_score:
                break

            row = self._df.iloc[idx]
            results.append(SearchResult(
                rank=rank,
                score=score,
                track_id=str(row.get("id", row.get("track_id", idx))),
                track_name=str(row.get("track_name", "")),
                artist_name=str(row.get("artist_name", "")),
                album_name=str(row.get("album_name", "")),
                popularity=float(row.get("popularity", 0)),
                energy=float(row["energy"]) if "energy" in row and pd.notna(row["energy"]) else None,
                danceability=float(row["danceability"]) if "danceability" in row and pd.notna(row["danceability"]) else None,
                extra={
                    col: row[col]
                    for col in ["tempo", "valence", "acousticness", "genre"]
                    if col in row.index and pd.notna(row[col])
                },
            ))

        logger.info(
            "Busca '%s' | tokens=%s | resultados=%d | top_score=%.4f",
            query, query_tokens, len(results), results[0].score if results else 0,
        )
        return results

    def get_scores_raw(self, query: str, use_stemming: bool = True) -> np.ndarray:
        """
        Retorna o array completo de scores BM25 para todos os documentos.
        Usado pelas Partes 2 e 3 para combinar com scores vetoriais / LTR.
        """
        tokens = preprocessar_tokens(query, use_stemming=use_stemming)
        return self._bm25.get_scores(tokens) if tokens else np.zeros(self.n_docs)

    # ------------------------------------------------------------------
    # Estatísticas do índice
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Retorna estatísticas básicas do índice para o relatório."""
        doc_lengths = np.array([len(doc) for doc in self._corpus])
        return {
            "n_docs":          self.n_docs,
            "vocab_size":      len(self._bm25.idf),
            "avg_doc_length":  float(doc_lengths.mean().round(2)),
            "max_doc_length":  int(doc_lengths.max()),
            "min_doc_length":  int(doc_lengths.min()),
            "empty_docs":      int((doc_lengths == 0).sum()),
            "k1":              self.k1,
            "b":               self.b,
        }

    def __repr__(self) -> str:
        return (
            f"MusicIndex(n_docs={self.n_docs}, "
            f"vocab={len(self._bm25.idf)}, k1={self.k1}, b={self.b})"
        )