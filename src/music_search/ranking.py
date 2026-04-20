"""Modelos de ranking: TF-IDF e Okapi BM25 sobre o índice invertido multi-campo.

TF-IDF (issue #5): pondera termos pela frequência no documento (tf) e
raridade no corpus (idf), com similaridade de cosseno entre vetores
esparsos. Três variantes de tf suportadas (raw, log, augmented).

BM25 (issue #6): pontua documentos combinando saturação de tf (controlada
por `k1`) e normalização pelo tamanho do documento (controlada por `b`).

Fórmulas:

    cos_sim(d, q) = (v_d · v_q) / (||v_d|| · ||v_q||)
    w(t, d)       = tf_weight(tf(t,d)) · idf(t)
    idf_tfidf(t)  = ln(N / df(t))

    BM25(D, Q)    = Σ_{t ∈ Q}  IDF_bm25(t) · (tf(t,D) · (k1+1))
                               / (tf(t,D) + k1 · (1 - b + b · |D|/avgdl))
    IDF_bm25(t)   = ln( (N - df(t) + 0.5) / (df(t) + 0.5)  +  1 )

Valores padrão do BM25 (`k1=1.5`, `b=0.75`) seguem a convenção do
Elasticsearch/Lucene; no TF-IDF o padrão é `log` (esquema ltc do SMART).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from music_search.indexer import InvertedIndex
from music_search.preprocessing import preprocess

TfScheme = Literal["raw", "log", "augmented"]
_SCHEMES: tuple[TfScheme, ...] = ("raw", "log", "augmented")


def tfidf_idf(num_docs: int, df: int) -> float:
    """IDF clássico `ln(N/df)`. Retorna 0 para termos fora do vocabulário (df=0)."""
    if df <= 0 or num_docs <= 0:
        return 0.0
    return math.log(num_docs / df)


def bm25_idf(num_docs: int, df: int) -> float:
    """IDF do BM25 com suavização +0.5 e +1 para garantir valores não negativos."""
    return math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)


def tf_weight(count: int, scheme: TfScheme, max_count: int = 0) -> float:
    """Peso de tf conforme o esquema escolhido.

    - raw: contagem bruta.
    - log: `1 + ln(count)` — amortece termos muito repetidos.
    - augmented: `0.5 + 0.5 · count/max_count` — também amortece, mas normaliza
      por documento (o termo mais frequente do doc tem peso 1.0).

    `count <= 0` sempre retorna 0 (termo ausente).
    """
    if count <= 0:
        return 0.0
    if scheme == "raw":
        return float(count)
    if scheme == "log":
        return 1.0 + math.log(count)
    if scheme == "augmented":
        if max_count <= 0:
            return 0.5
        return 0.5 + 0.5 * count / max_count
    raise ValueError(f"esquema de TF desconhecido: {scheme!r} (válidos: {_SCHEMES})")


@dataclass(frozen=True)
class TFIDF:
    """Ranker TF-IDF com similaridade de cosseno sobre um campo do índice.

    Parâmetros:
        index:     índice invertido já construído.
        field:     campo sobre o qual pontuar (ex.: "title", "album").
        tf_scheme: variante de tf — "raw", "log" (padrão) ou "augmented".

    Normas dos documentos são pré-calculadas na primeira consulta e
    memoizadas; isso paga um custo único proporcional ao tamanho do
    vocabulário do campo e mantém o ranking `O(|query| · |candidatos|)`.
    """

    index: InvertedIndex
    field: str
    tf_scheme: TfScheme = "log"

    def __post_init__(self) -> None:
        if self.field not in self.index.fields:
            raise KeyError(f"campo desconhecido: {self.field!r} (válidos: {self.index.fields})")
        if self.tf_scheme not in _SCHEMES:
            raise ValueError(f"tf_scheme inválido: {self.tf_scheme!r} (válidos: {_SCHEMES})")

    @cached_property
    def idfs(self) -> dict[str, float]:
        """IDF de todos os termos do vocabulário do campo (memoizado)."""
        n = self.index.num_docs
        return {
            term: tfidf_idf(n, self.index.df(self.field, term))
            for term in self.index.vocabulary(self.field)
        }

    @cached_property
    def _max_tf_per_doc(self) -> list[int]:
        """Maior tf observado por documento — usado no esquema augmented."""
        max_tf = [0] * self.index.num_docs
        for postings in self.index.postings[self.field].values():
            for doc_id, tf in postings:
                if tf > max_tf[doc_id]:
                    max_tf[doc_id] = tf
        return max_tf

    @cached_property
    def doc_norms(self) -> list[float]:
        """||v_d|| para cada doc — pré-calculado percorrendo o vocabulário uma vez."""
        sq = [0.0] * self.index.num_docs
        max_tf = self._max_tf_per_doc if self.tf_scheme == "augmented" else None
        for term, postings in self.index.postings[self.field].items():
            idf = self.idfs.get(term, 0.0)
            if idf == 0.0:
                continue
            for doc_id, tf in postings:
                m = max_tf[doc_id] if max_tf is not None else 0
                w = tf_weight(tf, self.tf_scheme, m) * idf
                sq[doc_id] += w * w
        return [math.sqrt(v) for v in sq]

    def idf(self, term: str) -> float:
        """IDF(term) no campo; 0 para termos fora do vocabulário."""
        return self.idfs.get(term, 0.0)

    def score(self, query_terms: Iterable[str], doc_id: int) -> float:
        """Similaridade de cosseno entre o vetor do doc e o vetor da query."""
        if not 0 <= doc_id < self.index.num_docs:
            raise IndexError(f"doc_id fora do intervalo: {doc_id}")
        tokens = list(query_terms)
        if not tokens:
            return 0.0
        tf_q = _count_terms(tokens)
        q_max = max(tf_q.values())
        q_weights: dict[str, float] = {}
        q_norm_sq = 0.0
        for term, count in tf_q.items():
            idf = self.idfs.get(term, 0.0)
            if idf == 0.0:
                continue
            w = tf_weight(count, self.tf_scheme, q_max) * idf
            q_weights[term] = w
            q_norm_sq += w * w
        if q_norm_sq == 0.0:
            return 0.0
        d_norm = self.doc_norms[doc_id]
        if d_norm == 0.0:
            return 0.0
        doc_max = self._max_tf_per_doc[doc_id] if self.tf_scheme == "augmented" else 0
        dot = 0.0
        for term, wq in q_weights.items():
            tf = _tf_for_doc(self.index.get_postings(self.field, term), doc_id)
            if tf == 0:
                continue
            wd = tf_weight(tf, self.tf_scheme, doc_max) * self.idfs[term]
            dot += wq * wd
        return dot / (d_norm * math.sqrt(q_norm_sq))

    def rank(
        self,
        query: str | Sequence[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Pontua e devolve os `top_k` melhores `(external_id, cosine_sim)`."""
        if top_k <= 0:
            raise ValueError(f"top_k deve ser > 0 (recebido {top_k})")
        tokens = _coerce_query(query)
        if not tokens:
            return []
        tf_q = _count_terms(tokens)
        q_max = max(tf_q.values())
        q_weights: dict[str, float] = {}
        q_norm_sq = 0.0
        for term, count in tf_q.items():
            idf = self.idfs.get(term, 0.0)
            if idf == 0.0:
                continue
            w = tf_weight(count, self.tf_scheme, q_max) * idf
            q_weights[term] = w
            q_norm_sq += w * w
        if q_norm_sq == 0.0:
            return []
        q_norm = math.sqrt(q_norm_sq)
        term_tfs: list[tuple[str, float, dict[int, int]]] = []
        candidates: set[int] = set()
        for term, wq in q_weights.items():
            postings = self.index.get_postings(self.field, term)
            if not postings:
                continue
            tfs = dict(postings)
            term_tfs.append((term, wq, tfs))
            candidates.update(tfs)
        if not candidates:
            return []
        doc_norms = self.doc_norms
        max_tf = self._max_tf_per_doc if self.tf_scheme == "augmented" else None
        scored: list[tuple[int, float]] = []
        for doc_id in candidates:
            d_norm = doc_norms[doc_id]
            if d_norm == 0.0:
                continue
            doc_max = max_tf[doc_id] if max_tf is not None else 0
            dot = 0.0
            for term, wq, tfs in term_tfs:
                tf = tfs.get(doc_id)
                if not tf:
                    continue
                wd = tf_weight(tf, self.tf_scheme, doc_max) * self.idfs[term]
                dot += wq * wd
            if dot > 0:
                scored.append((doc_id, dot / (d_norm * q_norm)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        top = scored[:top_k]
        return [(self.index.external_id(doc_id), score) for doc_id, score in top]


@dataclass(frozen=True)
class BM25:
    """Ranker Okapi BM25 sobre um campo de um `InvertedIndex`.

    Parâmetros:
        index: índice invertido já construído.
        field: campo sobre o qual pontuar (ex.: "title", "album").
        k1:   saturação de tf; >0. Valores maiores aproximam o comportamento
              linear em tf; típico 1.2 a 2.0.
        b:    normalização pelo tamanho do documento em [0, 1]. b=0 ignora
              o tamanho; b=1 normaliza completamente pela média.
    """

    index: InvertedIndex
    field: str
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.field not in self.index.fields:
            raise KeyError(f"campo desconhecido: {self.field!r} (válidos: {self.index.fields})")
        if self.k1 <= 0:
            raise ValueError(f"k1 deve ser > 0 (recebido {self.k1})")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError(f"b deve estar em [0, 1] (recebido {self.b})")

    @cached_property
    def avgdl(self) -> float:
        return self.index.avg_doc_length(self.field)

    def idf(self, term: str) -> float:
        """IDF(term) no campo. Termo fora do vocabulário recebe IDF máximo."""
        return bm25_idf(self.index.num_docs, self.index.df(self.field, term))

    def score(self, query_terms: Iterable[str], doc_id: int) -> float:
        """Pontua um único documento (`doc_id` interno) para os termos da query.

        A query deve ser passada já pré-processada (mesma normalização do
        índice). Use `rank` para receber a string bruta.
        """
        if not 0 <= doc_id < self.index.num_docs:
            raise IndexError(f"doc_id fora do intervalo: {doc_id}")
        dl = self.index.doc_length(self.field, doc_id)
        avgdl = self.avgdl or 1.0  # avgdl=0 só acontece em índice vazio
        length_norm = 1.0 - self.b + self.b * dl / avgdl
        total = 0.0
        for term in query_terms:
            tf = _tf_for_doc(self.index.get_postings(self.field, term), doc_id)
            if tf == 0:
                continue
            idf = self.idf(term)
            total += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
        return total

    def rank(
        self,
        query: str | Sequence[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Pontua e devolve os `top_k` melhores `(external_id, score)`.

        Aceita a query como string crua (será pré-processada com o mesmo
        pipeline do índice) ou como lista de tokens já pré-processados.
        Apenas documentos que contêm ao menos um termo da query são
        pontuados — todos os demais têm score 0 e são omitidos.
        """
        if top_k <= 0:
            raise ValueError(f"top_k deve ser > 0 (recebido {top_k})")
        terms = list(_coerce_query(query))
        if not terms:
            return []
        # Agrega tf por documento candidato, percorrendo posting lists uma única vez.
        term_tfs: list[tuple[str, dict[int, int]]] = []
        candidates: set[int] = set()
        for term in terms:
            postings = self.index.get_postings(self.field, term)
            if not postings:
                continue
            tfs = dict(postings)
            term_tfs.append((term, tfs))
            candidates.update(tfs)
        if not candidates:
            return []
        avgdl = self.avgdl or 1.0
        scored: list[tuple[int, float]] = []
        for doc_id in candidates:
            dl = self.index.doc_length(self.field, doc_id)
            length_norm = 1.0 - self.b + self.b * dl / avgdl
            total = 0.0
            for term, tfs in term_tfs:
                tf = tfs.get(doc_id)
                if not tf:
                    continue
                idf = self.idf(term)
                total += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
            if total > 0:
                scored.append((doc_id, total))
        # Desempate por doc_id para estabilidade de ordenação.
        scored.sort(key=lambda item: (-item[1], item[0]))
        top = scored[:top_k]
        return [(self.index.external_id(doc_id), score) for doc_id, score in top]


def _coerce_query(query: str | Sequence[str]) -> list[str]:
    if isinstance(query, str):
        return preprocess(query)
    return list(query)


def _count_terms(tokens: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts


def _tf_for_doc(postings: list[tuple[int, int]], doc_id: int) -> int:
    # Postings são ordenadas por doc_id; busca binária mantém score() O(log n).
    lo, hi = 0, len(postings)
    while lo < hi:
        mid = (lo + hi) // 2
        mid_doc, mid_tf = postings[mid]
        if mid_doc == doc_id:
            return mid_tf
        if mid_doc < doc_id:
            lo = mid + 1
        else:
            hi = mid
    return 0
