"""Modelos de ranking: Okapi BM25 sobre o índice invertido multi-campo.

BM25 (issue #6): pontua documentos combinando saturação de tf (controlada
por `k1`) e normalização pelo tamanho do documento (controlada por `b`).

Fórmula (variante com IDF positivo — "BM25+smooth"):

    score(D, Q) = Σ_{t ∈ Q}  IDF(t) · (tf(t, D) · (k1 + 1))
                             / (tf(t, D) + k1 · (1 - b + b · |D|/avgdl))

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5)  +  1 )

Valores padrão (`k1=1.5`, `b=0.75`) seguem a convenção do Elasticsearch/
Lucene; são bons pontos de partida antes de calibração fina.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

from music_search.indexer import InvertedIndex
from music_search.preprocessing import preprocess


def bm25_idf(num_docs: int, df: int) -> float:
    """IDF do BM25 com suavização +0.5 e +1 para garantir valores não negativos."""
    return math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)


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
