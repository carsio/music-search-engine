"""Índice invertido multi-campo (issue #4).

Estrutura:
    postings[field][term] = [(doc_id, tf), ...]   # ordenado por doc_id
    doc_lengths[field][doc_id] = int              # nº de tokens no campo após pré-processamento
    doc_ids[doc_id] = external_id                 # mapeia id interno (int) -> id externo (str)

A numeração interna dos documentos é um inteiro contíguo atribuído na ordem
em que os docs são adicionados ao builder. Esse id é denso e pequeno, o que
ajuda os modelos de ranking (TF-IDF, BM25) e facilita a persistência.

Persistência via `pickle` — mais rápido e preserva tipos Python; o índice
não é um contrato público, então o binário opaco é aceitável.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from music_search.preprocessing import preprocess

Tokenizer = Callable[[str], list[str]]


@dataclass
class InvertedIndex:
    """Índice invertido imutável após construção (use `IndexBuilder` para criar)."""

    fields: tuple[str, ...]
    doc_ids: list[str]
    postings: dict[str, dict[str, list[tuple[int, int]]]]
    doc_lengths: dict[str, list[int]]

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    def get_postings(self, field: str, term: str) -> list[tuple[int, int]]:
        """Retorna a posting list `[(doc_id, tf), ...]` para (campo, termo)."""
        self._require_field(field)
        return self.postings[field].get(term, [])

    def df(self, field: str, term: str) -> int:
        """Document frequency: em quantos documentos do campo o termo aparece."""
        return len(self.get_postings(field, term))

    def doc_length(self, field: str, doc_id: int) -> int:
        self._require_field(field)
        return self.doc_lengths[field][doc_id]

    def avg_doc_length(self, field: str) -> float:
        self._require_field(field)
        lengths = self.doc_lengths[field]
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

    def external_id(self, doc_id: int) -> str:
        return self.doc_ids[doc_id]

    def vocabulary(self, field: str) -> Iterable[str]:
        self._require_field(field)
        return self.postings[field].keys()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> InvertedIndex:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"arquivo não contém {cls.__name__}: {type(obj).__name__}")
        return obj

    def _require_field(self, field: str) -> None:
        if field not in self.fields:
            raise KeyError(f"campo desconhecido: {field!r} (válidos: {self.fields})")


@dataclass
class IndexBuilder:
    """Construtor incremental do índice invertido.

    Use `add(doc_id, {campo: texto, ...})` para alimentar e `build()` para
    obter o índice final. O tokenizer padrão é `preprocessing.preprocess`.
    """

    fields: tuple[str, ...]
    tokenizer: Tokenizer = field(default=preprocess)
    _doc_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _postings: dict[str, dict[str, dict[int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _doc_lengths: dict[str, list[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("fields não pode estar vazio")
        self.fields = tuple(self.fields)
        self._postings = {f: defaultdict(dict) for f in self.fields}
        self._doc_lengths = {f: [] for f in self.fields}

    @property
    def num_docs(self) -> int:
        return len(self._doc_ids)

    def add(self, doc_id: str, values: Mapping[str, str | None]) -> int:
        """Indexa um documento e devolve seu id interno."""
        internal_id = len(self._doc_ids)
        self._doc_ids.append(doc_id)
        for fname in self.fields:
            text = values.get(fname) or ""
            tokens = self.tokenizer(text)
            self._doc_lengths[fname].append(len(tokens))
            term_map = self._postings[fname]
            for term in tokens:
                per_doc = term_map[term]
                per_doc[internal_id] = per_doc.get(internal_id, 0) + 1
        return internal_id

    def extend(self, docs: Iterable[Mapping[str, str | None]], id_key: str = "id") -> int:
        """Atalho para adicionar um iterável de docs com id em `id_key`.

        Retorna a quantidade de documentos adicionados.
        """
        added = 0
        for doc in docs:
            raw_id = doc.get(id_key)
            if raw_id is None:
                raise ValueError(f"documento sem chave {id_key!r}: {doc!r}")
            self.add(str(raw_id), doc)
            added += 1
        return added

    def build(self) -> InvertedIndex:
        compact: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for fname, term_map in self._postings.items():
            compact[fname] = {term: sorted(doc_map.items()) for term, doc_map in term_map.items()}
        return InvertedIndex(
            fields=self.fields,
            doc_ids=list(self._doc_ids),
            postings=compact,
            doc_lengths={f: list(v) for f, v in self._doc_lengths.items()},
        )


def build_index(
    docs: Iterable[Mapping[str, str | None]],
    fields: Sequence[str],
    id_key: str = "id",
    tokenizer: Tokenizer = preprocess,
) -> InvertedIndex:
    """Atalho funcional: constrói um índice a partir de um iterável de docs."""
    builder = IndexBuilder(fields=tuple(fields), tokenizer=tokenizer)
    builder.extend(docs, id_key=id_key)
    return builder.build()
