"""Pipeline de pré-processamento de texto para indexação e busca.

Etapas previstas (issue #3):
    1. Normalização (lowercase, remoção de acentos e caracteres especiais)
    2. Tokenização
    3. Remoção de stopwords (pt/en)
    4. Stemming / lemmatização

A função `preprocess` orquestra as etapas na ordem canônica e é o ponto
de entrada usado pelo indexer e pelo processamento de queries.
"""

from __future__ import annotations

from collections.abc import Iterable

Language = str


def normalize(text: str) -> str:
    """Aplica lowercase, remove acentos e caracteres não alfanuméricos."""
    raise NotImplementedError


def tokenize(text: str) -> list[str]:
    """Divide o texto em tokens usando NLTK (punkt_tab)."""
    raise NotImplementedError


def remove_stopwords(
    tokens: Iterable[str],
    languages: Iterable[Language] = ("portuguese", "english"),
) -> list[str]:
    """Remove stopwords das linguagens fornecidas."""
    raise NotImplementedError


def stem(tokens: Iterable[str], language: Language = "portuguese") -> list[str]:
    """Aplica stemming (RSLP para português, Snowball para inglês)."""
    raise NotImplementedError


def preprocess(text: str, languages: Iterable[Language] = ("portuguese", "english")) -> list[str]:
    """Executa o pipeline completo: normalize -> tokenize -> remove_stopwords -> stem."""
    raise NotImplementedError
