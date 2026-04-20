"""Pipeline de pré-processamento de texto para indexação e busca.

Etapas (issue #3):
    1. Normalização (lowercase, remoção de acentos e caracteres especiais)
    2. Tokenização
    3. Remoção de stopwords (pt/en)
    4. Stemming (RSLP para português, Snowball para inglês)

A função `preprocess` orquestra as etapas na ordem canônica e é o ponto
de entrada usado pelo indexer e pelo processamento de queries.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import RSLPStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

Language = str

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Aplica lowercase, remove acentos (NFD) e caracteres não alfanuméricos."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = _NON_ALNUM.sub(" ", without_marks)
    return _WHITESPACE.sub(" ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    """Divide o texto em tokens usando NLTK (punkt_tab)."""
    tokens: list[str] = word_tokenize(text)
    return tokens


@lru_cache(maxsize=8)
def _stopwords_for(languages: tuple[Language, ...]) -> frozenset[str]:
    result: set[str] = set()
    for lang in languages:
        result.update(nltk_stopwords.words(lang))
    return frozenset(result)


def remove_stopwords(
    tokens: Iterable[str],
    languages: Iterable[Language] = ("portuguese", "english"),
) -> list[str]:
    """Remove stopwords das linguagens fornecidas."""
    sw = _stopwords_for(tuple(languages))
    return [t for t in tokens if t not in sw]


@lru_cache(maxsize=8)
def _stemmer_for(language: Language) -> Any:
    if language == "portuguese":
        return RSLPStemmer()
    return SnowballStemmer(language)


def stem(tokens: Iterable[str], language: Language = "portuguese") -> list[str]:
    """Aplica stemming (RSLP para português, Snowball para demais idiomas)."""
    stemmer = _stemmer_for(language)
    return [str(stemmer.stem(t)) for t in tokens]


def preprocess(
    text: str,
    languages: Iterable[Language] = ("portuguese", "english"),
) -> list[str]:
    """Executa o pipeline: normalize → tokenize → remove_stopwords → stem.

    O primeiro idioma de `languages` define o stemmer usado; todos os
    idiomas são considerados na remoção de stopwords.
    """
    langs = tuple(languages)
    if not langs:
        raise ValueError("languages não pode estar vazio")
    normalized = normalize(text)
    tokens = tokenize(normalized)
    filtered = remove_stopwords(tokens, langs)
    return stem(filtered, langs[0])
