"""Testes do pipeline de pré-processamento (issue #3).

Os testes nascem marcados como xfail enquanto as funções são stubs; à medida
que cada etapa for implementada, remover o marker correspondente.
"""

from __future__ import annotations

import pytest

from music_search import preprocessing


@pytest.mark.xfail(reason="normalize ainda é stub", strict=True, raises=NotImplementedError)
def test_normalize_lowercase_e_acentos() -> None:
    assert preprocessing.normalize("Canção DE Ninar") == "cancao de ninar"


@pytest.mark.xfail(reason="tokenize ainda é stub", strict=True, raises=NotImplementedError)
def test_tokenize_separa_palavras() -> None:
    assert preprocessing.tokenize("hello world") == ["hello", "world"]


@pytest.mark.xfail(reason="remove_stopwords ainda é stub", strict=True, raises=NotImplementedError)
def test_remove_stopwords_pt_en() -> None:
    tokens = ["the", "a", "music", "de", "amor"]
    assert preprocessing.remove_stopwords(tokens) == ["music", "amor"]


@pytest.mark.xfail(reason="stem ainda é stub", strict=True, raises=NotImplementedError)
def test_stem_portugues() -> None:
    assert preprocessing.stem(["cantando", "amores"]) == ["cant", "amor"]


@pytest.mark.xfail(reason="preprocess ainda é stub", strict=True, raises=NotImplementedError)
def test_preprocess_pipeline_completo() -> None:
    resultado = preprocessing.preprocess("As Canções de Amor")
    assert "amor" in resultado
