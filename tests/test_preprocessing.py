"""Testes do pipeline de pré-processamento (issue #3)."""

from __future__ import annotations

from music_search import preprocessing


def test_normalize_lowercase_e_acentos() -> None:
    assert preprocessing.normalize("Canção DE Ninar") == "cancao de ninar"


def test_normalize_remove_pontuacao() -> None:
    assert preprocessing.normalize("Rock'n Roll!!!") == "rock n roll"


def test_tokenize_separa_palavras() -> None:
    assert preprocessing.tokenize("hello world") == ["hello", "world"]


def test_remove_stopwords_pt_en() -> None:
    tokens = ["the", "a", "music", "de", "amor"]
    assert preprocessing.remove_stopwords(tokens) == ["music", "amor"]


def test_remove_stopwords_respeita_idiomas_customizados() -> None:
    # Apenas inglês: "de" (stopword pt) sobrevive; "the" (stopword en) sai.
    tokens = ["the", "de", "music"]
    assert preprocessing.remove_stopwords(tokens, languages=["english"]) == ["de", "music"]


def test_stem_portugues_colapsa_flexoes() -> None:
    stems = preprocessing.stem(["cantando", "cantaram", "cantar"])
    assert stems[0] == stems[1] == stems[2] == "cant"


def test_stem_ingles_snowball() -> None:
    assert preprocessing.stem(["running", "jumps"], language="english") == ["run", "jump"]


def test_preprocess_pipeline_completo() -> None:
    resultado = preprocessing.preprocess("As Canções de Amor")
    assert len(resultado) == 2
    assert "as" not in resultado
    assert "de" not in resultado


def test_preprocess_texto_vazio_retorna_lista_vazia() -> None:
    assert preprocessing.preprocess("") == []
