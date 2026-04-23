"""Smoke tests para o módulo de busca vetorial.

Não dependem de Milvus nem Ollama rodando — tudo é mockado.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from music_search.vector.config import EmbeddingConfig
from music_search.vector.indexing import (
    _safe_bool,
    _safe_int,
    row_to_json,
    row_to_text,
    truncate,
)
from music_search.vector.search import VectorSearch, _format_hits

# ── Helpers de serialização ───────────────────────────────────────────────────


def test_row_to_text_omits_empty_fields() -> None:
    row = {
        "track_name": "Imagine",
        "artist_names": "John Lennon",
        "album_name": "",
        "artist_genres": None,
        "album_type": "album",
    }
    text = row_to_text(row)
    assert "track: Imagine" in text
    assert "artists: John Lennon" in text
    assert "album_name" not in text  # campo vazio foi omitido
    assert "genres" not in text


def test_row_to_text_includes_numeric_defaults() -> None:
    # Campos numéricos com valor 0 são tratados como "vazios" para não
    # poluir o embedding com zeros redundantes.
    row = {"track_name": "X"}
    text = row_to_text(row)
    assert "track: X" in text
    assert "explicit" not in text  # default False é omitido
    assert "duration_ms" not in text


def test_truncate_respects_utf8_boundaries() -> None:
    s = "café com bolo"  # 'é' ocupa 2 bytes em UTF-8
    truncated = truncate(s, 5)
    # Não pode quebrar o 'é' em dois
    assert truncated.encode("utf-8").decode("utf-8") == truncated


def test_truncate_keeps_short_strings_intact() -> None:
    assert truncate("hi", 100) == "hi"


def test_truncate_handles_none() -> None:
    assert truncate(None, 10) == ""


def test_safe_int_falls_back_to_zero() -> None:
    assert _safe_int("42") == 42
    assert _safe_int(None) == 0
    assert _safe_int("abc") == 0


def test_safe_bool_handles_strings() -> None:
    assert _safe_bool("true") is True
    assert _safe_bool("false") is False
    assert _safe_bool("yes") is True
    assert _safe_bool(None) is False


def test_row_to_json_roundtrip() -> None:
    import json

    row = {"a": 1, "b": "texto", "c": None}
    assert json.loads(row_to_json(row)) == row


# ── EmbeddingConfig ───────────────────────────────────────────────────────────


def test_embedding_config_defaults_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_OLLAMA", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    cfg = EmbeddingConfig.from_env()
    assert cfg.use_ollama is True
    assert cfg.model == "nomic-embed-text"
    assert cfg.dim == 768


def test_embedding_config_uses_openai_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USE_OLLAMA", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = EmbeddingConfig.from_env()
    assert cfg.use_ollama is False
    assert cfg.model == "text-embedding-3-small"
    assert cfg.dim == 1536


def test_embedding_config_falls_back_to_ollama_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # USE_OLLAMA=false mas sem OPENAI_API_KEY: cai no Ollama como salvaguarda.
    monkeypatch.setenv("USE_OLLAMA", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = EmbeddingConfig.from_env()
    assert cfg.use_ollama is True


# ── Formatação de resultados ──────────────────────────────────────────────────


def test_format_hits_parses_json_data() -> None:
    hits = [
        {
            "distance": 0.85,
            "entity": {
                "track_name": "Imagine",
                "artist_names": "John Lennon",
                "json_data": '{"id": "abc", "extra": 42}',
            },
        }
    ]
    out = _format_hits(hits)
    assert len(out) == 1
    assert out[0]["rank"] == 1
    assert out[0]["score"] == 0.85
    assert out[0]["track_name"] == "Imagine"
    assert out[0]["data_completa"] == {"id": "abc", "extra": 42}


def test_format_hits_survives_invalid_json() -> None:
    hits = [{"distance": 0.5, "entity": {"track_name": "X", "json_data": "{not valid"}}]
    out = _format_hits(hits)
    assert out[0]["data_completa"] == {}


def test_format_hits_ranks_in_order() -> None:
    hits = [
        {"distance": 0.9, "entity": {"track_name": "A", "json_data": "{}"}},
        {"distance": 0.7, "entity": {"track_name": "B", "json_data": "{}"}},
        {"distance": 0.5, "entity": {"track_name": "C", "json_data": "{}"}},
    ]
    out = _format_hits(hits)
    assert [r["rank"] for r in out] == [1, 2, 3]
    assert [r["track_name"] for r in out] == ["A", "B", "C"]


# ── VectorSearch ──────────────────────────────────────────────────────────────


def test_vector_search_rejects_empty_query() -> None:
    vs = VectorSearch(milvus_uri="unused://test")
    with pytest.raises(ValueError):
        vs.search("")
    with pytest.raises(ValueError):
        vs.search("   ")


def test_vector_search_uses_mocked_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garante que search() pluga embed + Milvus corretamente quando mockados."""
    vs = VectorSearch(
        milvus_uri="unused://test",
        embedding_config=EmbeddingConfig(
            use_ollama=True,
            model="fake-model",
            dim=4,
            openai_api_key=None,
            ollama_url="http://fake",
        ),
    )

    fake_embed_client = MagicMock()
    fake_embed_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3, 0.4])]
    )
    vs._embed_client = fake_embed_client  # injeta cliente mock

    fake_milvus = MagicMock()
    fake_milvus.search.return_value = [
        [
            {
                "distance": 0.9,
                "entity": {
                    "track_name": "Imagine",
                    "artist_names": "John Lennon",
                    "json_data": "{}",
                },
            }
        ]
    ]
    vs._milvus = fake_milvus  # injeta cliente Milvus mock

    results = vs.search("imagine peace", top_k=5)

    assert len(results) == 1
    assert results[0]["track_name"] == "Imagine"
    assert results[0]["rank"] == 1
    fake_embed_client.embeddings.create.assert_called_once()
    fake_milvus.search.assert_called_once()
