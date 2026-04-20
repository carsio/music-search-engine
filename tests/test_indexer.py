"""Testes do índice invertido multi-campo (issue #4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_search.indexer import IndexBuilder, InvertedIndex, build_index

DOCS = [
    {"id": "t1", "title": "As Canções de Amor", "artist": "Roberto Carlos", "album": "Emoções"},
    {"id": "t2", "title": "Amor de Carnaval", "artist": "Marisa Monte", "album": "Verdade"},
    {"id": "t3", "title": "Running on Empty", "artist": "Jackson Browne", "album": "Running"},
]
FIELDS = ("title", "artist", "album")


def _index() -> InvertedIndex:
    return build_index(DOCS, fields=FIELDS)


def test_builder_exige_fields_nao_vazios() -> None:
    with pytest.raises(ValueError):
        IndexBuilder(fields=())


def test_num_docs_reflete_documentos_adicionados() -> None:
    idx = _index()
    assert idx.num_docs == 3


def test_external_id_mapeia_id_interno() -> None:
    idx = _index()
    assert [idx.external_id(i) for i in range(idx.num_docs)] == ["t1", "t2", "t3"]


def test_postings_agregam_tf_no_mesmo_doc() -> None:
    # Stem de "amor" em RSLP vira "am"; aparece 1x no título de t1 e 1x em t2.
    idx = _index()
    postings = idx.get_postings("title", "am")
    assert postings == [(0, 1), (1, 1)]


def test_tf_conta_repeticoes_no_mesmo_campo() -> None:
    docs = [{"id": "d1", "title": "amor amor amor"}]
    idx = build_index(docs, fields=("title",))
    assert idx.get_postings("title", "am") == [(0, 3)]


def test_df_conta_documentos_distintos() -> None:
    idx = _index()
    assert idx.df("title", "am") == 2  # stem de "amor"
    assert idx.df("title", "canco") == 1  # stem de "canções"
    assert idx.df("title", "inexistente") == 0


def test_isolamento_entre_campos() -> None:
    # "running" aparece no título e no álbum de t3 — cada campo tem sua
    # própria posting list, e o termo não aparece entre os artistas.
    idx = _index()
    assert idx.df("title", "running") == 1
    assert idx.df("album", "running") == 1
    assert idx.df("artist", "running") == 0


def test_preprocess_normaliza_termos_no_indice() -> None:
    idx = _index()
    # Termo bruto "Canções" deve ser normalizado/stemizado antes de indexar.
    assert "cancoes" not in list(idx.vocabulary("title"))
    assert "canco" in list(idx.vocabulary("title"))


def test_doc_length_em_tokens_apos_preprocessing() -> None:
    # "As Canções de Amor": stopwords "as"/"de" saem, restam 2 stems.
    idx = _index()
    assert idx.doc_length("title", 0) == 2


def test_avg_doc_length_por_campo() -> None:
    idx = _index()
    lengths = [idx.doc_length("title", i) for i in range(idx.num_docs)]
    assert idx.avg_doc_length("title") == pytest.approx(sum(lengths) / len(lengths))


def test_campo_desconhecido_levanta_keyerror() -> None:
    idx = _index()
    with pytest.raises(KeyError):
        idx.get_postings("genre", "pop")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    idx = _index()
    target = tmp_path / "idx" / "spotify.pkl"
    idx.save(target)
    restored = InvertedIndex.load(target)
    assert restored.fields == idx.fields
    assert restored.doc_ids == idx.doc_ids
    assert restored.doc_lengths == idx.doc_lengths
    assert restored.get_postings("title", "amor") == idx.get_postings("title", "amor")


def test_load_falha_em_arquivo_incompativel(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.pkl"
    import pickle

    bogus.write_bytes(pickle.dumps({"not": "an index"}))
    with pytest.raises(TypeError):
        InvertedIndex.load(bogus)


def test_extend_exige_chave_id() -> None:
    builder = IndexBuilder(fields=("title",))
    with pytest.raises(ValueError):
        builder.extend([{"title": "sem id"}])


def test_campos_ausentes_no_doc_viram_strings_vazias() -> None:
    docs = [{"id": "d1", "title": "so titulo"}]  # sem artist/album
    idx = build_index(docs, fields=("title", "artist"))
    assert idx.doc_length("artist", 0) == 0
    assert list(idx.vocabulary("artist")) == []


def test_postings_ordenados_por_doc_id() -> None:
    docs = [{"id": f"d{i}", "title": "rock"} for i in range(5)]
    idx = build_index(docs, fields=("title",))
    postings = idx.get_postings("title", "rock")
    assert [doc_id for doc_id, _ in postings] == sorted(doc_id for doc_id, _ in postings)
