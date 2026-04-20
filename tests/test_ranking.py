"""Testes dos rankers TF-IDF (issue #5) e BM25 (issue #6)."""

from __future__ import annotations

import math

import pytest

from music_search.indexer import build_index
from music_search.ranking import BM25, TFIDF, bm25_idf, tf_weight, tfidf_idf


def _make_index() -> object:
    docs = [
        {"id": "t1", "title": "As Canções de Amor", "artist": "Roberto Carlos", "album": "Emoções"},
        {"id": "t2", "title": "Amor de Carnaval", "artist": "Marisa Monte", "album": "Verdade"},
        {"id": "t3", "title": "Running on Empty", "artist": "Jackson Browne", "album": "Running"},
        {"id": "t4", "title": "Amor Eterno Amor", "artist": "Zeca Pagodinho", "album": "Amor"},
    ]
    return build_index(docs, fields=("title", "artist", "album"))


def test_bm25_idf_nunca_negativo_e_cresce_com_raridade() -> None:
    # Termos raros valem mais que termos frequentes.
    assert bm25_idf(100, 1) > bm25_idf(100, 50)
    # Termo em todos os docs ainda é estritamente positivo (graças ao +1).
    assert bm25_idf(100, 100) > 0


def test_bm25_rejeita_parametros_invalidos() -> None:
    idx = _make_index()
    with pytest.raises(KeyError):
        BM25(idx, field="genre")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BM25(idx, field="title", k1=0.0)
    with pytest.raises(ValueError):
        BM25(idx, field="title", b=1.5)


def test_bm25_ordena_docs_relevantes_acima_e_omite_nao_relevantes() -> None:
    idx = _make_index()
    ranker = BM25(idx, field="title")
    results = ranker.rank("amor", top_k=10)
    retrieved = {doc_id for doc_id, _ in results}
    # t1, t2, t4 mencionam "amor"; t3 não — e portanto não deve aparecer.
    assert retrieved == {"t1", "t2", "t4"}
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_score_cresce_com_tf_mas_satura() -> None:
    idx = _make_index()
    ranker = BM25(idx, field="title", k1=1.5, b=0.0)  # b=0 isola o efeito de tf.
    # t4 ("amor eterno amor") tem tf=2; t2 ("amor de carnaval") tem tf=1.
    scores = dict(ranker.rank("amor", top_k=10))
    assert scores["t4"] > scores["t2"]
    # Saturação: dobrar tf não dobra o score.
    assert scores["t4"] < 2 * scores["t2"]


def test_bm25_top_k_limita_resultados() -> None:
    idx = _make_index()
    ranker = BM25(idx, field="title")
    assert len(ranker.rank("amor", top_k=2)) == 2
    assert len(ranker.rank("amor", top_k=10)) == 3


def test_bm25_top_k_invalido() -> None:
    idx = _make_index()
    ranker = BM25(idx, field="title")
    with pytest.raises(ValueError):
        ranker.rank("amor", top_k=0)


def test_bm25_query_sem_matches_retorna_vazio() -> None:
    idx = _make_index()
    ranker = BM25(idx, field="title")
    assert ranker.rank("reggae", top_k=10) == []
    # String vazia → nenhum termo pré-processado → resultado vazio.
    assert ranker.rank("", top_k=10) == []


def test_bm25_aceita_tokens_ja_preprocessados() -> None:
    idx = _make_index()
    ranker = BM25(idx, field="title")
    # "amor" → stem RSLP "am". Passar tokens direto deve dar o mesmo ranking.
    via_string = ranker.rank("amor", top_k=10)
    via_tokens = ranker.rank(["am"], top_k=10)
    assert via_string == via_tokens


def test_bm25_normalizacao_por_tamanho_com_b_alto() -> None:
    # Com b=1, docs curtos com o mesmo tf pontuam mais que docs longos.
    docs = [
        {"id": "curto", "title": "amor"},
        {"id": "longo", "title": "amor fim de semana praia sol verao ferias"},
    ]
    idx = build_index(docs, fields=("title",))
    ranker = BM25(idx, field="title", b=1.0)
    scores = dict(ranker.rank("amor", top_k=10))
    assert scores["curto"] > scores["longo"]

    # Já com b=0 (sem normalização), o tamanho não importa e os scores empatam.
    ranker_flat = BM25(idx, field="title", b=0.0)
    scores_flat = dict(ranker_flat.rank("amor", top_k=10))
    assert scores_flat["curto"] == pytest.approx(scores_flat["longo"])


def test_bm25_score_bate_com_formula_manual() -> None:
    idx = _make_index()
    k1, b = 1.5, 0.75
    ranker = BM25(idx, field="title", k1=k1, b=b)
    # Calcula manualmente o score de t2 para "amor" (stem "am").
    term = "am"
    doc_internal = 1  # t2 é o 2º doc indexado.
    tf = 1
    df = idx.df("title", term)  # 3 docs: t1, t2, t4
    n = idx.num_docs
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
    dl = idx.doc_length("title", doc_internal)
    avgdl = idx.avg_doc_length("title")
    expected = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    assert ranker.score([term], doc_internal) == pytest.approx(expected)


def test_bm25_ranking_difere_entre_campos() -> None:
    idx = _make_index()
    # "running" aparece em title e album de t3, e só em album de t3 novamente.
    ranker_title = BM25(idx, field="title")
    ranker_album = BM25(idx, field="album")
    assert [d for d, _ in ranker_title.rank("running")] == ["t3"]
    assert [d for d, _ in ranker_album.rank("running")] == ["t3"]


def test_bm25_desempate_estavel_por_doc_id() -> None:
    docs = [{"id": f"d{i}", "title": "rock"} for i in range(5)]
    idx = build_index(docs, fields=("title",))
    ranker = BM25(idx, field="title")
    results = ranker.rank("rock", top_k=5)
    # Todos com mesmo score: ordem deve seguir doc_id interno (ordem de inserção).
    assert [doc_id for doc_id, _ in results] == [f"d{i}" for i in range(5)]


# -------------------- TF-IDF (#5) --------------------


def test_tfidf_idf_raro_vale_mais_que_frequente() -> None:
    assert tfidf_idf(100, 1) > tfidf_idf(100, 50)
    # Termo fora do vocabulário (df=0): IDF 0.
    assert tfidf_idf(100, 0) == 0.0
    # Termo em todos os docs: IDF 0 (log(1) = 0), comportamento clássico.
    assert tfidf_idf(100, 100) == pytest.approx(0.0)


def test_tf_weight_variantes() -> None:
    # raw: linear no count.
    assert tf_weight(3, "raw") == 3.0
    # log: 1 + ln(count), amortece.
    assert tf_weight(1, "log") == pytest.approx(1.0)
    assert tf_weight(math.e, "log") == pytest.approx(2.0)  # 1 + ln(e)
    # augmented: 0.5 + 0.5 * count/max_count.
    assert tf_weight(1, "augmented", max_count=2) == pytest.approx(0.75)
    assert tf_weight(2, "augmented", max_count=2) == pytest.approx(1.0)
    # count=0 zera em qualquer esquema.
    for scheme in ("raw", "log", "augmented"):
        assert tf_weight(0, scheme) == 0.0  # type: ignore[arg-type]
    # Esquema desconhecido → erro.
    with pytest.raises(ValueError):
        tf_weight(1, "exotic")  # type: ignore[arg-type]


def test_tfidf_rejeita_parametros_invalidos() -> None:
    idx = _make_index()
    with pytest.raises(KeyError):
        TFIDF(idx, field="genre")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TFIDF(idx, field="title", tf_scheme="quadratic")  # type: ignore[arg-type]


def test_tfidf_cosseno_em_intervalo_unitario() -> None:
    idx = _make_index()
    ranker = TFIDF(idx, field="title")
    for _, score in ranker.rank("amor", top_k=10):
        assert 0.0 < score <= 1.0 + 1e-9


def test_tfidf_doc_identico_a_query_tem_cosseno_um() -> None:
    docs = [
        {"id": "a", "title": "rock pop jazz"},
        {"id": "b", "title": "funk samba"},
    ]
    idx = build_index(docs, fields=("title",))
    ranker = TFIDF(idx, field="title", tf_scheme="log")
    results = dict(ranker.rank("rock pop jazz", top_k=10))
    assert results["a"] == pytest.approx(1.0)


def test_tfidf_ordena_docs_relevantes_e_omite_nao_relevantes() -> None:
    idx = _make_index()
    ranker = TFIDF(idx, field="title")
    results = ranker.rank("amor", top_k=10)
    retrieved = {doc_id for doc_id, _ in results}
    # t1, t2, t4 mencionam "amor"; t3 não.
    assert retrieved == {"t1", "t2", "t4"}
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_tfidf_tf_maior_aumenta_score_no_scheme_raw() -> None:
    idx = _make_index()
    # No scheme "raw" com cosseno, dobrar tf no doc pode compensar a norma:
    # validamos apenas que t4 (tf=2 de "amor") supera t2 (tf=1).
    ranker = TFIDF(idx, field="title", tf_scheme="raw")
    scores = dict(ranker.rank("amor", top_k=10))
    assert scores["t4"] > scores["t2"]


def test_tfidf_query_sem_matches_retorna_vazio() -> None:
    idx = _make_index()
    ranker = TFIDF(idx, field="title")
    assert ranker.rank("reggae", top_k=10) == []
    assert ranker.rank("", top_k=10) == []


def test_tfidf_top_k_invalido() -> None:
    idx = _make_index()
    ranker = TFIDF(idx, field="title")
    with pytest.raises(ValueError):
        ranker.rank("amor", top_k=0)


def test_tfidf_aceita_tokens_pre_processados() -> None:
    idx = _make_index()
    ranker = TFIDF(idx, field="title")
    assert ranker.rank("amor", top_k=10) == ranker.rank(["am"], top_k=10)


def test_tfidf_score_bate_com_formula_manual() -> None:
    # Validação numérica do cosseno com scheme "log" em um índice pequeno.
    docs = [
        {"id": "a", "title": "rock pop"},
        {"id": "b", "title": "rock jazz"},
    ]
    idx = build_index(docs, fields=("title",))
    ranker = TFIDF(idx, field="title", tf_scheme="log")
    n = idx.num_docs
    # "rock" aparece nos 2 docs → idf=0 → não contribui para nada.
    # Fica "pop" (df=1) contribuindo apenas em a.
    idf_pop = math.log(n / 1)
    idf_jazz = math.log(n / 1)
    w_pop_a = (1.0 + math.log(1)) * idf_pop  # tf=1 em a
    w_rock_a = 0.0  # idf=0
    norm_a = math.sqrt(w_pop_a**2 + w_rock_a**2)
    # Query "pop":
    wq_pop = (1.0 + math.log(1)) * idf_pop
    norm_q = math.sqrt(wq_pop**2)
    expected_a = (wq_pop * w_pop_a) / (norm_a * norm_q)
    results = dict(ranker.rank("pop"))
    assert results["a"] == pytest.approx(expected_a)
    # b não deve aparecer: rock tem idf=0 e "pop" não existe em b.
    assert "b" not in results
    # Sanidade cruzada: "jazz" só em b, cosseno deve ser positivo.
    assert ranker.rank("jazz")[0][0] == "b"
    assert idf_jazz > 0


def test_tfidf_desempate_estavel_por_doc_id() -> None:
    # Corpus misto: "rock" em 5 docs, "pop" em outros 3 (idf("rock") > 0).
    docs = [{"id": f"d{i}", "title": "rock"} for i in range(5)]
    docs += [{"id": f"x{i}", "title": "pop"} for i in range(3)]
    idx = build_index(docs, fields=("title",))
    ranker = TFIDF(idx, field="title")
    results = ranker.rank("rock", top_k=5)
    # Todos os 5 com mesmo score → ordem segue doc_id interno (inserção).
    assert [doc_id for doc_id, _ in results] == [f"d{i}" for i in range(5)]


def test_tfidf_augmented_scheme_funciona() -> None:
    idx = _make_index()
    ranker = TFIDF(idx, field="title", tf_scheme="augmented")
    results = dict(ranker.rank("amor", top_k=10))
    assert set(results) == {"t1", "t2", "t4"}
    for score in results.values():
        assert 0.0 < score <= 1.0 + 1e-9
