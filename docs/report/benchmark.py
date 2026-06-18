"""Benchmark reproduzível para a seção de Resultados do relatório técnico.

Mede, sobre o corpus curado `br_curated_lyrics.parquet`:

1. Tempo de construção do índice invertido multi-campo (cold build) e o
   tamanho do artefato persistido em disco.
2. Estatísticas do índice (nº de documentos, vocabulário por campo).
3. Latência de consulta (média, mediana, p95) para BM25 e TF-IDF, sobre um
   conjunto fixo de consultas representativas.
4. Comparação qualitativa BM25 x TF-IDF (top-3) em algumas consultas.

Não reimplementa scoring: consome `SparseSearchEngine` e o core de RI. A saída
vai para o console e para `docs/report/results.md` (rastreabilidade dos números
citados no `main.tex`).

Uso:
    uv run python docs/report/benchmark.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from music_search.core.indexer import IndexBuilder
from music_search.data.datasets import (
    CURATED_FIELDS,
    DEFAULT_CURATED_CORPUS_PATH,
    BrazilianLyricsLoader,
)
from music_search.motors.search import (
    DEFAULT_INDEX_PATH,
    SparseSearchEngine,
    load_or_build_default_engine,
)

# Conjunto fixo de consultas representativas dos três regimes de busca
# (letra, artista/banda, gênero). Mantido pequeno e estável para que o
# benchmark seja determinístico e citável.
QUERIES: list[tuple[str, str]] = [
    ("amor saudade", "letra"),
    ("coração partido", "letra"),
    ("saudade do meu amor", "letra"),
    ("dançar a noite toda", "letra"),
    ("liberdade pra dentro da cabeça", "letra"),
    ("chega de saudade", "letra"),
    ("anitta", "artista"),
    ("caetano veloso", "artista"),
    ("legião urbana", "artista"),
    ("chico buarque", "artista"),
    ("racionais", "artista"),
    ("samba", "gênero"),
    ("bossa nova", "gênero"),
    ("forró pé de serra", "gênero"),
    ("rock nacional", "gênero"),
    ("mpb", "gênero"),
    ("funk carioca", "gênero"),
    ("sertanejo universitário", "gênero"),
]

# Consultas usadas na comparação qualitativa lado a lado.
QUALITATIVE: list[str] = [
    "amor saudade",
    "chega de saudade",
    "bossa nova",
]

REPEATS = 5  # repetições por consulta para estabilizar a latência
WARMUP = 3  # consultas de aquecimento (descartadas)


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}"


def measure_index_build(corpus_path: Path) -> tuple[float, int, int]:
    """Mede o tempo de cold build do índice e o nº de documentos."""
    loader = BrazilianLyricsLoader(corpus_path=corpus_path)
    documents = list(loader.iter_docs())
    builder = IndexBuilder(fields=tuple(CURATED_FIELDS))
    start = time.perf_counter()
    builder.extend(documents)
    index = builder.build()
    elapsed = time.perf_counter() - start
    return elapsed, index.num_docs, len(documents)


def vocab_stats(engine: SparseSearchEngine) -> dict[str, int]:
    return {field: len(list(engine.index.vocabulary(field))) for field in engine.index.fields}


def measure_latency(engine: SparseSearchEngine) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for algorithm in ("bm25", "tfidf"):
        # warmup
        for query, _ in QUERIES[:WARMUP]:
            engine.search(query, algorithm=algorithm, top_k=10)
        samples: list[float] = []
        for query, _ in QUERIES:
            for _ in range(REPEATS):
                start = time.perf_counter()
                engine.search(query, algorithm=algorithm, top_k=10)
                samples.append(time.perf_counter() - start)
        samples.sort()
        p95 = samples[int(0.95 * (len(samples) - 1))]
        out[algorithm] = {
            "mean": statistics.fmean(samples),
            "median": statistics.median(samples),
            "p95": p95,
            "min": samples[0],
            "max": samples[-1],
            "n": float(len(samples)),
        }
    return out


def qualitative_top3(engine: SparseSearchEngine) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for query in QUALITATIVE:
        out[query] = {}
        for algorithm in ("bm25", "tfidf"):
            hits = engine.search(query, algorithm=algorithm, top_k=3)
            out[query][algorithm] = [
                f"{h.track_name} — {h.artist_names or h.primary_artist_name}" for h in hits
            ]
    return out


def render_markdown(
    *,
    corpus_path: Path,
    build_time: float,
    num_docs: int,
    index_bytes: int,
    vocab: dict[str, int],
    latency: dict[str, dict[str, float]],
    qualitative: dict[str, dict[str, list[str]]],
) -> str:
    lines: list[str] = []
    lines.append("# Resultados do benchmark (gerado por `benchmark.py`)\n")
    lines.append(f"- Corpus: `{corpus_path}`")
    lines.append(f"- Documentos indexados: **{num_docs}**")
    lines.append(f"- Tempo de construção do índice (cold build): **{build_time:.2f} s**")
    lines.append(
        f"- Tamanho do índice persistido (`{DEFAULT_INDEX_PATH}`): "
        f"**{index_bytes / 1_000_000:.1f} MB** ({index_bytes} bytes)"
    )
    lines.append(f"- Repetições por consulta: {REPEATS} | consultas: {len(QUERIES)}\n")

    lines.append("## Vocabulário por campo (termos distintos)\n")
    lines.append("| Campo | Termos distintos |")
    lines.append("| --- | ---: |")
    for field, n in vocab.items():
        lines.append(f"| `{field}` | {n} |")
    lines.append("")

    lines.append("## Latência de consulta (ms)\n")
    lines.append("| Algoritmo | Média | Mediana | p95 | Mín | Máx |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for algo, m in latency.items():
        lines.append(
            f"| {algo.upper()} | {_fmt_ms(m['mean'])} | {_fmt_ms(m['median'])} | "
            f"{_fmt_ms(m['p95'])} | {_fmt_ms(m['min'])} | {_fmt_ms(m['max'])} |"
        )
    lines.append("")

    lines.append("## Comparação qualitativa BM25 × TF-IDF (top-3)\n")  # noqa: RUF001
    for query, by_algo in qualitative.items():
        lines.append(f"### Consulta: `{query}`\n")
        lines.append("| # | BM25 | TF-IDF |")
        lines.append("| --- | --- | --- |")
        bm25 = by_algo["bm25"]
        tfidf = by_algo["tfidf"]
        for i in range(3):
            b = bm25[i] if i < len(bm25) else "—"
            t = tfidf[i] if i < len(tfidf) else "—"
            lines.append(f"| {i + 1} | {b} | {t} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    corpus_path = DEFAULT_CURATED_CORPUS_PATH

    print("==> Medindo cold build do índice...")
    build_time, num_docs, _ = measure_index_build(corpus_path)
    print(f"    build: {build_time:.2f} s | docs: {num_docs}")

    print("==> Carregando engine (reconstrói o índice persistido)...")
    engine = load_or_build_default_engine(corpus_path=corpus_path, rebuild_index=True)

    index_bytes = DEFAULT_INDEX_PATH.stat().st_size if DEFAULT_INDEX_PATH.exists() else 0
    print(f"    índice persistido: {index_bytes / 1_000_000:.1f} MB")

    print("==> Vocabulário por campo...")
    vocab = vocab_stats(engine)
    for field, n in vocab.items():
        print(f"    {field}: {n}")

    print("==> Medindo latência (BM25 e TF-IDF)...")
    latency = measure_latency(engine)
    for algo, m in latency.items():
        print(
            f"    {algo.upper()}: média={_fmt_ms(m['mean'])} ms | "
            f"mediana={_fmt_ms(m['median'])} ms | p95={_fmt_ms(m['p95'])} ms"
        )

    print("==> Comparação qualitativa...")
    qualitative = qualitative_top3(engine)

    md = render_markdown(
        corpus_path=corpus_path,
        build_time=build_time,
        num_docs=num_docs,
        index_bytes=index_bytes,
        vocab=vocab,
        latency=latency,
        qualitative=qualitative,
    )
    out_path = Path(__file__).parent / "results.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\n==> Resultados salvos em {out_path}")


if __name__ == "__main__":
    main()
