"""
notebooks/05_stemming_comparativo.py
=====================================
Demonstra a aplicação do PorterStemmer em metadados de músicas e compara
o impacto da indexação BM25 com e sem stemming.

Seções:
    1. Como o PorterStemmer transforma tokens musicais
    2. Análise de redução do vocabulário
    3. Comparativo de recuperação: com vs. sem stemming
    4. Casos onde stemming ajuda e onde atrapalha
    5. Exporta relatório para o grupo

Dependências:
    pip install rank-bm25 pandas nltk matplotlib seaborn tabulate
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab",  quiet=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

CORPUS_PATH = ROOT / "data" / "songs_clean.csv"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# Recursos compartilhados
# ---------------------------------------------------------------------------

_stemmer = PorterStemmer()

STOPWORDS = set(stopwords.words("english")) | {
    "feat", "ft", "featuring", "remix", "remaster", "remastered",
    "version", "edit", "extended", "radio", "live", "acoustic",
    "instrumental", "deluxe", "edition", "bonus", "anniversary",
    "explicit", "official", "vol", "pt", "part",
}


# =============================================================================
# 1. PIPELINE DE PRÉ-PROCESSAMENTO (com e sem stemming)
# =============================================================================

def tokenizar_base(texto: str) -> list[str]:
    """Tokenização + remoção de stopwords (etapa comum aos dois modos)."""
    if not isinstance(texto, str) or not texto.strip():
        return []
    tokens = word_tokenize(texto.lower())
    return [t for t in tokens if t.isalpha() and len(t) > 1 and t not in STOPWORDS]


def pipeline_sem_stemming(texto: str) -> list[str]:
    return tokenizar_base(texto)


def pipeline_com_stemming(texto: str) -> list[str]:
    return [_stemmer.stem(t) for t in tokenizar_base(texto)]


# =============================================================================
# 2. ANÁLISE DO STEMMER — TOKEN A TOKEN
# =============================================================================

def secao_1_exemplos_stemmer() -> None:
    print("\n" + "=" * 65)
    print("  1. COMO O PORTERSTEMMER TRANSFORMA TOKENS MUSICAIS")
    print("=" * 65)

    # Grupos temáticos de termos relevantes para o domínio musical
    grupos = {
        "Gêneros e estilos": [
            "dancing", "danced", "dancer", "dances",
            "rocking", "rocked", "rocker",
            "singing", "singer", "singers",
            "classical", "classically",
        ],
        "Emoções e mood": [
            "loving", "loved", "lover", "lovingly",
            "sadness", "sadly", "sadder",
            "happiness", "happily", "happier",
            "energetic", "energetically", "energy",
        ],
        "Instrumentos e produção": [
            "guitars", "guitarist", "guitarists",
            "drumming", "drummer", "drums",
            "producing", "produced", "producer",
            "recording", "recorded", "recordings",
        ],
        "Casos problemáticos": [
            # Termos que o PorterStemmer trata de forma não-intuitiva
            "universe", "university",   # mesmo stem → falso positivo
            "general", "generally",     # stem correto
            "wand", "wandering",        # stem diferente do esperado
            "caring", "car",            # colisão de stem
        ],
    }

    for grupo, termos in grupos.items():
        print(f"\n  [{grupo}]")
        print(f"  {'Token original':<20} {'Stem':<15} {'Mudou?'}")
        print(f"  {'-'*20} {'-'*15} {'-'*6}")
        for termo in termos:
            stem = _stemmer.stem(termo)
            mudou = "✗ sim" if stem != termo else "  não"
            print(f"  {termo:<20} {stem:<15} {mudou}")


# =============================================================================
# 3. ANÁLISE DE REDUÇÃO DE VOCABULÁRIO
# =============================================================================

def secao_2_vocabulario(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 65)
    print("  2. IMPACTO NO VOCABULÁRIO DO CORPUS")
    print("=" * 65)

    col = "text_field" if "text_field" in df.columns else "track_name"
    textos = df[col].fillna("").astype(str).tolist()

    print(f"\n  Processando {len(textos):,} documentos... ", end="", flush=True)

    vocab_sem  = Counter()
    vocab_com  = Counter()
    stem_map   = defaultdict(set)   # stem → conjunto de formas originais

    t0 = time.perf_counter()
    for texto in textos:
        tokens_orig = tokenizar_base(texto)
        tokens_stem = [_stemmer.stem(t) for t in tokens_orig]

        vocab_sem.update(tokens_orig)
        vocab_com.update(tokens_stem)

        for orig, stem in zip(tokens_orig, tokens_stem):
            stem_map[stem].add(orig)

    elapsed = time.perf_counter() - t0
    print(f"concluído em {elapsed:.1f}s")

    # Métricas
    tamanho_sem = len(vocab_sem)
    tamanho_com = len(vocab_com)
    reducao_abs = tamanho_sem - tamanho_com
    reducao_pct = reducao_abs / tamanho_sem * 100

    print(f"\n  {'Vocabulário sem stemming':<35}: {tamanho_sem:>8,} termos únicos")
    print(f"  {'Vocabulário com stemming':<35}: {tamanho_com:>8,} termos únicos")
    print(f"  {'Redução absoluta':<35}: {reducao_abs:>8,} termos  ({reducao_pct:.1f}%)")

    # Stems com maior cobertura (agrupam mais formas)
    top_stems = sorted(stem_map.items(), key=lambda x: len(x[1]), reverse=True)[:12]
    print(f"\n  Top stems por cobertura (mais formas agrupadas):")
    print(f"  {'Stem':<15} {'Formas agrupadas'}")
    print(f"  {'-'*15} {'-'*50}")
    for stem, formas in top_stems:
        formas_str = ", ".join(sorted(formas)[:6])
        extra = f" (+{len(formas)-6} mais)" if len(formas) > 6 else ""
        print(f"  {stem:<15} {formas_str}{extra}")

    return {
        "vocab_sem":   vocab_sem,
        "vocab_com":   vocab_com,
        "stem_map":    stem_map,
        "reducao_pct": reducao_pct,
    }


def plotar_vocabulario(vocab_info: dict) -> None:
    vocab_sem = vocab_info["vocab_sem"]
    vocab_com = vocab_info["vocab_com"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 3a. Frequência dos 20 tokens mais comuns — sem stemming
    top20_sem = vocab_sem.most_common(20)
    termos, freqs = zip(*top20_sem)
    axes[0].barh(termos[::-1], freqs[::-1],
                 color=sns.color_palette("muted")[0])
    axes[0].set_title("Top 20 tokens\n(sem stemming)")
    axes[0].set_xlabel("Frequência no corpus")

    # 3b. Frequência dos 20 stems mais comuns
    top20_com = vocab_com.most_common(20)
    stems, freqs2 = zip(*top20_com)
    axes[1].barh(stems[::-1], freqs2[::-1],
                 color=sns.color_palette("muted")[2])
    axes[1].set_title("Top 20 stems\n(com stemming)")
    axes[1].set_xlabel("Frequência no corpus")

    # 3c. Distribuição de cobertura dos stems
    coberturas = [len(v) for v in vocab_info["stem_map"].values()]
    bins = [1, 2, 3, 4, 5, 10, 20, max(coberturas) + 1]
    labels = ["1", "2", "3", "4", "5–9", "10–19", f"20+"]
    contagens = pd.cut(coberturas, bins=bins, labels=labels,
                       right=False).value_counts().sort_index()
    axes[2].bar(labels, contagens.values,
                color=sns.color_palette("muted")[4], edgecolor="white")
    axes[2].set_title("Stems por nº de formas agrupadas")
    axes[2].set_xlabel("Formas originais por stem")
    axes[2].set_ylabel("Número de stems")

    plt.suptitle("Análise de vocabulário — com vs. sem stemming", y=1.01)
    plt.tight_layout()
    plt.savefig(ROOT / "data" / "stemming_vocabulario.png",
                bbox_inches="tight", dpi=130)
    plt.show()
    print("  Gráfico salvo: data/stemming_vocabulario.png")


# =============================================================================
# 4. COMPARATIVO DE RECUPERAÇÃO BM25
# =============================================================================

def construir_indices(df: pd.DataFrame) -> tuple[BM25Okapi, BM25Okapi, list, list]:
    """Constrói dois índices BM25: um sem e um com stemming."""
    col = "text_field" if "text_field" in df.columns else "track_name"
    textos = df[col].fillna("").astype(str).tolist()

    print("\n  Construindo índices BM25... ", end="", flush=True)
    t0 = time.perf_counter()

    corpus_sem  = [pipeline_sem_stemming(t) for t in textos]
    corpus_com  = [pipeline_com_stemming(t) for t in textos]

    bm25_sem = BM25Okapi(corpus_sem, k1=1.5, b=0.75)
    bm25_com = BM25Okapi(corpus_com, k1=1.5, b=0.75)

    print(f"concluído em {time.perf_counter() - t0:.1f}s")
    print(f"  Vocab sem stemming: {len(bm25_sem.idf):,} | "
          f"com stemming: {len(bm25_com.idf):,}")

    return bm25_sem, bm25_com, corpus_sem, corpus_com


def buscar(bm25: BM25Okapi, df: pd.DataFrame,
           query_tokens: list[str], k: int = 10) -> pd.DataFrame:
    """Executa busca e retorna DataFrame com top-K resultados."""
    scores  = bm25.get_scores(query_tokens)
    top_idx = np.argsort(scores)[::-1][:k]

    rows = []
    for rank, idx in enumerate(top_idx, 1):
        row = df.iloc[idx]
        rows.append({
            "rank":        rank,
            "score":       round(float(scores[idx]), 4),
            "track_name":  str(row.get("track_name", "")),
            "artist_name": str(row.get("artist_name", "")),
            "album_name":  str(row.get("album_name", "")),
            "popularity":  row.get("popularity", 0),
        })
    return pd.DataFrame(rows)


def secao_3_comparativo(
    df: pd.DataFrame,
    bm25_sem: BM25Okapi,
    bm25_com: BM25Okapi,
) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  3. COMPARATIVO DE RECUPERAÇÃO — COM vs. SEM STEMMING")
    print("=" * 65)

    # Queries organizadas por categoria
    queries = [
        # (descrição, query_sem, query_com)
        ("Forma flexionada — verbo",
         "dancing",        "danc"),
        ("Forma flexionada — plural",
         "guitars",        "guitar"),
        ("Forma flexionada — -er",
         "singer",         "singer"),
        ("Termo base coincide",
         "love",           "love"),
        ("Múltiplos termos",
         "rock guitar band", "rock guitar band"),
        ("Nome próprio (artista)",
         "beatles",        "beatl"),
        ("Termo raro",
         "psychedelic",    "psychedel"),
    ]

    registros = []

    for descricao, q_sem, q_stem in queries:
        tokens_sem = pipeline_sem_stemming(q_sem)
        tokens_com = pipeline_com_stemming(q_stem)

        res_sem = buscar(bm25_sem, df, tokens_sem, k=5)
        res_com = buscar(bm25_com, df, tokens_com, k=5)

        top_sem = res_sem.iloc[0] if len(res_sem) else None
        top_com = res_com.iloc[0] if len(res_com) else None

        print(f"\n  [{descricao}]")
        print(f"  Query (sem): {tokens_sem}  →  Query (com): {tokens_com}")
        print(f"  {'Rank':<5} {'SEM STEMMING':<40} {'COM STEMMING'}")
        print(f"  {'-'*5} {'-'*40} {'-'*40}")

        for i in range(5):
            r_s = res_sem.iloc[i] if i < len(res_sem) else None
            r_c = res_com.iloc[i] if i < len(res_com) else None

            nome_s = f"{r_s['track_name'][:22]} ({r_s['score']:.3f})" if r_s is not None else "—"
            nome_c = f"{r_c['track_name'][:22]} ({r_c['score']:.3f})" if r_c is not None else "—"
            print(f"  {i+1:<5} {nome_s:<40} {nome_c}")

        # Score máximo de cada modalidade para a query
        score_sem = float(res_sem["score"].max()) if len(res_sem) else 0.0
        score_com = float(res_com["score"].max()) if len(res_com) else 0.0
        vencedor  = "stemming" if score_com > score_sem else ("empatado" if score_com == score_sem else "sem stemming")

        registros.append({
            "query_original":  q_sem,
            "tokens_sem":      str(tokens_sem),
            "tokens_com":      str(tokens_com),
            "top1_sem":        top_sem["track_name"] if top_sem is not None else "—",
            "score_sem":       score_sem,
            "top1_com":        top_com["track_name"] if top_com is not None else "—",
            "score_com":       score_com,
            "vencedor":        vencedor,
        })

    return pd.DataFrame(registros)


# =============================================================================
# 5. CASOS ONDE O STEMMING AJUDA E ONDE ATRAPALHA
# =============================================================================

def secao_4_casos_extremos(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  4. CASOS ONDE STEMMING AJUDA vs. ATRAPALHA")
    print("=" * 65)

    casos = {
        "AJUDA — recuperação por variação morfológica": [
            ("dance",    "danc"),   # dance / dancing / dancer / danced → danc
            ("rock",     "rock"),   # rock / rocking / rocked → rock
            ("love",     "love"),   # love / loving / loved / lover → love
            ("produce",  "produc"), # produce / producer / produced → produc
        ],
        "ATRAPALHA — colisões semânticas (falsos positivos)": [
            ("universe", "univers"),  # universe → univers (≠ university → univers)
            ("wand",     "wand"),     # wand → wand (mas wander → wander, ok)
            ("caring",   "care"),     # caring → care (mas car → car, colisão)
            ("general",  "general"),  # fine aqui, mas gênero musical "general" perde contexto
        ],
        "ATRAPALHA — nomes próprios distorcidos": [
            ("beatles",  "beatl"),
            ("eminem",   "eminem"),
            ("adele",    "adel"),
            ("rihanna",  "rihanna"),
        ],
    }

    for categoria, pares in casos.items():
        print(f"\n  {categoria}")
        print(f"  {'Token':<18} {'Stem':<18} {'Formas que seriam agrupadas no corpus'}")
        print(f"  {'-'*18} {'-'*18} {'-'*36}")

        for token, stem_esperado in pares:
            stem_real = _stemmer.stem(token)
            # Busca no corpus tokens que têm o mesmo stem
            tokens_corpus = [t for t in df.get("text_field", pd.Series(dtype=str))
                             .fillna("").str.lower().str.split().explode()
                             .unique() if isinstance(t, str)
                             and _stemmer.stem(t) == stem_real][:5]
            print(f"  {token:<18} {stem_real:<18} {', '.join(tokens_corpus) or '(não encontrado)'}")

    print("""
  CONCLUSÃO PARA O RELATÓRIO:
  ─────────────────────────────────────────────────────────────────
  ✓ Stemming beneficia buscas por termos de ação/descrição musical
    (dance, rock, love, produce) agrupando formas flexionadas.

  ✗ Stemming prejudica nomes próprios (artistas, álbuns) e pode
    causar colisões semânticas entre termos não relacionados.

  → DECISÃO DO PROJETO: usar stemming apenas no campo text_field
    (que concatena track_name + artist_name + album_name após
    limpeza). Nomes próprios já foram parcialmente normalizados
    pelo pipeline de limpeza, reduzindo o impacto negativo.
  ─────────────────────────────────────────────────────────────────
""")


# =============================================================================
# 6. GRÁFICO COMPARATIVO DE SCORES
# =============================================================================

def plotar_scores_comparativo(
    df: pd.DataFrame,
    bm25_sem: BM25Okapi,
    bm25_com: BM25Okapi,
) -> None:
    queries_plot = [
        ("dancing",     "danc"),
        ("guitars",     "guitar"),
        ("loving",      "love"),
        ("rocking",     "rock"),
        ("singer",      "singer"),
    ]

    n = len(queries_plot)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)

    for ax, (q_sem, q_stem) in zip(axes, queries_plot):
        tokens_sem = pipeline_sem_stemming(q_sem)
        tokens_com = pipeline_com_stemming(q_stem)

        scores_sem = bm25_sem.get_scores(tokens_sem)
        scores_com = bm25_com.get_scores(tokens_com)

        top5_sem = np.sort(scores_sem)[::-1][:5]
        top5_com = np.sort(scores_com)[::-1][:5]

        x = np.arange(5)
        w = 0.35
        ax.bar(x - w/2, top5_sem, w, label="Sem stemming",
               color=sns.color_palette("muted")[0], alpha=0.85)
        ax.bar(x + w/2, top5_com, w, label="Com stemming",
               color=sns.color_palette("muted")[2], alpha=0.85)
        ax.set_title(f'"{q_sem}"', fontsize=11)
        ax.set_xlabel("Top-5 rank")
        ax.set_ylabel("Score BM25")
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{i+1}" for i in range(5)])
        if ax == axes[0]:
            ax.legend(fontsize=9)

    plt.suptitle("Score BM25 top-5 — com vs. sem stemming por query", y=1.02)
    plt.tight_layout()
    plt.savefig(ROOT / "data" / "stemming_scores_comparativo.png",
                bbox_inches="tight", dpi=130)
    plt.show()
    print("  Gráfico salvo: data/stemming_scores_comparativo.png")


# =============================================================================
# 7. EXPORTAR RELATÓRIO
# =============================================================================

def exportar_relatorio(df_comparativo: pd.DataFrame, vocab_info: dict) -> None:
    print("\n" + "=" * 65)
    print("  5. EXPORTAÇÃO DO RELATÓRIO")
    print("=" * 65)

    # CSV com comparativo por query
    path_csv = ROOT / "data" / "stemming_comparativo.csv"
    df_comparativo.to_csv(path_csv, index=False)
    print(f"  Tabela comparativa  → {path_csv}")

    # Resumo em texto
    path_txt = ROOT / "data" / "stemming_relatorio.txt"
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE STEMMING — PARTE 1\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Redução de vocabulário: {vocab_info['reducao_pct']:.1f}%\n")
        f.write(f"Vocab sem stemming    : {len(vocab_info['vocab_sem']):,}\n")
        f.write(f"Vocab com stemming    : {len(vocab_info['vocab_com']):,}\n\n")
        f.write("Comparativo por query:\n")
        f.write(df_comparativo.to_string(index=False))
    print(f"  Relatório em texto  → {path_txt}")

    # Resumo no terminal
    print("\n  RESUMO FINAL:")
    print(f"  {'Query':<22} {'Score sem':>10} {'Score com':>10} {'Vencedor'}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12}")
    for _, row in df_comparativo.iterrows():
        print(
            f"  {row['query_original']:<22}"
            f" {row['score_sem']:>10.4f}"
            f" {row['score_com']:>10.4f}"
            f"  {row['vencedor']}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 65)
    print("  STEMMING EM METADADOS MUSICAIS — ANÁLISE COMPARATIVA")
    print("=" * 65)

    # Seção 1 — exemplos do stemmer (não precisa do dataset)
    secao_1_exemplos_stemmer()

    # Carregar dataset
    if not CORPUS_PATH.exists():
        print(f"\n  [aviso] Dataset não encontrado em {CORPUS_PATH}")
        print("  Execute 02_limpeza_dataset.py primeiro.")
        print("  Usando amostra sintética para demonstração...\n")
        df = pd.DataFrame({
            "track_name":  ["Dancing Queen", "Rocking in the Free World",
                            "Lover", "Singer", "Guitar Man",
                            "I Loved You", "The Dancer", "Rock Band",
                            "Love Story", "Dance Monkey"],
            "artist_name": ["ABBA", "Neil Young", "Taylor Swift", "Travis",
                            "Elvis Presley", "Adele", "Dua Lipa", "Fall Out Boy",
                            "Taylor Swift", "Tones and I"],
            "album_name":  ["Arrival", "Ragged Glory", "Lover", "The Man Who",
                            "Elvis", "21", "Future Nostalgia", "Folie à Deux",
                            "Fearless", "Eyes Shut"],
            "popularity":  [88, 72, 85, 65, 78, 90, 82, 70, 87, 91],
        })
        df["text_field"] = (
            df["track_name"] + " " + df["artist_name"] + " " + df["album_name"]
        )
    else:
        print(f"\n  Carregando dataset: {CORPUS_PATH}")
        df = pd.read_csv(CORPUS_PATH, low_memory=False)
        print(f"  {len(df):,} faixas carregadas.")

    # Seção 2 — vocabulário
    vocab_info = secao_2_vocabulario(df)
    plotar_vocabulario(vocab_info)

    # Construir índices
    bm25_sem, bm25_com, _, _ = construir_indices(df)

    # Seção 3 — comparativo de recuperação
    df_comparativo = secao_3_comparativo(df, bm25_sem, bm25_com)

    # Seção 4 — casos extremos
    secao_4_casos_extremos(df)

    # Gráfico de scores
    plotar_scores_comparativo(df, bm25_sem, bm25_com)

    # Seção 5 — exportar
    exportar_relatorio(df_comparativo, vocab_info)

    print("\n  Concluído.\n")


if __name__ == "__main__":
    main()