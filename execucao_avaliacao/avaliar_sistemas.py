"""
avaliar_sistemas.py
Avaliação sistemática dos 3 métodos de busca — Buscador de Músicas Brasileiras
ICC222 / UFAM 2026/1

Executa as 6 fases:
  1. Verificação e sanidade da coleção
  2. Cômputo de métricas (agregadas + por query)
  3. Análise estratificada por intent
  4. Testes de significância estatística (Wilcoxon + Bonferroni)
  5. Análise qualitativa (error analysis)
  6. Geração de figuras

Saída:
  resultados/tabela_principal.csv
  resultados/tabela_por_intent.csv
  resultados/significancia.csv
  resultados/scores_por_query_pivot.csv
  resultados/analise_qualitativa.txt
  resultados/por_query/<sistema>_por_query.csv
  figuras/f1_barras_ndcg_intent.png
  figuras/f2_boxplot_ndcg.png
  figuras/f3_scatter_bm25_vs_dense.png
  figuras/f4_heatmap_resultados.png
  relatorio_avaliacao.txt
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── verificação de dependências ───────────────────────────────────────────────
def _check_deps():
    missing = []
    for pkg in ["ir_measures", "scipy", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERRO] Dependências ausentes: {', '.join(missing)}")
        print(f"       Instale com: pip install {' '.join(missing)}")
        sys.exit(1)

_check_deps()

import ir_measures as ir
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
COLECAO_DIR = Path(__file__).resolve().parents[1] / "colecao_referencia"
OUTPUT_DIR     = Path(__file__).parent
RESULTADOS_DIR = OUTPUT_DIR / "resultados"
FIGURAS_DIR    = OUTPUT_DIR / "figuras"
POR_QUERY_DIR  = RESULTADOS_DIR / "por_query"

for d in [RESULTADOS_DIR, FIGURAS_DIR, POR_QUERY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SISTEMAS = {
    "BM25"  : COLECAO_DIR / "runs" / "bm25.txt",
    "TF-IDF": COLECAO_DIR / "runs" / "tfidf.txt",
    "Dense" : COLECAO_DIR / "runs" / "dense.txt",
}

# Métricas: objetos ir_measures → nome de exibição
METRICAS = [ir.MRR, ir.MAP, ir.nDCG@10, ir.P@10, ir.Bpref]
DISPLAY  = {"RR": "MRR", "AP": "MAP"}   # renomeia chaves internas
COLS_AGG = ["MRR", "MAP", "nDCG@10", "P@10", "Bpref"]

CORES_SISTEMA = {"BM25": "#2196F3", "TF-IDF": "#FF9800", "Dense": "#9C27B0"}
CORES_INTENT  = {
    "lyric": "#E91E63", "track": "#9C27B0",
    "artist": "#2196F3", "genre": "#FF9800", "album": "#4CAF50",
}


def _dname(m) -> str:
    """Converte objeto/string de métrica para nome de exibição."""
    s = str(m)
    return DISPLAY.get(s, s)


def _log(msg: str):
    # substitui caracteres fora do ASCII para compatibilidade com cp1252
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {safe}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — VERIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def fase1_verificar(qrels_df: pd.DataFrame, topics_df: pd.DataFrame) -> bool:
    _log("FASE 1 — Verificação e sanidade da coleção")
    erros = []

    qids_qrels  = set(qrels_df["query_id"].unique())
    qids_topics = set(topics_df["qid"].unique())

    diff = qids_qrels.symmetric_difference(qids_topics)
    if diff:
        erros.append(f"  Query IDs divergem entre qrels e topics: {diff}")

    for nome, path in SISTEMAS.items():
        if not path.exists():
            erros.append(f"  Run file ausente: {path}")
            continue
        run_df = pd.read_csv(
            path, sep=r"\s+", header=None,
            names=["qid", "q0", "doc_id", "rank", "score", "sistema"]
        )
        diff_run = qids_topics - set(run_df["qid"].unique())
        if diff_run:
            erros.append(f"  {nome}: queries ausentes no run file: {diff_run}")
        docs_por_q = run_df.groupby("qid")["doc_id"].count()
        _log(f"  {nome}: {len(run_df)} linhas | docs/query: "
             f"min={docs_por_q.min()} max={docs_por_q.max()} "
             f"media={docs_por_q.mean():.1f}")

    sem_rel = [
        qid for qid in qids_qrels
        if qrels_df[qrels_df["query_id"] == qid]["relevance"].max() < 1
    ]
    if sem_rel:
        erros.append(f"  Queries sem documento relevante: {sem_rel}")

    # Distribuição dos graus
    dist = qrels_df["relevance"].value_counts().sort_index()
    _log(f"  Distribuição de graus: {dist.to_dict()}")
    _log(f"  Queries com relevante: "
         f"{(~qrels_df[qrels_df['relevance']>0].empty) and len(qids_qrels)}/{len(qids_topics)}")

    if erros:
        for e in erros:
            print(f"  [AVISO] {e}")
        return False
    _log("  OK — coleção válida e completa")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — CÔMPUTO DE MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
def fase2_calcular_metricas(qrels_ir):
    _log("FASE 2 — Calculando métricas (agregadas + por query)")

    agg_rows  = []
    query_dfs = {}

    for nome, path in SISTEMAS.items():
        _log(f"  → {nome}")
        run = ir.read_trec_run(str(path))

        # ── por query via iter_calc ───────────────────────────────────────────
        rows = [
            {"query_id": r.query_id, "measure": _dname(r.measure), "value": r.value}
            for r in ir.iter_calc(METRICAS, qrels_ir, run)
        ]
        df_q = pd.DataFrame(rows)
        df_q["sistema"] = nome
        query_dfs[nome] = df_q
        df_q.to_csv(
            POR_QUERY_DIR / f"{nome.lower().replace('-','_')}_por_query.csv",
            index=False
        )

        # ── agregado calculado a partir dos scores por query (evita re-consumir run) ──
        agg_means = df_q.groupby("measure")["value"].mean()
        row = {"Sistema": nome}
        row.update({m: round(float(v), 4) for m, v in agg_means.items()})
        agg_rows.append(row)

    df_agg = (
        pd.DataFrame(agg_rows)
        .set_index("Sistema")
        .reindex(columns=[c for c in COLS_AGG if c in pd.DataFrame(agg_rows).columns])
    )
    df_agg.to_csv(RESULTADOS_DIR / "tabela_principal.csv")
    _log("  → resultados/tabela_principal.csv")
    return df_agg, query_dfs


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — ANÁLISE POR INTENT
# ══════════════════════════════════════════════════════════════════════════════
def fase3_por_intent(qrels_ir, topics_df: pd.DataFrame, query_dfs: dict):
    _log("FASE 3 — Análise estratificada por intent")

    intents      = sorted(topics_df["intent"].unique())
    sistemas     = list(SISTEMAS.keys())
    metricas_vis = ["nDCG@10", "MAP", "MRR", "P@10"]
    rows = []

    for intent in intents:
        qids = topics_df[topics_df["intent"] == intent]["qid"].tolist()
        for nome in sistemas:
            df_q = query_dfs[nome]
            sub  = df_q[df_q["query_id"].isin(qids)]
            row  = {"Intent": intent, "Sistema": nome, "N": len(qids)}
            for m in metricas_vis:
                vals = sub[sub["measure"] == m]["value"]
                row[m] = round(vals.mean(), 4) if len(vals) > 0 else float("nan")
            rows.append(row)

    df_intent = pd.DataFrame(rows)
    df_intent.to_csv(RESULTADOS_DIR / "tabela_por_intent.csv", index=False)

    # pivot nDCG@10
    pivot = df_intent.pivot(index="Intent", columns="Sistema", values="nDCG@10")
    pivot.to_csv(RESULTADOS_DIR / "tabela_intent_ndcg_pivot.csv")
    _log("  → resultados/tabela_por_intent.csv")
    return df_intent


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — SIGNIFICÂNCIA ESTATÍSTICA
# ══════════════════════════════════════════════════════════════════════════════
def fase4_significancia(query_dfs: dict):
    _log("FASE 4 — Testes de significância (Wilcoxon signed-rank + Bonferroni)")

    sistemas     = list(SISTEMAS.keys())
    metricas_sig = ["nDCG@10", "MAP", "MRR", "P@10", "Bpref"]
    pares        = [
        (sistemas[i], sistemas[j])
        for i in range(len(sistemas))
        for j in range(i + 1, len(sistemas))
    ]
    n_testes  = len(pares) * len(metricas_sig)
    alpha_bon = 0.05 / n_testes

    rows = []
    for s1, s2 in pares:
        for m in metricas_sig:
            v1 = query_dfs[s1][query_dfs[s1]["measure"] == m]["value"].values
            v2 = query_dfs[s2][query_dfs[s2]["measure"] == m]["value"].values

            # alinha pelos mesmos query IDs
            qids1 = query_dfs[s1][query_dfs[s1]["measure"] == m]["query_id"].values
            qids2 = query_dfs[s2][query_dfs[s2]["measure"] == m]["query_id"].values
            common = sorted(set(qids1) & set(qids2))
            d1 = query_dfs[s1][(query_dfs[s1]["measure"] == m) &
                                (query_dfs[s1]["query_id"].isin(common))].sort_values("query_id")["value"].values
            d2 = query_dfs[s2][(query_dfs[s2]["measure"] == m) &
                                (query_dfs[s2]["query_id"].isin(common))].sort_values("query_id")["value"].values

            diffs = d1 - d2
            if np.all(diffs == 0):
                stat, p = float("nan"), 1.0
            else:
                try:
                    stat, p = stats.wilcoxon(d1, d2, zero_method="wilcox")
                except Exception:
                    stat, p = float("nan"), float("nan")

            rows.append({
                "Sistema 1"  : s1,
                "Sistema 2"  : s2,
                "Métrica"    : m,
                "Média S1"   : round(float(np.mean(d1)), 4),
                "Média S2"   : round(float(np.mean(d2)), 4),
                "Melhor"     : s1 if np.mean(d1) >= np.mean(d2) else s2,
                "W"          : round(float(stat), 2) if not np.isnan(stat) else "—",
                "p-valor"    : round(float(p), 5) if not np.isnan(p) else "—",
                "Sig α=0.05" : "Sim" if (not np.isnan(p) and p < 0.05) else "Não",
                "Sig Bonf."  : "Sim" if (not np.isnan(p) and p < alpha_bon) else "Não",
            })

    df_sig = pd.DataFrame(rows)
    df_sig.to_csv(RESULTADOS_DIR / "significancia.csv", index=False)
    _log(f"  n_testes={n_testes} | α_Bonferroni={alpha_bon:.5f}")
    _log("  → resultados/significancia.csv")
    return df_sig, alpha_bon


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5 — ANÁLISE QUALITATIVA
# ══════════════════════════════════════════════════════════════════════════════
def fase5_qualitativa(query_dfs: dict, topics_df: pd.DataFrame) -> pd.DataFrame:
    _log("FASE 5 — Análise qualitativa (error analysis)")

    sistemas   = list(SISTEMAS.keys())
    metrica    = "nDCG@10"
    intent_map = dict(zip(topics_df["qid"], topics_df["intent"]))
    query_map  = dict(zip(topics_df["qid"], topics_df["query"]))

    # ── pivot query × sistema ────────────────────────────────────────────────
    pivot_data = {}
    for nome in sistemas:
        sub = (
            query_dfs[nome][query_dfs[nome]["measure"] == metrica]
            [["query_id", "value"]]
            .set_index("query_id")["value"]
        )
        pivot_data[nome] = sub

    df_pivot = pd.DataFrame(pivot_data).reset_index()
    df_pivot.columns = ["qid"] + sistemas
    df_pivot["intent"]     = df_pivot["qid"].map(intent_map)
    df_pivot["query"]      = df_pivot["qid"].map(query_map)
    df_pivot["media"]      = df_pivot[sistemas].mean(axis=1)
    df_pivot["max_diff"]   = df_pivot[sistemas].max(axis=1) - df_pivot[sistemas].min(axis=1)
    df_pivot["melhor_sys"] = df_pivot[sistemas].idxmax(axis=1)
    df_pivot = df_pivot.sort_values("qid").reset_index(drop=True)
    df_pivot.to_csv(RESULTADOS_DIR / "scores_por_query_pivot.csv", index=False)

    # ── relatório qualitativo ────────────────────────────────────────────────
    linhas = []
    sep70  = "=" * 70
    sep50  = "-" * 65

    linhas += [
        "ANÁLISE QUALITATIVA — nDCG@10 por Query e Sistema",
        sep70,
        f"Data     : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Métrica  : nDCG@10 | Sistemas: {', '.join(sistemas)}",
        "",
    ]

    # 1. Queries difíceis (média < 0.10)
    dificeis = df_pivot[df_pivot["media"] < 0.10].sort_values("media")
    linhas.append(f"1. QUERIES DIFÍCEIS PARA TODOS OS SISTEMAS (média nDCG@10 < 0.10)")
    if dificeis.empty:
        linhas.append("   Nenhuma.")
    else:
        header = f"   {'QID':<7} {'Intent':<8}" + "".join(f"{s:>12}" for s in sistemas) + "  Query"
        linhas += [header, "   " + sep50]
        for _, r in dificeis.iterrows():
            vals = "".join(f"{r[s]:>12.4f}" for s in sistemas)
            linhas.append(f"   {r['qid']:<7} {r['intent']:<8}{vals}  {str(r['query'])[:40]}")
    linhas.append("")

    # 2. Maior divergência entre sistemas
    divergentes = df_pivot.nlargest(10, "max_diff")
    linhas.append("2. TOP-10 QUERIES COM MAIOR DIVERGÊNCIA ENTRE SISTEMAS")
    header2 = (f"   {'QID':<7} {'Intent':<8}"
                + "".join(f"{s:>12}" for s in sistemas)
                + f"  {'Diff':>6}  {'Melhor':<14}  Query")
    linhas += [header2, "   " + sep50]
    for _, r in divergentes.iterrows():
        vals = "".join(f"{r[s]:>12.4f}" for s in sistemas)
        linhas.append(
            f"   {r['qid']:<7} {r['intent']:<8}{vals}"
            f"  {r['max_diff']:>6.3f}  {r['melhor_sys']:<14}  {str(r['query'])[:35]}"
        )
    linhas.append("")

    # 3. Melhor sistema por intent
    linhas.append("3. SISTEMA VENCEDOR POR INTENT (nDCG@10 médio)")
    linhas.append("   " + sep50)
    for intent in sorted(df_pivot["intent"].unique()):
        sub    = df_pivot[df_pivot["intent"] == intent][sistemas]
        medias = sub.mean()
        melhor = medias.idxmax()
        linhas.append(f"   {intent.upper():<8}: {melhor} ({medias[melhor]:.4f})")
        for s in sistemas:
            marker = " <-- melhor" if s == melhor else ""
            linhas.append(f"            {s:<14}: {medias[s]:.4f}{marker}")
    linhas.append("")

    # 4. Scores completos por query
    linhas.append("4. SCORES COMPLETOS — nDCG@10 POR QUERY")
    header3 = (f"   {'QID':<7} {'Intent':<8}"
                + "".join(f"{s:>12}" for s in sistemas)
                + f"  {'Média':>7}  Query")
    linhas += [header3, "   " + sep50]
    for _, r in df_pivot.iterrows():
        vals = "".join(f"{r[s]:>12.4f}" for s in sistemas)
        linhas.append(
            f"   {r['qid']:<7} {r['intent']:<8}{vals}"
            f"  {r['media']:>7.4f}  {str(r['query'])[:35]}"
        )

    (RESULTADOS_DIR / "analise_qualitativa.txt").write_text(
        "\n".join(linhas), encoding="utf-8"
    )
    _log("  → resultados/analise_qualitativa.txt")
    _log("  → resultados/scores_por_query_pivot.csv")
    return df_pivot


# ══════════════════════════════════════════════════════════════════════════════
# FASE 6 — FIGURAS
# ══════════════════════════════════════════════════════════════════════════════
def fase6_figuras(df_agg: pd.DataFrame, df_intent: pd.DataFrame, df_pivot: pd.DataFrame):
    _log("FASE 6 — Gerando figuras")

    sistemas = list(SISTEMAS.keys())
    intents  = sorted(df_intent["Intent"].unique())
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size"  : 11,
        "axes.grid"  : True,
        "grid.alpha" : 0.3,
        "grid.linestyle": "--",
    })

    # ── F1: Barras nDCG@10 por intent e sistema ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x      = np.arange(len(intents))
    width  = 0.25
    offs   = [-width, 0, width]

    for i, nome in enumerate(sistemas):
        vals = []
        for intent in intents:
            sub = df_intent[
                (df_intent["Intent"] == intent) & (df_intent["Sistema"] == nome)
            ]["nDCG@10"]
            vals.append(float(sub.values[0]) if len(sub) > 0 else 0.0)

        bars = ax.bar(
            x + offs[i], vals, width,
            label=nome, color=CORES_SISTEMA[nome],
            edgecolor="white", linewidth=0.8
        )
        for bar, val in zip(bars, vals):
            if val >= 0.01:
                rot = 90 if val < 0.15 else 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, rotation=rot
                )

    ax.set_xlabel("Intenção de Busca (Intent)", fontsize=12)
    ax.set_ylabel("nDCG@10", fontsize=12)
    ax.set_title("nDCG@10 por Intenção de Busca e Método de Busca",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(intents, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(title="Sistema", fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "f1_barras_ndcg_intent.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  → figuras/f1_barras_ndcg_intent.png")

    # ── F2: Box plot nDCG@10 por sistema ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    data_box = [df_pivot[nome].dropna().values for nome in sistemas]

    bp = ax.boxplot(
        data_box, tick_labels=sistemas, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.6},
    )
    for patch, nome in zip(bp["boxes"], sistemas):
        patch.set_facecolor(CORES_SISTEMA[nome])
        patch.set_alpha(0.75)

    for i, (nome, box_data) in enumerate(zip(sistemas, data_box)):
        if len(box_data) > 0:
            ax.plot(i + 1, np.mean(box_data), marker="^", color="red",
                    markersize=9, zorder=5, label="Média" if i == 0 else "")
            ax.text(i + 1 + 0.15, np.mean(box_data),
                    f"{np.mean(box_data):.3f}", va="center", fontsize=9, color="red")

    ax.set_ylabel("nDCG@10 (score por query)", fontsize=12)
    ax.set_title("Distribuição de nDCG@10 por Query — Comparação entre Sistemas",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "f2_boxplot_ndcg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  → figuras/f2_boxplot_ndcg.png")

    # ── F3: Scatter BM25 vs Dense, colorido por intent ───────────────────────
    if "BM25" in df_pivot.columns and "Dense" in df_pivot.columns:
        fig, ax = plt.subplots(figsize=(7.5, 7))

        for intent in sorted(df_pivot["intent"].dropna().unique()):
            sub = df_pivot[df_pivot["intent"] == intent]
            ax.scatter(
                sub["BM25"], sub["Dense"],
                c=CORES_INTENT.get(intent, "#999"),
                label=intent, s=90, alpha=0.85,
                edgecolors="white", linewidths=0.7
            )

        lim = max(df_pivot["BM25"].max(), df_pivot["Dense"].max()) + 0.06
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, alpha=0.45, label="y = x (empate)")
        ax.set_xlim(-0.03, lim)
        ax.set_ylim(-0.03, lim)
        ax.set_xlabel("BM25 — nDCG@10", fontsize=12)
        ax.set_ylabel("Dense — nDCG@10", fontsize=12)
        ax.set_title(
            "BM25 vs Busca Densa — nDCG@10 por Query\n(acima da diagonal: Dense superior)",
            fontsize=13, fontweight="bold", pad=12
        )
        # regiões de dominância
        ax.text(lim * 0.55, lim * 0.04, "BM25 superior", fontsize=9, alpha=0.5, style="italic")
        ax.text(0.02, lim * 0.88, "Dense superior", fontsize=9, alpha=0.5, style="italic")
        ax.legend(title="Intent", fontsize=10, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGURAS_DIR / "f3_scatter_bm25_vs_dense.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log("  → figuras/f3_scatter_bm25_vs_dense.png")

    # ── F4: Heatmap de resultados agregados ───────────────────────────────────
    cols_heat = [c for c in COLS_AGG if c in df_agg.columns]
    df_heat   = df_agg[cols_heat].copy()

    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(df_heat.values.astype(float), cmap="YlGn",
                   aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols_heat)))
    ax.set_yticks(range(len(df_heat.index)))
    ax.set_xticklabels(cols_heat, fontsize=12)
    ax.set_yticklabels(df_heat.index, fontsize=12)
    ax.set_title("Resultados Agregados por Sistema e Métrica",
                 fontsize=13, fontweight="bold", pad=14)

    # anota valores e destaca melhores por coluna
    best_per_col = df_heat.max()
    for i in range(len(df_heat.index)):
        for j, col in enumerate(cols_heat):
            val = df_heat.values[i, j]
            is_best = abs(val - best_per_col[col]) < 1e-6
            color   = "white" if val > 0.55 else "black"
            weight  = "bold" if is_best else "normal"
            txt = f"{val:.3f}"
            if is_best:
                txt = f"[{txt}]"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, fontweight=weight, color=color)

    ax.set_xlabel("Métrica  ([valor] = melhor por coluna)", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Score", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "f4_heatmap_resultados.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  → figuras/f4_heatmap_resultados.png")


# ══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO FINAL
# ══════════════════════════════════════════════════════════════════════════════
def gerar_relatorio(df_agg, df_intent, df_sig, alpha_bon):
    _log("Gerando relatório final")

    sistemas = list(SISTEMAS.keys())
    intents  = sorted(df_intent["Intent"].unique())
    sep70    = "=" * 72
    sep50    = "-" * 72

    linhas = [
        "RELATÓRIO DE AVALIAÇÃO — BUSCADOR DE MÚSICAS BRASILEIRAS",
        sep70,
        f"Data        : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Coleção     : 50 queries | 1.604 julgamentos | 3 sistemas",
        f"Corpus      : 36.017 faixas (br_curated_lyrics.parquet)",
        "",
    ]

    # ── Tabela 1: resultados agregados ────────────────────────────────────────
    cols = list(df_agg.columns)
    best_per_col = df_agg.max()
    linhas += ["TABELA 1 — RESULTADOS AGREGADOS", sep50]
    header = f"  {'Sistema':<14}" + "".join(f"{c:>10}" for c in cols)
    linhas.append(header)
    linhas.append(sep50)
    for idx, row in df_agg.iterrows():
        line = f"  {idx:<14}"
        for col in cols:
            val = row[col]
            marker = "*" if abs(val - best_per_col[col]) < 1e-6 else " "
            line += f"{val:>9.4f}{marker}"
        linhas.append(line)
    linhas += ["  (* = melhor valor na coluna)", ""]

    # ── Tabela 2: nDCG@10 por intent ─────────────────────────────────────────
    linhas += ["TABELA 2 — nDCG@10 POR INTENT", sep50]
    h2 = f"  {'Intent':<10}" + "".join(f"{s:>14}" for s in sistemas)
    linhas += [h2, sep50]
    for intent in intents:
        row_str = f"  {intent:<10}"
        vals    = []
        for nome in sistemas:
            sub = df_intent[
                (df_intent["Intent"] == intent) & (df_intent["Sistema"] == nome)
            ]["nDCG@10"]
            v = float(sub.values[0]) if len(sub) > 0 else float("nan")
            vals.append(v)
            row_str += f"{v:>14.4f}"
        melhor = sistemas[int(np.argmax(vals))]
        row_str += f"  <- {melhor}"
        linhas.append(row_str)
    linhas.append("")

    # ── Tabela 3: significância ───────────────────────────────────────────────
    sig_05  = df_sig[df_sig["Sig α=0.05"] == "Sim"]
    sig_bon = df_sig[df_sig["Sig Bonf."] == "Sim"]
    linhas += [
        "TABELA 3 — SIGNIFICÂNCIA ESTATÍSTICA (Wilcoxon signed-rank)",
        sep50,
        f"  alpha nominal   = 0.05",
        f"  alpha Bonferroni = 0.05 / {len(df_sig)} testes = {alpha_bon:.5f}",
        f"  Sig. em alpha=0.05  : {len(sig_05):>3} de {len(df_sig)} comparações",
        f"  Sig. após Bonferroni: {len(sig_bon):>3} de {len(df_sig)} comparações",
        "",
        f"  {'Par':<30} {'Métrica':<10} {'Med S1':>8} {'Med S2':>8} {'p-valor':>9}  S05  SBon",
        sep50,
    ]
    for _, r in df_sig.iterrows():
        par = f"{r['Sistema 1']} vs {r['Sistema 2']}"
        m1  = r["Média S1"]
        m2  = r["Média S2"]
        p   = r["p-valor"]
        s05 = "†" if r["Sig α=0.05"] == "Sim" else " "
        sbn = "‡" if r["Sig Bonf."] == "Sim" else " "
        linhas.append(
            f"  {par:<30} {r['Métrica']:<10} {m1:>8.4f} {m2:>8.4f} {str(p):>9}  {s05:^3}  {sbn:^4}"
        )
    linhas += ["  (†: p<0.05  |  ‡: p<alpha Bonferroni)", ""]

    # ── Arquivos gerados ───────────────────────────────────────────────────────
    linhas += ["ARQUIVOS GERADOS", sep50]
    for f in sorted(RESULTADOS_DIR.rglob("*")):
        if f.is_file():
            linhas.append(f"  resultados/{f.relative_to(RESULTADOS_DIR)}")
    for f in sorted(FIGURAS_DIR.rglob("*")):
        if f.is_file():
            linhas.append(f"  figuras/{f.relative_to(FIGURAS_DIR)}")

    out = "\n".join(linhas)
    (OUTPUT_DIR / "relatorio_avaliacao.txt").write_text(out, encoding="utf-8")
    print("\n" + out)
    _log("→ relatorio_avaliacao.txt")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  AVALIAÇÃO SISTEMÁTICA — BUSCADOR DE MÚSICAS BRASILEIRAS")
    print("=" * 60)

    # verifica arquivos de entrada
    arquivos = [
        COLECAO_DIR / "qrels.tsv",
        COLECAO_DIR / "topics.tsv",
        *SISTEMAS.values(),
    ]
    for p in arquivos:
        if not p.exists():
            print(f"[ERRO] Arquivo não encontrado: {p}")
            sys.exit(1)

    # qrels.tsv: sem cabeçalho (formato TREC puro)
    qrels_df  = pd.read_csv(
        COLECAO_DIR / "qrels.tsv", sep="\t", header=None,
        names=["query_id", "iter", "doc_id", "relevance"]
    )
    # topics.tsv: tem cabeçalho "query_id\tquery_text\tdescription\tintent"
    topics_df = pd.read_csv(
        COLECAO_DIR / "topics.tsv", sep="\t", header=0
    )
    topics_df.columns = ["qid", "query", "desc", "intent"]
    # read_trec_qrels retorna um generator — converte para lista para reusar nos 3 sistemas
    qrels_ir  = list(ir.read_trec_qrels(str(COLECAO_DIR / "qrels.tsv")))

    fase1_verificar(qrels_df, topics_df)
    df_agg, query_dfs      = fase2_calcular_metricas(qrels_ir)
    df_intent              = fase3_por_intent(qrels_ir, topics_df, query_dfs)
    df_sig, alpha_bon      = fase4_significancia(query_dfs)
    df_pivot               = fase5_qualitativa(query_dfs, topics_df)
    fase6_figuras(df_agg, df_intent, df_pivot)
    gerar_relatorio(df_agg, df_intent, df_sig, alpha_bon)

    print(f"\n[CONCLUÍDO] Todos os resultados em: {OUTPUT_DIR}")
