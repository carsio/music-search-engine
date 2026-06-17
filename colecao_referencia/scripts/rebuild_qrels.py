"""Re-gera apenas o qrels.tsv a partir dos run files existentes, sem re-executar o pooling."""
import sys, csv, unicodedata, re
from pathlib import Path
from collections import defaultdict

REPO_ROOT  = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
OUTPUT_DIR = REPO_ROOT / "colecao_referencia"

# ── importa helpers do script principal ───────────────────────────────────────
spec = __import__("importlib.util").util.spec_from_file_location(
    "btc", Path(__file__).parent / "build_test_collection.py"
)
mod = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(mod)

TOPICS    = mod.TOPICS
normalize = mod.normalize
assign_grade = mod.assign_grade

# ── lê os run files e reconstrói pool ─────────────────────────────────────────
import duckdb
path = str(REPO_ROOT / "data/derived/final/br_curated_lyrics.parquet")
con  = duckdb.connect()

def doc_meta(doc_id: str) -> dict:
    row = con.execute(f"""
        SELECT id, track_name, primary_artist_name, artist_names,
               artist_genres, macro_genre, album_name
        FROM '{path}' WHERE id=?
    """, [doc_id]).fetchone()
    if not row:
        return {}
    return {
        "id": row[0], "track_name": row[1], "primary_artist_name": row[2],
        "artist_names": row[3], "artist_genres": row[4],
        "macro_genre": row[5], "album_name": row[6],
    }

pool: dict[str, set[str]] = defaultdict(set)
for run_file in (OUTPUT_DIR / "runs").glob("*.txt"):
    with run_file.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pool[parts[0]].add(parts[2])

# ── augmentação: para known_doc_id e known_artist, adiciona ao pool ───────────
# Garante que topics conhecidos (known-item) sempre tenham pelo menos o doc
# alvo julgado, mesmo que o motor não o tenha encontrado no top-20.
for topic in TOPICS:
    qid, _, _, intent, known_doc_id, known_artist, _, _ = topic
    if known_doc_id:
        pool[qid].add(known_doc_id)
    if known_artist and intent in ("artist", "track"):
        rows = con.execute(f"""
            SELECT id FROM '{path}'
            WHERE primary_artist_name = ?
            ORDER BY track_popularity DESC LIMIT 10
        """, [known_artist]).fetchall()
        for (doc_id,) in rows:
            pool[qid].add(doc_id)

print(f"Pool: {sum(len(v) for v in pool.values())} pares de (query, doc)")

# ── re-escreve qrels ───────────────────────────────────────────────────────────
topic_map = {t[0]: t for t in TOPICS}
stats = {"total": 0, "grade2": 0, "grade1": 0, "grade0": 0, "queries_com_relevante": 0}

with (OUTPUT_DIR / "qrels.tsv").open("w", encoding="utf-8") as f:
    for qid in sorted(pool):
        topic = topic_map.get(qid)
        if not topic:
            continue
        _, _, _, intent, known_doc_id, known_artist, known_genre, known_album = topic
        has_relevant = False
        for doc_id in sorted(pool[qid]):
            doc_data = doc_meta(doc_id)
            if not doc_data:
                continue
            grade = assign_grade(doc_data, intent, known_doc_id, known_artist, known_genre, known_album)
            f.write(f"{qid}\t0\t{doc_id}\t{grade}\n")
            stats["total"] += 1
            stats[f"grade{grade}"] += 1
            if grade > 0:
                has_relevant = True
        if has_relevant:
            stats["queries_com_relevante"] += 1

print("\n=== NOVO QRELS ===")
print(f"Total pares       : {stats['total']}")
print(f"Grau 2 (alto)     : {stats['grade2']}")
print(f"Grau 1 (parcial)  : {stats['grade1']}")
print(f"Grau 0 (não rel.) : {stats['grade0']}")
print(f"Queries c/ rel.   : {stats['queries_com_relevante']} / {len(TOPICS)}")

# ── detalhamento por query ─────────────────────────────────────────────────────
print("\n=== Rel. por query ===")
with (OUTPUT_DIR / "qrels.tsv").open(encoding="utf-8") as f:
    rel_count: dict[str, int] = defaultdict(int)
    for line in f:
        p = line.strip().split("\t")
        if len(p) == 4 and int(p[3]) > 0:
            rel_count[p[0]] += 1

for qid in sorted(pool):
    t = topic_map.get(qid)
    intent = t[3] if t else "?"
    q_text = t[1][:40] if t else "?"
    n = rel_count.get(qid, 0)
    flag = " <<<" if n == 0 else ""
    print(f"  {qid} [{intent:6}] {n:3} rel  {q_text}{flag}")
