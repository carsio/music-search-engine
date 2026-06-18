"""
Gera a coleção de referência para avaliação dos motores de busca.

Saídas em OUTPUT_DIR:
  topics.tsv          50 consultas com ID, texto, descrição e intent
  protocolo.txt       critérios de relevância usados
  runs/bm25.txt       run file BM25 (formato TREC)
  runs/tfidf.txt      run file TF-IDF (formato TREC)
  runs/lyric.txt      run file busca lírica (formato TREC)
  runs/dense.txt      run file busca vetorial densa (formato TREC)
  qrels.tsv           julgamentos automáticos (formato TREC)
  resumo.txt          estatísticas da coleção

Pré-requisito para runs/dense.txt:
  O índice FAISS deve estar construído em data/indexes/dense_index.faiss.
  Se ausente, a etapa dense é pulada com aviso e o pool usa apenas 3 sistemas.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
OUTPUT_DIR = REPO_ROOT / "reference_collection"
DENSE_INDEX_PATH = REPO_ROOT / "data" / "indexes" / "dense_index.faiss"
DENSE_META_PATH = REPO_ROOT / "data" / "indexes" / "dense_meta.pkl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "runs").mkdir(exist_ok=True)

# ── 50 topics ─────────────────────────────────────────────────────────────────
# Campos: (id, query, description, intent, known_doc_id, known_artist, known_genre, known_album)
# known_doc_id: doc relevante de grau 2 quando existe (para known-item queries)
# Para queries de artista/gênero/álbum: usar os campos respectivos

TOPICS = [
    # ── LYRIC (15) ────────────────────────────────────────────────────────────
    (
        "Q001",
        "não sabe o que é sofrer ter que ver você sempre tão linda",
        "Usuário lembra de um trecho de uma música romântica MPB sobre alguém lindo.",
        "lyric",
        "0aASUtDb1N96NJDwmWj5Gf",
        "Los Hermanos",
        None,
        None,
    ),
    (
        "Q002",
        "vida loka sofrimento e glória da favela",
        "Usuário busca rap nacional sobre vida difícil nas periferias brasileiras.",
        "lyric",
        "6m8AgjfI28ER6odzMxmHtR",
        "Racionais MC's",
        None,
        None,
    ),
    (
        "Q003",
        "velha infância tribalistas",
        "Usuário quer encontrar a música sobre infância e nostalgia dos Tribalistas.",
        "lyric",
        "1mSxbLW7fKABfeY4lGpg0E",
        "Tribalistas",
        None,
        None,
    ),
    (
        "Q004",
        "ela quer rolar na minha silverado cowboy",
        "Usuário quer encontrar uma música sertaneja sobre caminhonete e vida de cowboy.",
        "lyric",
        "26l5FAEVr9cfaILS0szvvt",
        "Luan Pereira",
        None,
        None,
    ),
    (
        "Q005",
        "depois de tantos planos de um futuro para nós nos abandonamos",
        "Usuário lembra de trecho de música sobre o fim de um relacionamento longo.",
        "lyric",
        "2yeI0FRWd1sqGpHEUc5DFm",
        "Marisa Monte",
        None,
        None,
    ),
    (
        "Q006",
        "me encontra que espero a tanto tempo em meio à multidão",
        "Usuário busca a música do Charlie Brown Jr sobre esperar alguém especial.",
        "lyric",
        "5yEais1zgeW1MjLrx7tsie",
        "Charlie Brown Jr.",
        None,
        None,
    ),
    (
        "Q007",
        "um minuto para o fim do mundo punk rock",
        "Usuário busca música de rock brasileiro sobre urgência e fim do mundo.",
        "lyric",
        "7ATfTQCF4OGSY91yKk42km",
        "CPM 22",
        None,
        None,
    ),
    (
        "Q008",
        "louca por ti amor demais na praia nos dois",
        "Usuário busca música de forró ou arrocha sobre amor intenso à beira-mar.",
        "lyric",
        "21KbWPoY3MTpI5byU6d0dl",
        "Calcinha Preta",
        None,
        None,
    ),
    (
        "Q009",
        "há um vento fresco algo maravilhoso está por vir gospel",
        "Usuário busca música gospel de adoração sobre renovação e esperança.",
        "lyric",
        "6fP2IJGSEvzIFEGJDNWvf4",
        "Get Worship",
        None,
        None,
    ),
    (
        "Q010",
        "coração partido não me acostumei dividir você pagode",
        "Usuário busca pagode que fala sobre o sofrimento de dividir o amor com outra pessoa.",
        "lyric",
        "2sC9NcBWmZX2gt3gM0ZEmo",
        "Thiaguinho",
        None,
        None,
    ),
    (
        "Q011",
        "amor o meu coração pede só pra te amar ficar com você meu anjo",
        "Usuário busca música de forró romântico sobre querer amar e ficar com alguém.",
        "lyric",
        "5wV5dO1GOI4srs4Um5cJXo",
        "Junior Lima",
        None,
        None,
    ),
    (
        "Q012",
        "cedo ou tarde vamos nos encontrar rock nacional",
        "Usuário busca música de rock nacional sobre destino e encontro inevitável.",
        "lyric",
        "4RzFPIpnYo7vf0kQI8Gdg1",
        "NX Zero",
        None,
        None,
    ),
    (
        "Q013",
        "poderosa olhar de diamante agita o salão funk",
        "Usuário lembra de funk dos anos 2000 que elogia mulher elegante e poderosa.",
        "lyric",
        "2UeJXZ4W2gQw94Jcm2BlQ9",
        "MC Marcinho",
        None,
        None,
    ),
    (
        "Q014",
        "vai e vem não sai de órbita eita vida torta sertanejo",
        "Usuário busca música sertaneja de Marília Mendonça sobre relacionamento complicado.",
        "lyric",
        "4a5GPCo8oX4vvaRkciKlih",
        "Marília Mendonça",
        None,
        None,
    ),
    (
        "Q015",
        "mesmo que eu vá correndo sem direção ainda assim vou lutar gospel",
        "Usuário busca música gospel sobre perseverança mesmo sem direção clara.",
        "lyric",
        "726jL4DA03vptKr05so4Zu",
        "Central 3",
        None,
        None,
    ),
    # ── TRACK (12) ────────────────────────────────────────────────────────────
    (
        "Q016",
        "Anna Júlia Los Hermanos",
        "Usuário quer encontrar a música 'Anna Júlia' da banda Los Hermanos.",
        "track",
        "0aASUtDb1N96NJDwmWj5Gf",
        "Los Hermanos",
        None,
        None,
    ),
    (
        "Q017",
        "Velha Infância Tribalistas",
        "Usuário quer a música 'Velha Infância' do grupo Tribalistas.",
        "track",
        "1mSxbLW7fKABfeY4lGpg0E",
        "Tribalistas",
        None,
        None,
    ),
    (
        "Q018",
        "Vida Loka Racionais MC's",
        "Usuário procura 'Vida Loka Pt. 1' dos Racionais MC's.",
        "track",
        "6m8AgjfI28ER6odzMxmHtR",
        "Racionais MC's",
        None,
        None,
    ),
    (
        "Q019",
        "Dormi Na Praça Chitãozinho Xororó",
        "Usuário busca a música 'Dormi Na Praça' da dupla sertaneja Chitãozinho & Xororó.",
        "track",
        "0lqth62FJY6QobobChvnzB",
        "Chitãozinho & Xororó",
        None,
        None,
    ),
    (
        "Q020",
        "Obrigado por Estragar Tudo Marília Mendonça",
        "Usuário quer encontrar a música da Marília Mendonça sobre término de relacionamento.",
        "track",
        "4a5GPCo8oX4vvaRkciKlih",
        "Marília Mendonça",
        None,
        None,
    ),
    (
        "Q021",
        "Me Ama Ou Me Larga Simone Mendes",
        "Usuário busca a música 'Me Ama Ou Me Larga' de Simone Mendes.",
        "track",
        "2TSYxcrcrtLpdbsLY9NmzL",
        "Simone Mendes",
        None,
        None,
    ),
    (
        "Q022",
        "Apaguei Pra Todos Ferrugem",
        "Usuário quer a música 'Apaguei Pra Todos' do cantor Ferrugem.",
        "track",
        "65vJAh07BdwurqR9SRa6f8",
        "Ferrugem",
        None,
        None,
    ),
    (
        "Q023",
        "Dias Atrás CPM 22",
        "Usuário busca a música 'Dias Atrás' da banda de rock CPM 22.",
        "track",
        "7hjAhjnMzpMT9vU54w0LYF",
        "CPM 22",
        None,
        None,
    ),
    (
        "Q024",
        "Depois Marisa Monte",
        "Usuário quer a música 'Depois' de Marisa Monte sobre fim de um amor.",
        "track",
        "2yeI0FRWd1sqGpHEUc5DFm",
        "Marisa Monte",
        None,
        None,
    ),
    (
        "Q025",
        "Ovelhinha Isadora Pompeo",
        "Usuário busca a música gospel 'Ovelhinha' de Isadora Pompeo.",
        "track",
        "2lKiWWoeqOHhXwyZT8flw6",
        "Isadora Pompeo",
        None,
        None,
    ),
    (
        "Q026",
        "Silverado Luan Pereira sertanejo",
        "Usuário quer encontrar a música 'Silverado' do sertanejo Luan Pereira.",
        "track",
        "26l5FAEVr9cfaILS0szvvt",
        "Luan Pereira",
        None,
        None,
    ),
    (
        "Q027",
        "Insônia Tribo da Periferia rap",
        "Usuário busca a música 'Insônia' do grupo de rap Tribo da Periferia.",
        "track",
        "5G3ZjUMOlOpChRHpdlGALj",
        "Tribo da Periferia",
        None,
        None,
    ),
    # ── ARTIST (10) ───────────────────────────────────────────────────────────
    (
        "Q028",
        "Djavan",
        "Usuário quer músicas do cantor e compositor Djavan, referência da MPB.",
        "artist",
        None,
        "Djavan",
        None,
        None,
    ),
    (
        "Q029",
        "Racionais MC's rap",
        "Usuário quer músicas do grupo Racionais MC's, ícone do rap nacional.",
        "artist",
        None,
        "Racionais MC's",
        None,
        None,
    ),
    (
        "Q030",
        "Marília Mendonça sertanejo feminejo",
        "Usuário quer ouvir músicas da cantora sertaneja Marília Mendonça.",
        "artist",
        None,
        "Marília Mendonça",
        None,
        None,
    ),
    (
        "Q031",
        "Chitãozinho e Xororó dupla sertaneja",
        "Usuário quer músicas da dupla sertaneja clássica Chitãozinho & Xororó.",
        "artist",
        None,
        "Chitãozinho & Xororó",
        None,
        None,
    ),
    (
        "Q032",
        "Tribalistas MPB",
        "Usuário quer músicas do grupo Tribalistas, trio da MPB brasileira.",
        "artist",
        None,
        "Tribalistas",
        None,
        None,
    ),
    (
        "Q033",
        "CPM 22 rock nacional",
        "Usuário quer músicas da banda de rock CPM 22.",
        "artist",
        None,
        "CPM 22",
        None,
        None,
    ),
    (
        "Q034",
        "Ferrugem pagode",
        "Usuário quer músicas do cantor de pagode Ferrugem.",
        "artist",
        None,
        "Ferrugem",
        None,
        None,
    ),
    (
        "Q035",
        "Calcinha Preta forró",
        "Usuário quer músicas da banda de forró Calcinha Preta.",
        "artist",
        None,
        "Calcinha Preta",
        None,
        None,
    ),
    (
        "Q036",
        "Isadora Pompeo gospel",
        "Usuário quer músicas da cantora gospel Isadora Pompeo.",
        "artist",
        None,
        "Isadora Pompeo",
        None,
        None,
    ),
    (
        "Q037",
        "Jorge e Mateus sertanejo",
        "Usuário quer músicas da dupla sertaneja Jorge & Mateus.",
        "artist",
        None,
        "Jorge & Mateus",
        None,
        None,
    ),
    # ── GENRE (7) ─────────────────────────────────────────────────────────────
    (
        "Q038",
        "forró",
        "Usuário quer músicas do gênero forró, ritmo nordestino.",
        "genre",
        None,
        None,
        "forro_arrocha",
        None,
    ),
    (
        "Q039",
        "bossa nova MPB clássica",
        "Usuário quer músicas de bossa nova e MPB tradicional brasileira.",
        "genre",
        None,
        None,
        "mpb_bossa_choro",
        None,
    ),
    (
        "Q040",
        "samba pagode",
        "Usuário quer músicas do gênero samba e pagode brasileiro.",
        "genre",
        None,
        None,
        "pagode_samba",
        None,
    ),
    (
        "Q041",
        "funk carioca baile",
        "Usuário quer músicas do funk carioca para baile.",
        "genre",
        None,
        None,
        "funk",
        None,
    ),
    (
        "Q042",
        "gospel louvor adoração",
        "Usuário quer músicas gospel evangélicas de louvor e adoração.",
        "genre",
        None,
        None,
        "gospel",
        None,
    ),
    (
        "Q043",
        "rap nacional hip hop brasileiro",
        "Usuário quer músicas de rap e hip hop produzidas no Brasil.",
        "genre",
        None,
        None,
        "rap_trap",
        None,
    ),
    (
        "Q044",
        "sertanejo universitário",
        "Usuário quer músicas do sertanejo universitário contemporâneo.",
        "genre",
        None,
        None,
        "sertanejo",
        None,
    ),
    # ── ALBUM (6) ─────────────────────────────────────────────────────────────
    (
        "Q045",
        "album Tribalistas",
        "Usuário quer as músicas do álbum homônimo 'Tribalistas' do grupo Tribalistas.",
        "album",
        None,
        "Tribalistas",
        None,
        "Tribalistas",
    ),
    (
        "Q046",
        "Nada Como um Dia Após o Outro Dia Racionais",
        "Usuário quer músicas do álbum duplo dos Racionais MC's.",
        "album",
        None,
        "Racionais MC's",
        None,
        "Nada Como um Dia Após o Outro Dia",
    ),
    (
        "Q047",
        "Escândalo Íntimo Luísa Sonza",
        "Usuário quer as músicas do álbum 'Escândalo Íntimo' de Luísa Sonza.",
        "album",
        None,
        "Luísa Sonza",
        None,
        "Escândalo Íntimo",
    ),
    (
        "Q048",
        "Caetano e Bethânia ao vivo album",
        "Usuário quer as músicas do álbum ao vivo de Caetano Veloso e Maria Bethânia.",
        "album",
        None,
        "Caetano Veloso",
        None,
        "Caetano e Bethânia Ao Vivo",
    ),
    (
        "Q049",
        "Acústico Engenheiros do Hawaii",
        "Usuário quer as faixas do álbum acústico ao vivo dos Engenheiros do Hawaii.",
        "album",
        None,
        "Engenheiros Do Hawaii",
        None,
        "Acústico (Ao Vivo / Deluxe)",
    ),
    (
        "Q050",
        "Grandes Sucessos Raça Negra pagode",
        "Usuário quer o álbum de grandes sucessos do grupo de pagode Raça Negra.",
        "album",
        None,
        "Raça Negra",
        None,
        "Grandes Sucessos",
    ),
]


# ── helpers ───────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lowercase + remove acentos + remove pontuação."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", t)


def _contains(haystack: str, needle: str, threshold: float = 0.6) -> bool:
    """Verifica se needle aparece (normalizado) em haystack.

    Primeiro tenta match exato de substring (resolve nomes com apóstrofo como
    "MC's" → "mc s", "CPM 22" → "cpm 22"). Se não bater, usa fração de palavras.
    """
    h = normalize(haystack)
    n = normalize(needle)
    if not n:
        return False
    # Match exato como substring (cobre "racionais mc s" in "racionais mc s")
    if n in h:
        return True
    # Fallback: fração de palavras únicas da needle encontradas no haystack
    h_words = set(h.split())
    n_words = [w for w in n.split() if w]
    if not n_words:
        return False
    hits = sum(1 for w in n_words if w in h_words)
    return hits / len(n_words) >= threshold


def assign_grade(
    doc: dict,
    intent: str,
    known_doc_id: str | None,
    known_artist: str | None,
    known_genre: str | None,
    known_album: str | None,
) -> int:
    """
    Atribui grau 0/1/2 a um documento dado o contexto da consulta.

    Protocolo:
      2 = Altamente relevante: corresponde exatamente ao que a query busca
      1 = Parcialmente relevante: relacionado mas não exato
      0 = Não relevante
    """
    doc_id = doc.get("id", "")
    artist = doc.get("primary_artist_name") or doc.get("artist_names") or ""
    genres = doc.get("artist_genres", "") or ""
    macro = doc.get("macro_genre", "") or ""
    album = doc.get("album_name", "") or ""

    if intent == "lyric":
        # grau 2: documento exato pela ID
        if known_doc_id and doc_id == known_doc_id:
            return 2
        # grau 1: mesmo artista
        if known_artist and _contains(artist, known_artist):
            return 1
        return 0

    elif intent == "track":
        # grau 2: doc exato ou mesmo nome de track + artista
        if known_doc_id and doc_id == known_doc_id:
            return 2
        if known_artist and known_doc_id is None and _contains(artist, known_artist):
            return 1
        # grau 1: mesmo artista diferente track
        if known_artist and _contains(artist, known_artist):
            return 1
        return 0

    elif intent == "artist":
        # grau 2: artista exato
        if known_artist and _contains(artist, known_artist):
            return 2
        # grau 1: mesmo macro_genre
        if known_genre and macro == known_genre:
            return 1
        return 0

    elif intent == "genre":
        # grau 2: macro_genre exato
        if known_genre and macro == known_genre:
            return 2
        # grau 1: genres contém termo relacionado
        if known_genre:
            # mapeamento de termos de query para macro_genre
            genre_map = {
                "forro_arrocha": ["forro", "arrocha", "xote", "quadrilha"],
                "mpb_bossa_choro": ["bossa", "mpb", "choro", "samba jazz"],
                "pagode_samba": ["pagode", "samba", "axé"],
                "funk": ["funk", "baile"],
                "gospel": ["gospel", "worship", "louvor", "evangelica"],
                "rap_trap": ["rap", "hip hop", "trap", "drill"],
                "sertanejo": ["sertanejo", "caipira", "country br"],
            }
            related = genre_map.get(known_genre, [])
            if any(r in normalize(genres) for r in related):
                return 1
        return 0

    elif intent == "album":
        # grau 2: album_name contém o nome do álbum buscado
        if known_album and _contains(album, known_album, threshold=0.5):
            return 2
        # grau 1: mesmo artista
        if known_artist and _contains(artist, known_artist):
            return 1
        return 0

    return 0


# ── motor de busca ────────────────────────────────────────────────────────────


def load_engine():
    from music_search.motors.search import load_or_build_default_engine

    print("Carregando motor de busca...", flush=True)
    engine = load_or_build_default_engine()
    print(f"  {engine.num_docs} documentos indexados.", flush=True)
    return engine


def load_dense_engine():
    """Carrega o DenseSearchEngine a partir do índice FAISS salvo. Retorna None se ausente."""
    if not DENSE_INDEX_PATH.exists() or not DENSE_META_PATH.exists():
        print(
            f"  [AVISO] Índice FAISS não encontrado em {DENSE_INDEX_PATH}.\n"
            "          Execute primeiro: uv run python -m "
            "music_search.scripts.prepare_search_artifacts\n"
            "          O run file dense.txt será omitido do pool.",
            flush=True,
        )
        return None
    from music_search.motors.dense_search import DenseSearchEngine

    print("Carregando índice FAISS...", flush=True)
    engine = DenseSearchEngine.load(DENSE_INDEX_PATH, DENSE_META_PATH)
    print(f"  {engine.num_docs} documentos no índice denso.", flush=True)
    return engine


def run_dense_queries(dense_engine, topics, top_k: int = 20) -> dict[str, list[dict]]:
    """Roda a busca densa sobre todos os topics e retorna {qid: [hit_dict, ...]}."""
    results = {}
    for i, topic in enumerate(topics, 1):
        qid, query, *_ = topic
        hits = dense_engine.search(query, top_k=top_k)
        results[qid] = [h.to_dict() for h in hits]
        if i % 10 == 0:
            print(f"  [dense] {i}/{len(topics)} consultas processadas...", flush=True)
    return results


def run_queries(engine, topics, algorithm: str, top_k: int = 20) -> dict[str, list[dict]]:
    """Roda o algoritmo sobre todos os topics e retorna {qid: [hit_dict, ...]}."""
    results = {}
    for i, topic in enumerate(topics, 1):
        qid, query, *_ = topic
        intent = topic[3]
        # Para queries líricas, força boosts de letra
        from music_search.motors.tuning import SearchProfile

        profile: SearchProfile = "lyrics" if intent == "lyric" else "balanced"
        hits = engine.search(
            query,
            algorithm=algorithm,
            top_k=top_k,
            profile=profile,
        )
        results[qid] = [h.to_dict() for h in hits]
        if i % 10 == 0:
            print(f"  [{algorithm}] {i}/{len(topics)} consultas processadas...", flush=True)
    return results


def write_run_file(results: dict[str, list[dict]], path: Path, system_name: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid, hits in results.items():
            for rank, h in enumerate(hits, 1):
                doc_id = h.get("id", "")
                score = h.get("score", 0.0)
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{system_name}\n")


def build_pool(all_results: dict[str, dict[str, list[dict]]]) -> dict[str, dict[str, dict]]:
    """
    all_results: {system: {qid: [hit_dict]}}
    Retorna: {qid: {doc_id: hit_dict}} — union dos sistemas
    """
    pool: dict[str, dict[str, dict]] = defaultdict(dict)
    for system_hits in all_results.values():
        for qid, hits in system_hits.items():
            for h in hits:
                doc_id = h.get("id", "")
                if doc_id and doc_id not in pool[qid]:
                    pool[qid][doc_id] = h
    return dict(pool)


def write_qrels(pool: dict[str, dict[str, dict]], topics: list, path: Path) -> dict:
    """Escreve qrels e retorna estatísticas."""
    topic_map = {t[0]: t for t in topics}
    stats = {"total": 0, "grade2": 0, "grade1": 0, "grade0": 0, "queries_com_relevante": 0}

    with path.open("w", encoding="utf-8") as f:
        for qid, docs in sorted(pool.items()):
            topic = topic_map[qid]
            _, _, _, intent, known_doc_id, known_artist, known_genre, known_album = topic
            has_relevant = False
            for doc_id, hit in sorted(docs.items()):
                doc_data = hit.get("data_completa") or hit
                grade = assign_grade(
                    doc_data,
                    intent,
                    known_doc_id,
                    known_artist,
                    known_genre,
                    known_album,
                )
                f.write(f"{qid}\t0\t{doc_id}\t{grade}\n")
                stats["total"] += 1
                stats[f"grade{grade}"] += 1
                if grade > 0:
                    has_relevant = True
            if has_relevant:
                stats["queries_com_relevante"] += 1

    return stats


def write_topics(topics: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("query_id\tquery_text\tdescription\tintent\n")
        for t in topics:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\n")


def write_protocolo(path: Path) -> None:
    text = """\
PROTOCOLO DE RELEVÂNCIA — COLEÇÃO DE REFERÊNCIA
================================================
Projeto: Buscador de Músicas Brasileiras (ICC222 / UFAM 2026/1)
Data: 2026-06-14

ESCALA DE RELEVÂNCIA
--------------------
  Grau 2 — Altamente relevante
    O documento é exatamente o que a consulta busca.
    Critérios por intent:
      lyric : documento com a ID exata da música fonte do trecho de letra
      track : documento com ID ou nome/artista exatos da faixa buscada
      artist: documento cujo primary_artist_name corresponde ao artista da query
      genre : documento cujo macro_genre corresponde ao gênero da query
      album : documento cujo album_name corresponde ao álbum da query

  Grau 1 — Parcialmente relevante
    Documento relacionado mas não exato.
    Critérios por intent:
      lyric : outra música do mesmo artista da fonte do trecho
      track : outra música do mesmo artista da faixa buscada
      artist: música de artista do mesmo macro_genre do artista buscado
      genre : música cujo artist_genres contém termo relacionado ao gênero
      album : outra música do mesmo artista do álbum buscado

  Grau 0 — Não relevante
    Documento sem relação com a necessidade informacional.

NOTA SOBRE JULGAMENTOS AUTOMÁTICOS
------------------------------------
Esta coleção usa julgamentos automáticos baseados em correspondência de
metadados (known-item + metadata matching), constituindo um "silver standard".
Para uso em avaliação final recomenda-se revisão manual dos casos de grau 1,
especialmente para consultas de gênero, onde a correspondência de macro_genre
pode ser imprecisa.

POOL
----
  Sistemas: BM25, TF-IDF, Busca Lírica (BM25 com perfil lyrics), Busca Vetorial Densa (FAISS)
  Profundidade: top-20 por sistema
  Documentos unjudged (fora do pool): tratados como não recuperados no MAP.
  Use Bpref como métrica complementar para mitigar este efeito.

CONVERSÃO PARA BINÁRIO (MRR, MAP, P@10, Bpref)
-------------------------------------------------
  Relevante  = grau >= 1
  Não relevante = grau 0
"""
    path.write_text(text, encoding="utf-8")


def write_resumo(stats: dict, n_topics: int, n_systems: int, path: Path) -> None:
    dense_status = " (gerado)" if n_systems == 4 else " (AUSENTE — índice FAISS não encontrado)"
    text = f"""\
RESUMO DA COLEÇÃO DE REFERÊNCIA
================================
Consultas (topics)     : {n_topics}
Sistemas no pool       : {n_systems}
Total de pares julgados: {stats["total"]}
  Grau 2 (altamente)  : {stats["grade2"]}
  Grau 1 (parcial)    : {stats["grade1"]}
  Grau 0 (irrelevante): {stats["grade0"]}
Queries com >= 1 doc rel.: {stats["queries_com_relevante"]} / {n_topics}
Média docs por query   : {stats["total"] / n_topics:.1f}

DISTRIBUIÇÃO DE INTENTS
  lyric  : 15 consultas (Q001-Q015)
  track  : 12 consultas (Q016-Q027)
  artist : 10 consultas (Q028-Q037)
  genre  :  7 consultas (Q038-Q044)
  album  :  6 consultas (Q045-Q050)

ARQUIVOS GERADOS
  topics.tsv           → {n_topics} consultas (ID, texto, descrição, intent)
  protocolo.txt        → critérios de relevância
  qrels.tsv            → {stats["total"]} julgamentos no formato TREC
  runs/bm25.txt        → run file BM25
  runs/tfidf.txt       → run file TF-IDF
  runs/lyric.txt       → run file busca lírica
  runs/dense.txt       → run file busca vetorial densa{dense_status}
"""
    path.write_text(text, encoding="utf-8")
    print(text.encode("ascii", errors="replace").decode("ascii"))


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("BUILD TEST COLLECTION")
    print("=" * 60)

    engine = load_engine()

    print("\n[1/7] Escrevendo topics.tsv...")
    write_topics(TOPICS, OUTPUT_DIR / "topics.tsv")

    print("[2/7] Escrevendo protocolo.txt...")
    write_protocolo(OUTPUT_DIR / "protocolo.txt")

    all_results: dict[str, dict] = {}

    print("\n[3/7] Executando BM25...")
    bm25_results = run_queries(engine, TOPICS, "bm25", top_k=20)
    write_run_file(bm25_results, OUTPUT_DIR / "runs" / "bm25.txt", "bm25")
    all_results["bm25"] = bm25_results

    print("\n[4/7] Executando TF-IDF...")
    tfidf_results = run_queries(engine, TOPICS, "tfidf", top_k=20)
    write_run_file(tfidf_results, OUTPUT_DIR / "runs" / "tfidf.txt", "tfidf")
    all_results["tfidf"] = tfidf_results

    print("\n[5/7] Executando Busca Lírica (BM25 + perfil lyrics)...")
    lyric_results: dict[str, list[dict]] = {}
    for i, topic in enumerate(TOPICS, 1):
        qid, query = topic[0], topic[1]
        hits = engine.search(query, algorithm="bm25", top_k=20, profile="lyrics")
        lyric_results[qid] = [h.to_dict() for h in hits]
        if i % 10 == 0:
            print(f"  [lyric] {i}/{len(TOPICS)} consultas...", flush=True)
    write_run_file(lyric_results, OUTPUT_DIR / "runs" / "lyric.txt", "lyric")
    all_results["lyric"] = lyric_results

    print("\n[6/7] Executando Busca Vetorial Densa (FAISS)...")
    dense_engine = load_dense_engine()
    if dense_engine is not None:
        dense_results = run_dense_queries(dense_engine, TOPICS, top_k=20)
        write_run_file(dense_results, OUTPUT_DIR / "runs" / "dense.txt", "dense")
        all_results["dense"] = dense_results

    print("\n[7/7] Construindo pool e gerando qrels.tsv + resumo...")
    pool = build_pool(all_results)
    stats = write_qrels(pool, TOPICS, OUTPUT_DIR / "qrels.tsv")
    write_resumo(stats, len(TOPICS), len(all_results), OUTPUT_DIR / "resumo.txt")

    print(f"\nColeção salva em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
