"""
02_stopwords_musicais.py
========================
Remoção de stopwords em metadados de músicas (track_name, artist_name, album_name).
Combina a lista padrão do NLTK (inglês) com stopwords customizadas do domínio musical.

Dependências:
    pip install nltk pandas
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixar recursos do NLTK (só precisa rodar uma vez)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)


# =============================================================================
# 1. DEFINIÇÃO DAS STOPWORDS
# =============================================================================

# --- 1a. Lista padrão do NLTK (inglês) ---
STOPWORDS_NLTK = set(stopwords.words("english"))

# --- 1b. Stopwords do domínio musical ---
# Palavras que aparecem muito em nomes de faixas/álbuns mas não
# ajudam a discriminar uma música de outra na busca.
STOPWORDS_MUSICAIS = {
    # Colaborações e créditos
    "feat", "ft", "featuring", "with", "vs", "versus",
    "presents", "pres", "prod", "produced",

    # Tipos de versão / edição
    "remix", "remixed", "remaster", "remastered",
    "version", "ver", "edit", "edited",
    "mix", "mixed", "extended", "radio",
    "original", "official", "explicit",
    "instrumental", "acapella", "acoustic",
    "live", "unplugged", "session", "sessions",
    "demo", "single", "bonus", "track",
    "deluxe", "edition", "expanded", "anniversary",
    "special", "limited", "collector",

    # Numerais e descritores comuns
    "part", "pt", "vol", "volume",
    "side", "disc", "disk", "cd",
    "intro", "outro", "interlude", "skit",
    "reprise", "medley",

    # Pontuação e símbolos que viram tokens
    "&", "-", "/", "|",
}

# --- 1c. União das duas listas ---
ALL_STOPWORDS = STOPWORDS_NLTK | STOPWORDS_MUSICAIS

print(f"Total de stopwords carregadas: {len(ALL_STOPWORDS)}")
print(f"  → NLTK (inglês): {len(STOPWORDS_NLTK)}")
print(f"  → Domínio musical: {len(STOPWORDS_MUSICAIS)}")


# =============================================================================
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO
# =============================================================================

def normalizar_texto(texto: str) -> str:
    """
    Normalização básica antes da tokenização:
    - Converte para minúsculas
    - Remove conteúdo entre parênteses e colchetes (ex: "(feat. Drake)")
    - Remove pontuação, exceto apóstrofos
    - Remove dígitos isolados
    """
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()

    # Remove o que está entre parênteses e colchetes (versões, featurings, etc.)
    texto = re.sub(r"\(.*?\)", " ", texto)
    texto = re.sub(r"\[.*?\]", " ", texto)

    # Remove pontuação (mantém apóstrofos por enquanto)
    texto = texto.translate(str.maketrans("", "", string.punctuation.replace("'", "")))

    # Remove apóstrofos
    texto = texto.replace("'", "")

    # Remove espaços extras
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def remover_stopwords(texto: str, stopwords_set: set = ALL_STOPWORDS) -> list[str]:
    """
    Tokeniza o texto e remove stopwords.

    Parâmetros:
        texto         : string já normalizada
        stopwords_set : conjunto de stopwords a aplicar

    Retorna:
        lista de tokens limpos (sem stopwords)
    """
    if not texto:
        return []

    tokens = word_tokenize(texto)
    tokens_limpos = [
        token
        for token in tokens
        if token not in stopwords_set and len(token) > 1
    ]
    return tokens_limpos


def preprocessar_campo(texto: str, stopwords_set: set = ALL_STOPWORDS) -> dict:
    """
    Pipeline completo para um campo de texto:
    normalização → tokenização → remoção de stopwords.

    Retorna dicionário com cada etapa para fins de auditoria.
    """
    original   = texto if isinstance(texto, str) else ""
    normalizado = normalizar_texto(original)
    tokens_raw  = word_tokenize(normalizado) if normalizado else []
    tokens_final = remover_stopwords(normalizado, stopwords_set)

    return {
        "original"       : original,
        "normalizado"    : normalizado,
        "tokens_raw"     : tokens_raw,
        "tokens_final"   : tokens_final,
        "texto_limpo"    : " ".join(tokens_final),
        "stopwords_rem"  : len(tokens_raw) - len(tokens_final),
        "reducao_%"      : round(
            (1 - len(tokens_final) / len(tokens_raw)) * 100, 1
        ) if tokens_raw else 0,
    }


# =============================================================================
# 3. APLICAÇÃO NO DATAFRAME
# =============================================================================

def aplicar_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o pipeline de pré-processamento em track_name, artist_name e album_name.
    Gera colunas '_clean' com o texto limpo e '_tokens' com a lista de tokens.
    Ao final, cria 'text_field' — campo unificado para o BM25.

    Parâmetros:
        df : DataFrame com as colunas originais

    Retorna:
        df enriquecido com as novas colunas
    """
    TEXT_COLS = {
        "track_name" : "track_name",
        "artist_name": "artist_name",
        "album_name" : "album_name",
    }

    for col_orig, col_base in TEXT_COLS.items():
        if col_orig not in df.columns:
            print(f"  [aviso] coluna '{col_orig}' não encontrada — pulando.")
            continue

        resultado = df[col_orig].apply(
            lambda x: preprocessar_campo(x, ALL_STOPWORDS)
        )

        df[f"{col_base}_clean"]  = resultado.apply(lambda r: r["texto_limpo"])
        df[f"{col_base}_tokens"] = resultado.apply(lambda r: r["tokens_final"])

        print(
            f"  {col_orig:15s} → redução média de stopwords: "
            f"{resultado.apply(lambda r: r['reducao_%']).mean():.1f}%"
        )

    # Campo unificado para o BM25 — concatena os três campos limpos
    clean_cols = [f"{c}_clean" for c in TEXT_COLS if f"{c}_clean" in df.columns]
    df["text_field"] = df[clean_cols].fillna("").agg(" ".join, axis=1).str.strip()

    # Tokens unificados (lista plana, usada pelo BM25Okapi)
    token_cols = [f"{c}_tokens" for c in TEXT_COLS if f"{c}_tokens" in df.columns]
    df["tokens_bm25"] = df[token_cols].apply(
        lambda row: sum(row.dropna().tolist(), []), axis=1
    )

    return df


# =============================================================================
# 4. DEMONSTRAÇÃO COM EXEMPLOS REAIS
# =============================================================================

exemplos = [
    {
        "track_name" : "Blinding Lights (feat. The Weeknd) [Radio Edit]",
        "artist_name": "The Weeknd",
        "album_name" : "After Hours (Deluxe Edition)",
    },
    {
        "track_name" : "God's Plan (Official Music Video)",
        "artist_name": "Drake ft. Future",
        "album_name" : "Scorpion (Expanded)",
    },
    {
        "track_name" : "Shape of You - Acoustic Version",
        "artist_name": "Ed Sheeran",
        "album_name" : "÷ (Divide) [Deluxe]",
    },
    {
        "track_name" : "Smells Like Teen Spirit - Remastered 2011",
        "artist_name": "Nirvana",
        "album_name" : "Nevermind (Anniversary Edition)",
    },
    {
        "track_name" : "HUMBLE. (Skrrrt Remix) ft. ScHoolboy Q",
        "artist_name": "Kendrick Lamar",
        "album_name" : "DAMN. (Collector's Edition)",
    },
]

df_demo = pd.DataFrame(exemplos)

print("\n" + "=" * 70)
print("DEMONSTRAÇÃO — PIPELINE DE STOPWORDS")
print("=" * 70)

df_demo = aplicar_preprocessing(df_demo)

# Exibir resultado por exemplo
for i, row in df_demo.iterrows():
    print(f"\n--- Exemplo {i+1} ---")
    print(f"  track_name original : {exemplos[i]['track_name']}")
    print(f"  track_name limpo    : {row['track_name_clean']}")
    print(f"  artist_name limpo   : {row['artist_name_clean']}")
    print(f"  album_name limpo    : {row['album_name_clean']}")
    print(f"  text_field (BM25)   : {row['text_field']}")
    print(f"  tokens_bm25         : {row['tokens_bm25']}")


# =============================================================================
# 5. APLICAÇÃO NO DATASET REAL
# =============================================================================

print("\n" + "=" * 70)
print("APLICAÇÃO NO DATASET REAL")
print("=" * 70)

# Descomente e ajuste o caminho para usar com seu CSV:

df = pd.read_csv("songs_clean.csv")
print(f"Dataset carregado: {len(df):,} linhas")

print("\nAplicando pré-processamento...")
df = aplicar_preprocessing(df)

# Salvar dataset com os novos campos
df.to_csv("songs_preprocessed.csv", index=False)
print("Salvo em: songs_preprocessed.csv")

# Salvar apenas as colunas necessárias para o BM25
cols_bm25 = ["track_name", "artist_name", "album_name",
             "text_field", "tokens_bm25",
             "acousticness", "danceability", "energy",
             "tempo", "popularity"]
cols_bm25 = [c for c in cols_bm25 if c in df.columns]
df[cols_bm25].to_csv("songs_for_bm25.csv", index=False)
print(f"Colunas para BM25 salvas em: songs_for_bm25.csv")

print("\nPara usar com o dataset real, descomente o bloco acima.")


# =============================================================================
# 6. UTILITÁRIO — INSPECIONAR UMA FAIXA ESPECÍFICA
# =============================================================================

def inspecionar_faixa(track: str, artist: str = "", album: str = "") -> None:
    """Mostra passo a passo o pipeline aplicado a uma única faixa."""
    print("\n" + "─" * 50)
    print("INSPEÇÃO PASSO A PASSO")
    print("─" * 50)

    for campo, valor in [("track_name", track), ("artist_name", artist), ("album_name", album)]:
        if not valor:
            continue
        r = preprocessar_campo(valor)
        print(f"\n[{campo}]")
        print(f"  Original     : {r['original']}")
        print(f"  Normalizado  : {r['normalizado']}")
        print(f"  Tokens raw   : {r['tokens_raw']}")
        print(f"  Tokens limpos: {r['tokens_final']}")
        print(f"  Stopwords removidas: {r['stopwords_rem']} ({r['reducao_%']}%)")


# Exemplo de uso do inspetor:
inspecionar_faixa(
    track  = "Lose Yourself (feat. Eminem) [Extended Mix] - Remastered",
    artist = "Eminem ft. D12",
    album  = "8 Mile: Music From the Motion Picture (Deluxe Edition)",
)