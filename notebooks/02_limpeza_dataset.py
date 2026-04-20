"""
03_limpeza_dataset.py
=====================
Pipeline completo de limpeza e normalização do dataset Spotify.

Etapas:
    1. Carregamento com diagnóstico inicial
    2. Remoção de duplicatas
    3. Tratamento de nulos — textuais e numéricos
    4. Normalização de strings
    5. Validação de ranges dos atributos de áudio
    6. Criação de colunas derivadas
    7. Relatório de limpeza
    8. Exportação de songs_clean.csv

Dependências:
    pip install pandas numpy
"""

import re
import string
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================

# Opção 1 — caminho absoluto baseado na localização do próprio script (mais robusto)
BASE       = Path(__file__).parent.parent / "data" / "spotify-metadata" / "spotify_clean_parquet"
AUDIO_BASE = Path(__file__).parent.parent / "data" / "spotify-metadata" / "spotify_clean_audio_features_parquet"
OUTPUT_PATH    = "songs_clean.csv"
RELATORIO_PATH = "relatorio_limpeza.txt"

# Colunas esperadas — o script se adapta se alguma não existir
TEXT_COLS    = ["track_name", "artist_name", "album_name", "track_genre", "genre"]
AUDIO_COLS   = ["acousticness", "danceability", "energy", "valence",
                "speechiness", "instrumentalness", "liveness"]
NUMERIC_COLS = ["popularity", "tempo", "loudness", "duration_ms",
                "key", "mode", "time_signature"]

# Ranges válidos para validação
AUDIO_RANGE    = (0.0, 1.0)
POPULARITY_RANGE = (0, 100)
TEMPO_RANGE    = (20.0, 320.0)
LOUDNESS_RANGE = (-60.0, 5.0)


# =============================================================================
# 1. HELPERS DE LOG
# =============================================================================

log_lines = []

def log(msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)

def separador(titulo: str = "") -> None:
    linha = "=" * 60
    if titulo:
        log(f"\n{linha}")
        log(f"  {titulo}")
        log(linha)
    else:
        log(linha)


# =============================================================================
# 2. CARREGAMENTO
# =============================================================================



def carregar_dataset(base: Path = BASE) -> pd.DataFrame:
    separador("1. CARREGAMENTO")

    tracks    = pd.read_parquet(base / "tracks.parquet")
    artists   = pd.read_parquet(base / "artists.parquet")
    t_artists = pd.read_parquet(base / "track_artists.parquet")
    albums    = pd.read_parquet(base / "albums.parquet")
    audio     = pd.read_parquet(AUDIO_BASE / "track_audio_features.parquet")

    try:
        genres = pd.read_parquet(base / "artist_genres.parquet")
        genres = genres.groupby("artist_rowid")["genre"].first().reset_index()
    except Exception:
        genres = None

    # Artista principal por faixa (primeiro artist_rowid de cada track_rowid)
    artist_primary = t_artists.drop_duplicates("track_rowid", keep="first")

    # Joins usando rowid como chave interna
    df = (
        tracks
        .rename(columns={"rowid": "track_rowid", "name": "track_name"})
        .merge(
            artist_primary,
            on="track_rowid", how="left"
        )
        .merge(
            artists[["rowid", "name"]]
                .rename(columns={"rowid": "artist_rowid", "name": "artist_name"}),
            on="artist_rowid", how="left"
        )
        .merge(
            albums[["rowid", "name"]]
                .rename(columns={"rowid": "album_rowid", "name": "album_name"}),
            left_on="album_rowid", right_on="album_rowid", how="left"
        )
        .merge(
            # audio liga pelo id externo do Spotify (coluna 'id' em tracks, 'track_id' em audio)
            audio.drop(columns=["fetched_at", "null_response", "duration_ms"], errors="ignore"),
            left_on="id", right_on="track_id", how="left"
        )
    )

    if genres is not None:
        df = df.merge(genres, on="artist_rowid", how="left")

    log(f"Arquivo base     : {base}")
    log(f"Linhas após join : {len(df):,}")
    log(f"Colunas          : {df.shape[1]}")
    log(f"Colunas presentes: {list(df.columns)}")
    return df

# =============================================================================
# 3. DIAGNÓSTICO INICIAL
# =============================================================================

def diagnostico(df: pd.DataFrame, label: str = "ANTES") -> dict:
    total = len(df)
    nulos = df.isnull().sum()
    stats = {
        "label"     : label,
        "total"     : total,
        "duplicatas": int(df.duplicated().sum()),
        "nulos"     : nulos[nulos > 0].to_dict(),
    }
    log(f"\n[{label}]  Linhas: {total:,}  |  Duplicatas: {stats['duplicatas']:,}")
    if stats["nulos"]:
        for col, n in stats["nulos"].items():
            log(f"  nulos em {col:<25}: {n:>7,}  ({n/total*100:5.2f}%)")
    else:
        log("  Nenhum nulo encontrado.")
    return stats


# =============================================================================
# 4. REMOÇÃO DE DUPLICATAS
# =============================================================================

def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    separador("2. REMOÇÃO DE DUPLICATAS")
    antes = len(df)

    # 4a. Linhas 100% idênticas
    df = df.drop_duplicates()
    apos_exato = len(df)
    log(f"Duplicatas exatas removidas : {antes - apos_exato:,}")

    # 4b. Duplicatas por chave de negócio (mesma música, metadados repetidos)
    key_cols = [c for c in ["track_name", "artist_name"] if c in df.columns]
    if key_cols:
        # Manter a linha com maior popularidade quando há empate
        sort_col = "popularity" if "popularity" in df.columns else key_cols[0]
        df = (
            df.sort_values(sort_col, ascending=False)
              .drop_duplicates(subset=key_cols, keep="first")
              .reset_index(drop=True)
        )
        apos_chave = len(df)
        log(f"Duplicatas por {key_cols} : {apos_exato - apos_chave:,}  (mantida a de maior popularidade)")
    else:
        apos_chave = apos_exato

    log(f"Total removidas             : {antes - apos_chave:,}  ({(antes - apos_chave)/antes*100:.2f}%)")
    log(f"Linhas restantes            : {apos_chave:,}")
    return df


# =============================================================================
# 5. NORMALIZAÇÃO DE STRINGS
# =============================================================================

# Padrão para conteúdo entre parênteses/colchetes que é ruído
_PATTERN_RUIDO = re.compile(
    r"\b(feat\.?|ft\.?|featuring|prod\.?|produced by|remix|remaster(?:ed)?|"
    r"version|edit|mix|extended|radio|live|acoustic|instrumental|deluxe|"
    r"edition|bonus|anniversary|reissue|explicit|official)\b.*",
    flags=re.IGNORECASE,
)
_PATTERN_PARENS  = re.compile(r"[\(\[\{].*?[\)\]\}]")
_PATTERN_ESPACOS = re.compile(r"\s{2,}")


def normalizar_string(texto: str, campo: str = "track") -> str:
    """
    Normalização de um campo textual:
    - Remove acentos (opcional — mantém compatibilidade cross-language)
    - Lowercase
    - Remove conteúdo entre parênteses e colchetes
    - Remove palavras de ruído (feat, remix, deluxe…)
    - Remove pontuação excessiva
    - Colapsa espaços
    - Strip final
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    # Normaliza encoding (NFD → remove acentos; NFC → mantém)
    # Use "NFC" para preservar acentos (recomendado para nomes de artistas)
    texto = unicodedata.normalize("NFC", texto)

    # Strip de espaços Unicode invisíveis
    texto = texto.strip("\u200b\u00a0\ufeff ")

    # Remove o que está entre parênteses e colchetes
    texto = _PATTERN_PARENS.sub(" ", texto)

    # Remove ruído de versões/colaborações
    # Para artist_name e album_name, preservamos mais informação
    if campo == "track":
        texto = _PATTERN_RUIDO.sub("", texto)

    # Remove pontuação excessiva (mantém letras, números e espaços)
    texto = re.sub(r"[^\w\s]", " ", texto)

    # Colapsa múltiplos espaços
    texto = _PATTERN_ESPACOS.sub(" ", texto).strip()

    return texto


def normalizar_colunas_textuais(df: pd.DataFrame) -> pd.DataFrame:
    separador("3. NORMALIZAÇÃO DE STRINGS")

    texto_cols_presentes = [c for c in TEXT_COLS if c in df.columns]

    for col in texto_cols_presentes:
        campo_tipo = "track" if col == "track_name" else "meta"

        # Preenche nulos com string vazia antes de normalizar
        df[col] = df[col].fillna("").astype(str)

        original_sample = df[col].iloc[0]
        df[col] = df[col].apply(lambda x: normalizar_string(x, campo=campo_tipo))
        normalizado_sample = df[col].iloc[0]

        vazios = (df[col] == "").sum()
        log(f"  {col:<20} | vazios após norm.: {vazios:>6,} | ex: '{original_sample}' → '{normalizado_sample}'")

    return df


# =============================================================================
# 6. TRATAMENTO DE NULOS — CAMPOS NUMÉRICOS
# =============================================================================

def tratar_nulos_numericos(df: pd.DataFrame) -> pd.DataFrame:
    separador("4. TRATAMENTO DE NULOS — NUMÉRICOS")

     # Forçar tipos numéricos nas colunas de áudio e numéricas
    # (podem vir como object após o join entre parquets)
    cols_para_converter = AUDIO_COLS + ["popularity", "tempo", "loudness",
                                        "duration_ms", "key", "mode", "time_signature"]
    for col in cols_para_converter:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    audio_presentes   = [c for c in AUDIO_COLS   if c in df.columns]
    numeric_presentes = [c for c in NUMERIC_COLS if c in df.columns]

    # Atributos de áudio (0–1): imputar com mediana por gênero se existir,
    # caso contrário mediana global
    genre_col = next((c for c in ["track_genre", "genre"] if c in df.columns), None)

    for col in audio_presentes:
        n_nulos = df[col].isnull().sum()
        if n_nulos == 0:
            continue

        if genre_col:
            mediana_genero = df.groupby(genre_col)[col].transform("median")
            mediana_global = df[col].median()
            df[col] = df[col].fillna(mediana_genero).fillna(mediana_global)
            log(f"  {col:<25} | {n_nulos:>6,} nulos → imputados com mediana por gênero")
        else:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            log(f"  {col:<25} | {n_nulos:>6,} nulos → imputados com mediana global ({mediana:.4f})")

    # Popularidade: imputar com 0 (música desconhecida = sem popularidade)
    if "popularity" in df.columns:
        n = df["popularity"].isnull().sum()
        df["popularity"] = df["popularity"].fillna(0).astype(int)
        if n > 0:
            log(f"  {'popularity':<25} | {n:>6,} nulos → imputados com 0")

    # Tempo (BPM): mediana global
    if "tempo" in df.columns:
        n = df["tempo"].isnull().sum()
        if n > 0:
            mediana = df["tempo"].median()
            df["tempo"] = df["tempo"].fillna(mediana)
            log(f"  {'tempo':<25} | {n:>6,} nulos → imputados com mediana ({mediana:.1f} BPM)")

    # Loudness: mediana global
    if "loudness" in df.columns:
        n = df["loudness"].isnull().sum()
        if n > 0:
            mediana = df["loudness"].median()
            df["loudness"] = df["loudness"].fillna(mediana)
            log(f"  {'loudness':<25} | {n:>6,} nulos → imputados com mediana ({mediana:.2f} dB)")

    # duration_ms: mediana global
    if "duration_ms" in df.columns:
        n = df["duration_ms"].isnull().sum()
        if n > 0:
            mediana = df["duration_ms"].median()
            df["duration_ms"] = df["duration_ms"].fillna(mediana).astype(int)
            log(f"  {'duration_ms':<25} | {n:>6,} nulos → imputados com mediana ({mediana/1000:.1f}s)")

    # key, mode, time_signature: moda (valor mais comum)
    for col in ["key", "mode", "time_signature"]:
        if col not in df.columns:
            continue
        n = df[col].isnull().sum()
        if n > 0:
            moda = df[col].mode()[0]
            df[col] = df[col].fillna(moda).astype(int)
            log(f"  {col:<25} | {n:>6,} nulos → imputados com moda ({moda})")

    total_nulos = df.isnull().sum().sum()
    log(f"\n  Nulos restantes no dataset: {total_nulos}")
    return df


# =============================================================================
# 7. TRATAMENTO DE NULOS — CAMPOS TEXTUAIS
# =============================================================================

def tratar_nulos_textuais(df: pd.DataFrame) -> pd.DataFrame:
    separador("5. TRATAMENTO DE NULOS — TEXTUAIS")

    # Após normalização, campos textuais já devem estar preenchidos.
    # Esta etapa garante que vazios residuais sejam sinalizados.
    texto_cols = [c for c in TEXT_COLS if c in df.columns]

    for col in texto_cols:
        # Tratar strings que são só espaço ou marcadores comuns de nulo
        mask_vazio = df[col].isin(["", "nan", "none", "null", "unknown", "-"])
        n_vazio = mask_vazio.sum()

        if n_vazio > 0:
            if col in ["track_name", "artist_name"]:
                # Faixas sem nome ou artista são inúteis para RI — remover
                log(f"  {col:<20} | {n_vazio:>6,} vazios → linhas serão removidas")
                df = df[~mask_vazio].reset_index(drop=True)
            else:
                # album_name, genre: substituir por placeholder
                placeholder = "desconhecido"
                df.loc[mask_vazio, col] = placeholder
                log(f"  {col:<20} | {n_vazio:>6,} vazios → substituídos por '{placeholder}'")
        else:
            log(f"  {col:<20} | sem vazios")

    return df


# =============================================================================
# 8. VALIDAÇÃO DE RANGES
# =============================================================================

def validar_e_corrigir_ranges(df: pd.DataFrame) -> pd.DataFrame:
    separador("6. VALIDAÇÃO DE RANGES")

    # Atributos de áudio: clamp para [0, 1]
    audio_presentes = [c for c in AUDIO_COLS if c in df.columns]
    for col in audio_presentes:
        fora = ((df[col] < 0) | (df[col] > 1)).sum()
        if fora > 0:
            df[col] = df[col].clip(0.0, 1.0)
            log(f"  {col:<25} | {fora:>6,} valores fora de [0,1] → corrigidos com clip")
        else:
            log(f"  {col:<25} | OK")

    # Popularidade: clamp [0, 100]
    if "popularity" in df.columns:
        fora = ((df["popularity"] < 0) | (df["popularity"] > 100)).sum()
        if fora > 0:
            df["popularity"] = df["popularity"].clip(0, 100)
            log(f"  {'popularity':<25} | {fora:>6,} fora de [0,100] → clip aplicado")

    # Tempo: remover outliers extremos (<20 ou >320 BPM)
    if "tempo" in df.columns:
        fora = ((df["tempo"] < TEMPO_RANGE[0]) | (df["tempo"] > TEMPO_RANGE[1])).sum()
        if fora > 0:
            mediana = df["tempo"].median()
            df.loc[(df["tempo"] < TEMPO_RANGE[0]) | (df["tempo"] > TEMPO_RANGE[1]), "tempo"] = mediana
            log(f"  {'tempo':<25} | {fora:>6,} BPMs inválidos → substituídos pela mediana")

    # Loudness: clamp [-60, 5] dB
    if "loudness" in df.columns:
        fora = ((df["loudness"] < LOUDNESS_RANGE[0]) | (df["loudness"] > LOUDNESS_RANGE[1])).sum()
        if fora > 0:
            df["loudness"] = df["loudness"].clip(*LOUDNESS_RANGE)
            log(f"  {'loudness':<25} | {fora:>6,} valores fora de [-60,5] dB → clip aplicado")

    return df


# =============================================================================
# 9. COLUNAS DERIVADAS (úteis para as Partes 2, 3 e 4)
# =============================================================================

def criar_colunas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    separador("7. COLUNAS DERIVADAS")

    # Campo de texto unificado para indexação BM25
    text_parts = [c for c in ["track_name", "artist_name", "album_name"] if c in df.columns]
    df["text_field"] = df[text_parts].fillna("").agg(" ".join, axis=1).str.strip()
    log(f"  text_field criado a partir de: {text_parts}")

    # Duração em segundos (mais legível que ms)
    if "duration_ms" in df.columns:
        df["duration_s"] = (df["duration_ms"] / 1000).round(1)
        log("  duration_s  = duration_ms / 1000")

    # Faixa de popularidade (útil para estratificação nos testes)
    if "popularity" in df.columns:
        df["popularity_tier"] = pd.cut(
            df["popularity"],
            bins=[-1, 0, 25, 50, 75, 100],
            labels=["zero", "baixa", "media", "alta", "viral"],
        )
        log("  popularity_tier: zero / baixa / media / alta / viral")

    # Score de "dançabilidade energética" (feature composta para LTR — Parte 3)
    if "energy" in df.columns and "danceability" in df.columns:
        df["energy_dance"] = ((df["energy"] + df["danceability"]) / 2).round(4)
        log("  energy_dance = (energy + danceability) / 2")

    # Flag: música instrumental (speechiness < 0.1 e instrumentalness > 0.8)
    if "instrumentalness" in df.columns and "speechiness" in df.columns:
        df["is_instrumental"] = (
            (df["instrumentalness"] > 0.8) & (df["speechiness"] < 0.1)
        ).astype(int)
        log("  is_instrumental: 1 se instrumentalness>0.8 e speechiness<0.1")

    return df


# =============================================================================
# 10. RELATÓRIO FINAL + EXPORTAÇÃO
# =============================================================================

def exportar(df: pd.DataFrame, stats_antes: dict, stats_depois: dict) -> None:
    separador("8. EXPORTAÇÃO")

    # Garantir tipos corretos antes de salvar
    int_cols = ["popularity", "key", "mode", "time_signature", "duration_ms"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    float_cols = [c for c in AUDIO_COLS + ["tempo", "loudness", "duration_s"] if c in df.columns]
    for col in float_cols:
        df[col] = df[col].round(6)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    log(f"Dataset salvo     : {OUTPUT_PATH}")
    log(f"Linhas exportadas : {len(df):,}")
    log(f"Colunas           : {df.shape[1]}")

    # Relatório de limpeza em texto
    separador("RESUMO DA LIMPEZA")
    reducao = stats_antes["total"] - stats_depois["total"]
    log(f"  Linhas originais : {stats_antes['total']:>10,}")
    log(f"  Linhas limpas    : {stats_depois['total']:>10,}")
    log(f"  Linhas removidas : {reducao:>10,}  ({reducao/stats_antes['total']*100:.2f}%)")
    log(f"  Duplicatas rem.  : {stats_antes['duplicatas']:>10,}")
    log(f"  Nulos restantes  : {df.isnull().sum().sum():>10,}")
    log(f"\n  Colunas no CSV final:")
    for col in df.columns:
        log(f"    {col}")

    # Salvar relatório em .txt
    with open(RELATORIO_PATH, "w", encoding="utf-8") as f:
        f.write(f"Relatório de limpeza — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("\n".join(log_lines))
    log(f"\nRelatório salvo   : {RELATORIO_PATH}")


# =============================================================================
# 11. PIPELINE PRINCIPAL
# =============================================================================

def pipeline_limpeza(path: str = BASE) -> pd.DataFrame:
    separador("PIPELINE DE LIMPEZA — DATASET SPOTIFY")
    log(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = carregar_dataset(path)
    stats_antes = diagnostico(df, "ANTES DA LIMPEZA")

    df = remover_duplicatas(df)
    df = normalizar_colunas_textuais(df)
    df = tratar_nulos_textuais(df)
    df = tratar_nulos_numericos(df)
    df = validar_e_corrigir_ranges(df)
    df = criar_colunas_derivadas(df)

    stats_depois = diagnostico(df, "DEPOIS DA LIMPEZA")
    exportar(df, stats_antes, stats_depois)

    log(f"\nFim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return df


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    df_clean = pipeline_limpeza()

    # Prévia das primeiras linhas
    print("\nPrévia do dataset limpo:")
    preview_cols = [c for c in ["track_name", "artist_name", "album_name",
                                 "popularity", "energy", "danceability",
                                 "text_field"] if c in df_clean.columns]
    print(df_clean[preview_cols].head(5).to_string(index=False))

    # tracks    = pd.read_parquet(BASE / "tracks.parquet")
    # artists   = pd.read_parquet(BASE / "artists.parquet")
    # t_artists = pd.read_parquet(BASE / "track_artists.parquet")
    # albums    = pd.read_parquet(BASE / "albums.parquet")
    # audio     = pd.read_parquet(AUDIO_BASE / "track_audio_features.parquet")

    # # DEBUG — adicione estas linhas temporariamente
    # print("tracks columns   :", tracks.columns.tolist())
    # print("artists columns  :", artists.columns.tolist())
    # print("t_artists columns:", t_artists.columns.tolist())
    # print("albums columns   :", albums.columns.tolist())
    # print("audio columns    :", audio.columns.tolist())
  
    # print(pd.read_parquet(BASE / "tracks.parquet").columns.tolist())
    # print(pd.read_parquet(BASE / "track_artists.parquet").columns.tolist())