# ruff: noqa: E501  # SQL strings ficam mais legiveis sem quebrar regex/CASE
"""Expande o dataset curado brasileiro preservando os hits ja conquistados
e excluindo as faixas que ja tentamos e deram MISS/ERROR/BLOCKED.

Motivacao
---------
O parquet inicial (`notebooks/04_dataset_curado_brasileiro.ipynb`) gerou ~22k
faixas com filtros conservadores (`per_artist_cap=5`, `top_k_per_bucket=250`,
`popularity_min=30`). Apos o pipeline de letras, ~80% viraram HIT (~17.9k) e
~19% viraram MISS (~4.3k). Para crescer a base, este script:

1. Le o cache de letras (`data/derived/lyrics_cache.sqlite`).
2. Mantem **garantidos** todos os track_id com status=hit.
3. Exclui (skip) os track_id ja tentados que terminaram em
   miss/error/blocked — nao adianta tentar de novo o que ja sabemos que falha.
4. Reaplica os filtros base (genero BR, popularidade, ano) com parametros
   relaxados:
   - `--per-artist-cap` default 50 (antes 5)
   - `--top-k-per-bucket` default 5000 (antes 250)
   - `--popularity-min` default 25 (antes 30)
5. Escolhe novas candidatas ranqueadas por (popularidade desc, track_rowid)
   ate completar `--target` faixas.
6. Enriquece a tabela com metadados ja existentes nos parquets originais:
   album/artist stats, IDs Spotify, imagens, mercados, audio features e
   metadados de arquivo quando disponiveis.
7. Salva o parquet final no mesmo caminho do dataset curado, sobrescrevendo.
   Os HITs ja resolvidos continuam sendo pulados pelo pipeline (idempotencia
   via cache); o pipeline so ira processar as **novas** faixas.

Uso
---
    # Gera ate 50k faixas (default)
    uv run python -m music_search.scripts.expand_dataset

    # Customizando
    uv run python -m music_search.scripts.expand_dataset \\
        --target 50000 \\
        --per-artist-cap 50 \\
        --top-k-per-bucket 5000 \\
        --popularity-min 25 \\
        --output data/derived/final/br_curated_tracks.parquet

    # Dry-run para inspecionar contagens sem escrever o parquet
    uv run python -m music_search.scripts.expand_dataset --dry-run

Apos rodar, retome o pipeline de letras:
    uv run python -m music_search.lyrics fetch --concurrency 4
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import duckdb

from music_search.data.datasets import DEFAULT_CURATED_TRACKS_PATH

# Reutilizamos os mesmos termos / regex do notebook 04 para manter coerencia
# sobre o que conta como "musica brasileira".
BR_TERMS = [
    "brazilian",
    "brazil",
    "brasil",
    "mpb",
    "samba",
    "sertanejo",
    "pagode",
    "forro",
    "bossa",
    "axe",
    "baiao",
    "funk carioca",
    "arrocha",
    "piseiro",
    "frevo",
    "maracatu",
    "tropicalia",
    "choro",
    "musica brasileira",
]
BR_REGEX = r"(^|[^a-z])(" + "|".join(BR_TERMS) + r")([^a-z]|$)"

DEFAULT_DATASET_DIR = Path("data/spotify-metadata")
DEFAULT_CLEAN_DIR = DEFAULT_DATASET_DIR / "spotify_clean_parquet"
DEFAULT_AUDIO_FEATURES_PATH = (
    DEFAULT_DATASET_DIR / "spotify_clean_audio_features_parquet" / "track_audio_features.parquet"
)
DEFAULT_TRACK_FILES_PATH = (
    DEFAULT_DATASET_DIR / "spotify_clean_track_files_parquet" / "track_files.parquet"
)
DEFAULT_OUTPUT = DEFAULT_CURATED_TRACKS_PATH
DEFAULT_CACHE = Path("data/derived/lyrics_cache.sqlite")


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Wrapper para `SELECT count(*) ...` — devolve int em vez de Row | None."""
    row = con.execute(sql).fetchone()
    if row is None:
        return 0
    return int(row[0])


def _read_cache_status(cache_path: Path) -> tuple[set[str], set[str]]:
    """Devolve (hits, dead_ends).

    - hits: track_ids com status='hit' — vamos manter no parquet sem tentar de
      novo (o pipeline pula via cache).
    - dead_ends: track_ids com status in ('miss', 'error', 'blocked') — vamos
      remover do parquet para liberar espaco.
    """
    if not cache_path.exists():
        return set(), set()
    con = sqlite3.connect(cache_path)
    try:
        hits = {row[0] for row in con.execute("SELECT track_id FROM lyrics WHERE status = 'hit'")}
        dead = {
            row[0]
            for row in con.execute(
                "SELECT track_id FROM lyrics WHERE status IN ('miss','error','blocked')"
            )
        }
    finally:
        con.close()
    return hits, dead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expande o parquet curado brasileiro com base no cache de letras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=DEFAULT_CLEAN_DIR,
        help="Diretorio com os parquets brutos do Spotify (apos download).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Cache SQLite de letras — usado para preservar hits e excluir dead-ends.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Caminho do parquet final.",
    )
    parser.add_argument(
        "--audio-features",
        type=Path,
        default=DEFAULT_AUDIO_FEATURES_PATH,
        help="Parquet de audio features do Spotify; se ausente, campos ficam NULL.",
    )
    parser.add_argument(
        "--track-files",
        type=Path,
        default=DEFAULT_TRACK_FILES_PATH,
        help="Parquet de metadados dos arquivos Spotify; se ausente, campos ficam NULL.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50_000,
        help="Numero alvo de faixas no parquet final.",
    )
    parser.add_argument(
        "--per-artist-cap",
        type=int,
        default=50,
        help="Maximo de faixas (novas) por artista primario. 0 = sem cap.",
    )
    parser.add_argument(
        "--top-k-per-bucket",
        type=int,
        default=5000,
        help="Maximo por (macro_genre, decade) entre as novas candidatas.",
    )
    parser.add_argument(
        "--popularity-min",
        type=int,
        default=25,
        help="Popularidade Spotify minima (0-100).",
    )
    parser.add_argument(
        "--release-year-min",
        type=int,
        default=1990,
        help="Ano de lancamento minimo.",
    )
    parser.add_argument(
        "--keep-dead-ends",
        action="store_true",
        help="Nao remove os MISS/ERROR/BLOCKED do parquet (default: remove).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="So imprime contagens; nao escreve o parquet final.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    required_files = [
        args.clean_dir / "tracks.parquet",
        args.clean_dir / "track_artists.parquet",
        args.clean_dir / "artists.parquet",
        args.clean_dir / "artist_genres.parquet",
        args.clean_dir / "albums.parquet",
        args.clean_dir / "available_markets.parquet",
        args.clean_dir / "album_images.parquet",
        args.clean_dir / "artist_images.parquet",
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Arquivos ausentes em "
            f"{args.clean_dir}. Rode `./src/music_search/scripts/download_spotify_metadata.sh --truncated` "
            "ou `--full`.\n" + "\n".join(str(p) for p in missing)
        )

    hits, dead_ends = _read_cache_status(args.cache)
    print(f"Cache em {args.cache}:")
    print(f"  hits:      {len(hits):,}")
    print(
        f"  dead-ends: {len(dead_ends):,} ({'mantidos' if args.keep_dead_ends else 'serao removidos'})"
    )
    print()

    tracks_path = _sql_path(args.clean_dir / "tracks.parquet")
    track_artists_path = _sql_path(args.clean_dir / "track_artists.parquet")
    artists_path = _sql_path(args.clean_dir / "artists.parquet")
    artist_genres_path = _sql_path(args.clean_dir / "artist_genres.parquet")
    albums_path = _sql_path(args.clean_dir / "albums.parquet")
    available_markets_path = _sql_path(args.clean_dir / "available_markets.parquet")
    album_images_path = _sql_path(args.clean_dir / "album_images.parquet")
    artist_images_path = _sql_path(args.clean_dir / "artist_images.parquet")
    audio_features_path = _sql_path(args.audio_features) if args.audio_features.exists() else None
    track_files_path = _sql_path(args.track_files) if args.track_files.exists() else None

    con = duckdb.connect()
    # Listas de track_ids como tabelas temporarias para JOIN/EXCLUDE eficientes.
    con.execute("CREATE TEMP TABLE hit_ids (track_id TEXT)")
    con.execute("CREATE TEMP TABLE dead_ids (track_id TEXT)")
    if hits:
        con.executemany("INSERT INTO hit_ids VALUES (?)", [(h,) for h in hits])
    if dead_ends and not args.keep_dead_ends:
        con.executemany("INSERT INTO dead_ids VALUES (?)", [(d,) for d in dead_ends])

    print(
        f"Filtros base: BR-genero, popularity>={args.popularity_min}, "
        f"release_year>={args.release_year_min}"
    )

    # ----- Camada 1: faixas com pelo menos um artista de genero BR
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE br_track_ids AS
        SELECT DISTINCT ta.track_rowid
        FROM read_parquet('{track_artists_path}') ta
        JOIN read_parquet('{artist_genres_path}') ag
          ON ag.artist_rowid = ta.artist_rowid
        WHERE regexp_matches(lower(strip_accents(coalesce(ag.genre, ''))), '{BR_REGEX}')
        """
    )

    # ----- Camadas 2-3: agregacoes + filtros de popularidade/ano
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE br_candidates AS
        WITH artist_profile AS (
            SELECT
                ta.track_rowid,
                min(ta.artist_rowid) AS primary_artist_rowid,
                string_agg(DISTINCT ar.id, ' | ' ORDER BY ar.id) AS artist_ids,
                string_agg(DISTINCT ar.name, ' | ' ORDER BY ar.name) AS artist_names
            FROM br_track_ids b
            JOIN read_parquet('{track_artists_path}') ta
              ON ta.track_rowid = b.track_rowid
            LEFT JOIN read_parquet('{artists_path}') ar
              ON ar.rowid = ta.artist_rowid
            GROUP BY ta.track_rowid
        ),
        genre_profile AS (
            SELECT
                ta.track_rowid,
                string_agg(DISTINCT ag.genre, ' | ' ORDER BY ag.genre) AS artist_genres
            FROM br_track_ids b
            JOIN read_parquet('{track_artists_path}') ta
              ON ta.track_rowid = b.track_rowid
            JOIN read_parquet('{artist_genres_path}') ag
              ON ag.artist_rowid = ta.artist_rowid
            WHERE coalesce(ag.genre, '') <> ''
            GROUP BY ta.track_rowid
        )
        SELECT
            t.id AS track_id,
            t.rowid AS track_rowid,
            t.fetched_at AS track_fetched_at,
            coalesce(t.name, '') AS track_name,
            coalesce(t.preview_url, '') AS preview_url,
            t.external_id_isrc AS isrc,
            CASE
                WHEN substr(upper(coalesce(t.external_id_isrc, '')), 1, 2) = 'BR'
                THEN TRUE ELSE FALSE
            END AS isrc_br,
            t.track_number,
            t.disc_number,
            ap.primary_artist_rowid,
            par.id AS primary_artist_id,
            par.name AS primary_artist_name,
            par.followers_total AS primary_artist_followers_total,
            par.popularity AS primary_artist_popularity,
            coalesce(ap.artist_ids, '') AS artist_ids,
            coalesce(ap.artist_names, '') AS artist_names,
            coalesce(gp.artist_genres, '') AS artist_genres,
            alb.rowid AS album_rowid,
            alb.id AS album_id,
            coalesce(alb.name, '') AS album_name,
            coalesce(alb.album_type, '') AS album_type,
            coalesce(alb.label, '') AS album_label,
            alb.popularity AS album_popularity,
            alb.total_tracks AS album_total_tracks,
            coalesce(alb.external_id_upc, '') AS album_upc,
            coalesce(alb.copyright_c, '') AS album_copyright_c,
            coalesce(alb.copyright_p, '') AS album_copyright_p,
            coalesce(alb.release_date, '') AS release_date,
            coalesce(alb.release_date_precision, '') AS release_date_precision,
            try_cast(regexp_extract(coalesce(alb.release_date, ''), '^[0-9]{{4}}') AS INTEGER) AS release_year,
            t.popularity AS track_popularity,
            t.duration_ms,
            cast(t.explicit AS BOOLEAN) AS explicit,
            coalesce(tm.available_markets, '') AS track_available_markets,
            CASE
                WHEN coalesce(tm.available_markets, '') IN ('', 'unavailable') THEN 0
                ELSE list_count(string_split(tm.available_markets, ','))
            END AS track_available_markets_count,
            coalesce(am.available_markets, '') AS album_available_markets,
            CASE
                WHEN coalesce(am.available_markets, '') IN ('', 'unavailable') THEN 0
                ELSE list_count(string_split(am.available_markets, ','))
            END AS album_available_markets_count
        FROM br_track_ids b
        JOIN read_parquet('{tracks_path}') t
          ON t.rowid = b.track_rowid
        LEFT JOIN read_parquet('{albums_path}') alb
          ON alb.rowid = t.album_rowid
        LEFT JOIN read_parquet('{available_markets_path}') tm
          ON tm.rowid = t.available_markets_rowid
        LEFT JOIN read_parquet('{available_markets_path}') am
          ON am.rowid = alb.available_markets_rowid
        LEFT JOIN artist_profile ap
          ON ap.track_rowid = t.rowid
        LEFT JOIN read_parquet('{artists_path}') par
          ON par.rowid = ap.primary_artist_rowid
        LEFT JOIN genre_profile gp
          ON gp.track_rowid = t.rowid
        WHERE t.popularity >= {args.popularity_min}
          AND try_cast(regexp_extract(coalesce(alb.release_date, ''), '^[0-9]{{4}}') AS INTEGER) >= {args.release_year_min}
        """
    )
    n_cands = _count(con, "SELECT count(*) FROM br_candidates")
    print(f"Candidatas brutas (filtros base): {n_cands:,}")

    # ----- Camada 4: macrogenero + decada
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE br_with_macro AS
        SELECT
            *,
            cast(floor(release_year / 10) * 10 AS INTEGER) AS decade,
            CASE
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(gospel)([^a-z]|$)') THEN 'gospel'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(sertanejo|agronejo)([^a-z]|$)') THEN 'sertanejo'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(funk)([^a-z]|$)') THEN 'funk'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(forro|arrocha|piseiro|seresta|baiao)([^a-z]|$)') THEN 'forro_arrocha'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(pagode|samba)([^a-z]|$)') THEN 'pagode_samba'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(mpb|bossa|choro|tropicalia)([^a-z]|$)') THEN 'mpb_bossa_choro'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(axe|frevo|maracatu|brega|tecnobrega)([^a-z]|$)') THEN 'axe_regional'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(rap|trap|hip hop)([^a-z]|$)') THEN 'rap_trap'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(phonk)([^a-z]|$)') THEN 'phonk'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(rock)([^a-z]|$)') THEN 'rock_br'
                WHEN regexp_matches(lower(strip_accents(artist_genres)), '(^|[^a-z])(pop)([^a-z]|$)') THEN 'pop_br'
                ELSE 'outros'
            END AS macro_genre
        FROM br_candidates
        """
    )

    # ----- Marcar HIT / DEAD para priorizacao
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE br_marked AS
        SELECT
            m.*,
            CASE WHEN h.track_id IS NOT NULL THEN TRUE ELSE FALSE END AS is_hit,
            CASE WHEN d.track_id IS NOT NULL THEN TRUE ELSE FALSE END AS is_dead
        FROM br_with_macro m
        LEFT JOIN hit_ids h ON h.track_id = m.track_id
        LEFT JOIN dead_ids d ON d.track_id = m.track_id
        """
    )

    # Estatisticas pre-rank
    n_hits_in_pool = _count(con, "SELECT count(*) FROM br_marked WHERE is_hit")
    n_dead_in_pool = _count(con, "SELECT count(*) FROM br_marked WHERE is_dead")
    print(f"  hits ja resolvidos no pool atual: {n_hits_in_pool:,}")
    print(f"  dead-ends no pool atual:         {n_dead_in_pool:,}")

    # ----- Aplicar caps SO nas faixas novas (sem hit) — hits sempre passam.
    # Caso `keep_dead_ends`, dead também passa pelo cap normal; senão é excluído.
    cap_clause = f"AND rn_artist <= {args.per_artist_cap}" if args.per_artist_cap > 0 else ""
    bucket_clause = f"AND rn_bucket <= {args.top_k_per_bucket}" if args.top_k_per_bucket > 0 else ""

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE br_ranked AS
        WITH base AS (
            SELECT *
            FROM br_marked
            WHERE is_hit = TRUE
               OR (is_dead = FALSE OR {1 if args.keep_dead_ends else 0} = 1)
        ),
        with_artist_rn AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY primary_artist_rowid
                    ORDER BY is_hit DESC, track_popularity DESC, track_rowid
                ) AS rn_artist
            FROM base
        ),
        artist_capped AS (
            SELECT *
            FROM with_artist_rn
            WHERE is_hit = TRUE
               OR (1 = 1 {cap_clause})
        ),
        with_bucket_rn AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY macro_genre, decade
                    ORDER BY is_hit DESC, track_popularity DESC, track_rowid
                ) AS rn_bucket
            FROM artist_capped
        )
        SELECT *
        FROM with_bucket_rn
        WHERE is_hit = TRUE
           OR (1 = 1 {bucket_clause})
        """
    )
    n_after_caps = _count(con, "SELECT count(*) FROM br_ranked")
    print(
        f"\nApos caps (per_artist={args.per_artist_cap}, top_k_bucket={args.top_k_per_bucket}): {n_after_caps:,}"
    )

    # ----- Selecao final: priorizar hits, depois popularidade
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE br_final AS
        SELECT *
        FROM br_ranked
        ORDER BY is_hit DESC, track_popularity DESC, track_rowid
        LIMIT {args.target}
        """
    )
    n_final = _count(con, "SELECT count(*) FROM br_final")
    n_final_hits = _count(con, "SELECT count(*) FROM br_final WHERE is_hit")
    n_final_new = n_final - n_final_hits
    print(f"\nSelecao final: {n_final:,} faixas")
    print(f"  hits preservados: {n_final_hits:,}")
    print(f"  novas candidatas: {n_final_new:,}")

    print("\nDistribuicao por macrogenero (final):")
    rows = con.execute(
        """
        SELECT
            macro_genre,
            count(*) AS faixas,
            sum(CASE WHEN is_hit THEN 1 ELSE 0 END) AS com_letra,
            count(DISTINCT primary_artist_rowid) AS artistas
        FROM br_final
        GROUP BY macro_genre
        ORDER BY faixas DESC
        """
    ).fetchall()
    print(f"  {'macro_genre':<18} {'faixas':>8} {'com_letra':>10} {'artistas':>9}")
    for mg, n, n_hit, n_art in rows:
        print(f"  {mg:<18} {n:>8,} {n_hit:>10,} {n_art:>9,}")

    if args.dry_run:
        print("\n--dry-run: nao escrevendo o parquet.")
        con.close()
        return

    print("\nEnriquecendo selecao final com imagens, audio features e metadados de arquivo...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE album_best_images AS
        WITH album_ids AS (
            SELECT DISTINCT album_rowid
            FROM br_final
            WHERE album_rowid IS NOT NULL
        ),
        ranked AS (
            SELECT
                ai.album_rowid,
                ai.url,
                ai.width,
                ai.height,
                row_number() OVER (
                    PARTITION BY ai.album_rowid
                    ORDER BY ai.width * ai.height DESC NULLS LAST, ai.url
                ) AS rn
            FROM read_parquet('{album_images_path}') ai
            JOIN album_ids ids ON ids.album_rowid = ai.album_rowid
        )
        SELECT album_rowid, url, width, height
        FROM ranked
        WHERE rn = 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE artist_best_images AS
        WITH artist_ids AS (
            SELECT DISTINCT primary_artist_rowid AS artist_rowid
            FROM br_final
            WHERE primary_artist_rowid IS NOT NULL
        ),
        ranked AS (
            SELECT
                ai.artist_rowid,
                ai.url,
                ai.width,
                ai.height,
                row_number() OVER (
                    PARTITION BY ai.artist_rowid
                    ORDER BY ai.width * ai.height DESC NULLS LAST, ai.url
                ) AS rn
            FROM read_parquet('{artist_images_path}') ai
            JOIN artist_ids ids ON ids.artist_rowid = ai.artist_rowid
        )
        SELECT artist_rowid, url, width, height
        FROM ranked
        WHERE rn = 1
        """
    )

    if audio_features_path:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE selected_audio_features AS
            WITH ids AS (
                SELECT DISTINCT track_id FROM br_final
            ),
            ranked AS (
                SELECT
                    af.*,
                    row_number() OVER (
                        PARTITION BY af.track_id
                        ORDER BY try_cast(af.fetched_at AS BIGINT) DESC NULLS LAST
                    ) AS rn
                FROM read_parquet('{audio_features_path}') af
                JOIN ids ON ids.track_id = af.track_id
            )
            SELECT
                track_id,
                try_cast(fetched_at AS BIGINT) AS audio_features_fetched_at,
                CASE
                    WHEN lower(coalesce(null_response, '0')) IN ('1', 'true') THEN FALSE
                    ELSE TRUE
                END AS audio_features_available,
                try_cast(time_signature AS INTEGER) AS time_signature,
                try_cast(tempo AS DOUBLE) AS tempo,
                try_cast(key AS INTEGER) AS musical_key,
                try_cast(mode AS INTEGER) AS musical_mode,
                try_cast(danceability AS DOUBLE) AS danceability,
                try_cast(energy AS DOUBLE) AS energy,
                try_cast(loudness AS DOUBLE) AS loudness,
                try_cast(speechiness AS DOUBLE) AS speechiness,
                try_cast(acousticness AS DOUBLE) AS acousticness,
                try_cast(instrumentalness AS DOUBLE) AS instrumentalness,
                try_cast(liveness AS DOUBLE) AS liveness,
                try_cast(valence AS DOUBLE) AS valence
            FROM ranked
            WHERE rn = 1
            """
        )
    else:
        print(f"  [skip] audio features ausente: {args.audio_features}")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE selected_audio_features AS
            SELECT
                NULL::VARCHAR AS track_id,
                NULL::BIGINT AS audio_features_fetched_at,
                NULL::BOOLEAN AS audio_features_available,
                NULL::INTEGER AS time_signature,
                NULL::DOUBLE AS tempo,
                NULL::INTEGER AS musical_key,
                NULL::INTEGER AS musical_mode,
                NULL::DOUBLE AS danceability,
                NULL::DOUBLE AS energy,
                NULL::DOUBLE AS loudness,
                NULL::DOUBLE AS speechiness,
                NULL::DOUBLE AS acousticness,
                NULL::DOUBLE AS instrumentalness,
                NULL::DOUBLE AS liveness,
                NULL::DOUBLE AS valence
            WHERE FALSE
            """
        )

    if track_files_path:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE selected_track_files AS
            WITH ids AS (
                SELECT DISTINCT track_id FROM br_final
            ),
            ranked AS (
                SELECT
                    tf.*,
                    row_number() OVER (
                        PARTITION BY tf.track_id
                        ORDER BY
                            CASE WHEN tf.status = 'success' THEN 1 ELSE 0 END DESC,
                            tf.fetched_at DESC NULLS LAST
                    ) AS rn
                FROM read_parquet('{track_files_path}') tf
                JOIN ids ON ids.track_id = tf.track_id
            )
            SELECT
                track_id,
                fetched_at AS track_file_fetched_at,
                coalesce(status, '') AS track_file_status,
                coalesce(session_country, '') AS track_file_session_country,
                coalesce(language_of_performance, '') AS language_of_performance,
                coalesce(artist_roles, '') AS artist_roles,
                cast(has_lyrics AS BOOLEAN) AS spotify_has_lyrics,
                coalesce(licensor, '') AS licensor,
                coalesce(original_title, '') AS original_title,
                coalesce(version_title, '') AS version_title,
                coalesce(content_ratings, '') AS content_ratings,
                filesize_bytes
            FROM ranked
            WHERE rn = 1
            """
        )
    else:
        print(f"  [skip] track files ausente: {args.track_files}")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE selected_track_files AS
            SELECT
                NULL::VARCHAR AS track_id,
                NULL::BIGINT AS track_file_fetched_at,
                NULL::VARCHAR AS track_file_status,
                NULL::VARCHAR AS track_file_session_country,
                NULL::VARCHAR AS language_of_performance,
                NULL::VARCHAR AS artist_roles,
                NULL::BOOLEAN AS spotify_has_lyrics,
                NULL::VARCHAR AS licensor,
                NULL::VARCHAR AS original_title,
                NULL::VARCHAR AS version_title,
                NULL::VARCHAR AS content_ratings,
                NULL::BIGINT AS filesize_bytes
            WHERE FALSE
            """
        )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE br_final_enriched AS
        SELECT
            f.*,
            abi.url AS album_image_url,
            abi.width AS album_image_width,
            abi.height AS album_image_height,
            arbi.url AS primary_artist_image_url,
            arbi.width AS primary_artist_image_width,
            arbi.height AS primary_artist_image_height,
            af.audio_features_fetched_at,
            coalesce(af.audio_features_available, FALSE) AS audio_features_available,
            af.time_signature,
            af.tempo,
            af.musical_key,
            af.musical_mode,
            af.danceability,
            af.energy,
            af.loudness,
            af.speechiness,
            af.acousticness,
            af.instrumentalness,
            af.liveness,
            af.valence,
            tf.track_file_fetched_at,
            coalesce(tf.track_file_status, '') AS track_file_status,
            coalesce(tf.track_file_session_country, '') AS track_file_session_country,
            coalesce(tf.language_of_performance, '') AS language_of_performance,
            coalesce(tf.artist_roles, '') AS artist_roles,
            tf.spotify_has_lyrics,
            coalesce(tf.licensor, '') AS licensor,
            coalesce(tf.original_title, '') AS original_title,
            coalesce(tf.version_title, '') AS version_title,
            coalesce(tf.content_ratings, '') AS content_ratings,
            tf.filesize_bytes
        FROM br_final f
        LEFT JOIN album_best_images abi
          ON abi.album_rowid = f.album_rowid
        LEFT JOIN artist_best_images arbi
          ON arbi.artist_rowid = f.primary_artist_rowid
        LEFT JOIN selected_audio_features af
          ON af.track_id = f.track_id
        LEFT JOIN selected_track_files tf
          ON tf.track_id = f.track_id
        """
    )

    # ----- Exportar
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_sql_path = _sql_path(args.output)
    con.execute(
        f"""
        COPY (
            SELECT
                track_id,
                track_rowid,
                track_fetched_at,
                isrc,
                isrc_br,
                track_name,
                preview_url,
                track_number,
                disc_number,
                primary_artist_id,
                primary_artist_name,
                primary_artist_followers_total,
                primary_artist_popularity,
                artist_ids,
                artist_names,
                artist_genres,
                macro_genre,
                album_id,
                album_rowid,
                album_name,
                album_type,
                album_label,
                album_popularity,
                album_total_tracks,
                album_upc,
                album_copyright_c,
                album_copyright_p,
                release_date,
                release_date_precision,
                release_year,
                decade,
                track_popularity,
                duration_ms,
                explicit,
                track_available_markets,
                track_available_markets_count,
                album_available_markets,
                album_available_markets_count,
                album_image_url,
                album_image_width,
                album_image_height,
                primary_artist_image_url,
                primary_artist_image_width,
                primary_artist_image_height,
                audio_features_fetched_at,
                audio_features_available,
                time_signature,
                tempo,
                musical_key,
                musical_mode,
                danceability,
                energy,
                loudness,
                speechiness,
                acousticness,
                instrumentalness,
                liveness,
                valence,
                track_file_fetched_at,
                track_file_status,
                track_file_session_country,
                language_of_performance,
                artist_roles,
                spotify_has_lyrics,
                licensor,
                original_title,
                version_title,
                content_ratings,
                filesize_bytes
            FROM br_final_enriched
            ORDER BY is_hit DESC, track_popularity DESC, track_rowid
        ) TO '{output_sql_path}' (FORMAT PARQUET)
        """
    )
    con.close()

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nExportado: {args.output} ({size_mb:.2f} MB, {n_final:,} faixas)")
    print(
        "Proximo passo: rode o pipeline de letras para baixar as "
        f"{n_final_new:,} novas faixas:\n"
        "  uv run python -m music_search.lyrics fetch --concurrency 4"
    )


if __name__ == "__main__":
    main()
