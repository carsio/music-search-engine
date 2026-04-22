import duckdb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "spotify_unificado.csv")

def path(nome):
    return os.path.join(BASE_DIR, nome).replace("\\", "/")

print("Conectando ao DuckDB e processando joins...")

con = duckdb.connect()

query = f"""
COPY (
    WITH
    -- Melhor imagem por artista (maior resolução)
    melhor_imagem_artista AS (
        SELECT artist_rowid, url
        FROM (
            SELECT artist_rowid, url,
                   ROW_NUMBER() OVER (PARTITION BY artist_rowid ORDER BY COALESCE(width, 0) DESC) AS rn
            FROM read_parquet('{path("artist_images.parquet")}')
        )
        WHERE rn = 1
    ),

    -- Melhor imagem por album (maior resolução)
    melhor_imagem_album AS (
        SELECT album_rowid, url
        FROM (
            SELECT album_rowid, url,
                   ROW_NUMBER() OVER (PARTITION BY album_rowid ORDER BY COALESCE(width, 0) DESC) AS rn
            FROM read_parquet('{path("album_images.parquet")}')
        )
        WHERE rn = 1
    ),

    -- Artistas com generos e imagem agregados por track
    artistas_por_track AS (
        SELECT
            ta.track_rowid,
            string_agg(DISTINCT a.name,        ' | ' ORDER BY a.name)                   AS artist_names,
            string_agg(DISTINCT CAST(a.popularity AS VARCHAR),        ' | ')             AS artist_popularities,
            string_agg(DISTINCT CAST(a.followers_total AS VARCHAR),   ' | ')             AS artist_followers_total,
            string_agg(DISTINCT ag.genre,      ' | ' ORDER BY ag.genre)                 AS genres,
            string_agg(DISTINCT mia.url,       ' | ')                                   AS artist_image_urls
        FROM read_parquet('{path("track_artists.parquet")}')      ta
        LEFT JOIN read_parquet('{path("artists.parquet")}')        a   ON a.rowid  = ta.artist_rowid
        LEFT JOIN read_parquet('{path("artist_genres.parquet")}')  ag  ON ag.artist_rowid = ta.artist_rowid
        LEFT JOIN melhor_imagem_artista                            mia  ON mia.artist_rowid = ta.artist_rowid
        GROUP BY ta.track_rowid
    )

    SELECT
        -- Campos da faixa
        t.name                      AS track_name,
        t.preview_url               AS track_preview_url,
        t.track_number,
        t.disc_number,
        t.duration_ms,
        t.explicit,
        t.external_id_isrc,
        t.popularity                AS track_popularity,
        t.fetched_at                AS track_fetched_at,
        am_track.available_markets  AS track_available_markets,

        -- Campos do album
        al.name                     AS album_name,
        al.album_type,
        al.release_date,
        al.release_date_precision,
        al.total_tracks,
        al.label,
        al.popularity               AS album_popularity,
        al.external_id_upc,
        al.external_id_amgid,
        al.copyright_c,
        al.copyright_p,
        al.fetched_at               AS album_fetched_at,
        am_album.available_markets  AS album_available_markets,
        mia_album.url               AS album_image_url,

        -- Campos dos artistas (agregados)
        apt.artist_names,
        apt.artist_popularities,
        apt.artist_followers_total,
        apt.genres                  AS artist_genres,
        apt.artist_image_urls

    FROM read_parquet('{path("tracks.parquet")}')           t
    LEFT JOIN read_parquet('{path("albums.parquet")}')      al          ON al.rowid  = t.album_rowid
    LEFT JOIN read_parquet('{path("available_markets.parquet")}') am_track   ON am_track.rowid = t.available_markets_rowid
    LEFT JOIN read_parquet('{path("available_markets.parquet")}') am_album   ON am_album.rowid = al.available_markets_rowid
    LEFT JOIN melhor_imagem_album                           mia_album   ON mia_album.album_rowid = t.album_rowid
    LEFT JOIN artistas_por_track                            apt         ON apt.track_rowid = t.rowid

) TO '{OUTPUT_CSV.replace(chr(92), "/")}' (HEADER, DELIMITER ',');
"""

con.execute(query)
con.close()

print(f"CSV gerado com sucesso: {OUTPUT_CSV}")

# Exibe estatísticas básicas
import pandas as pd
df = pd.read_csv(OUTPUT_CSV, nrows=5)
print(f"\nColunas ({len(df.columns)}):")
for col in df.columns:
    print(f"  - {col}")
print(f"\nPrimeiras 5 linhas:")
print(df.to_string().encode("utf-8", errors="replace").decode("utf-8"))
