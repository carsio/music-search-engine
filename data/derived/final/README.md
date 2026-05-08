# Dataset final brasileiro

Esta pasta concentra os artefatos versionaveis do dataset final do projeto. Os caches,
arquivos intermediarios e o Spotify Metadata bruto ficam fora daqui.

## Arquivos

| Arquivo | Linhas atuais | Significado |
| --- | ---: | --- |
| `br_curated_tracks.parquet` | 50.000 | Tabela principal de faixas brasileiras curadas a partir do Spotify Metadata. Contem metadados estruturados, popularidade, album, artistas, generos, imagens, mercados, audio features e metadados de arquivo. |
| `br_curated_lyrics.parquet` | 36.017 | Corpus de busca textual. E um recorte de `br_curated_tracks.parquet` apenas com faixas que possuem letra consolidada no cache de letras. |
| `br_artists.parquet` | 7.255 | Dimensao de artistas enriquecida a partir da Wikipedia PT. Gerada por `python -m music_search.scripts.export_entities` a partir do materializado em `enrichment_cache.sqlite`. |
| `br_albums.parquet` | 0 | Dimensao de albuns enriquecida a partir da Wikipedia PT. Ainda nao gerada no snapshot atual do manifesto. |
| `br_genres.parquet` | 42 | Dimensao de generos enriquecida a partir da Wikipedia PT. |
| `br_composers.parquet` | 0 | Dimensao de compositores/letristas enriquecida a partir da Wikipedia PT. |
| `br_dataset_manifest.json` | - | Manifesto com versao do dataset, data de geracao, tamanho, hash SHA1 e contagem de registros por arquivo. |

Snapshot atual do manifesto: versao `0.3.0`, gerado em `2026-05-08T13:00:52+00:00`.

No snapshot atual, `br_artists.parquet` e `br_genres.parquet` ja estao versionados.
`br_albums.parquet` e `br_composers.parquet` ainda podem nao existir. A aplicacao foi
escrita para degradar graciosamente quando alguma dessas dimensoes ainda nao foi gerada.

## `br_curated_tracks.parquet`

Grao: uma linha por faixa Spotify (`track_id`).

Origem: `data/spotify-metadata/spotify_clean_parquet/`, audio features e track files
do dataset Spotify Metadata local.

Principais grupos de colunas:

| Grupo | Colunas | Descricao |
| --- | --- | --- |
| Identificacao da faixa | `track_id`, `track_rowid`, `isrc`, `isrc_br`, `track_name`, `preview_url` | IDs Spotify/ISRC, nome e URL de preview quando disponivel. |
| Posicao no album | `track_number`, `disc_number` | Numero da faixa e disco no album. |
| Artistas | `primary_artist_id`, `primary_artist_name`, `primary_artist_followers_total`, `primary_artist_popularity`, `artist_ids`, `artist_names` | Artista primario e lista agregada de artistas da faixa. |
| Generos | `artist_genres`, `macro_genre` | Generos Spotify dos artistas e macro-genero curado usado no projeto. |
| Album | `album_id`, `album_rowid`, `album_name`, `album_type`, `album_label`, `album_popularity`, `album_total_tracks`, `album_upc`, `album_copyright_c`, `album_copyright_p` | Metadados estruturados do album. |
| Datas | `release_date`, `release_date_precision`, `release_year`, `decade` | Data original do Spotify e derivacoes para ano/decada. |
| Popularidade e duracao | `track_popularity`, `duration_ms`, `explicit` | Campos diretos do Spotify. |
| Mercados | `track_available_markets`, `track_available_markets_count`, `album_available_markets`, `album_available_markets_count` | Paises em que faixa/album estavam disponiveis no snapshot do dataset. |
| Imagens | `album_image_url`, `album_image_width`, `album_image_height`, `primary_artist_image_url`, `primary_artist_image_width`, `primary_artist_image_height` | Melhor imagem disponivel por album/artista, escolhida pela maior resolucao. |
| Audio features | `audio_features_available`, `time_signature`, `tempo`, `musical_key`, `musical_mode`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` | Caracteristicas musicais estruturadas do Spotify. |
| Arquivo Spotify | `track_file_status`, `track_file_session_country`, `language_of_performance`, `artist_roles`, `spotify_has_lyrics`, `licensor`, `original_title`, `version_title`, `content_ratings`, `filesize_bytes` | Metadados do arquivo/audio quando presentes no dataset bruto. |
| Auditoria | `track_fetched_at`, `audio_features_fetched_at`, `track_file_fetched_at` | Timestamps dos snapshots originais. |

Cobertura atual:

- `album_image_url`: 50.000 / 50.000 faixas.
- `primary_artist_image_url`: 49.705 / 50.000 faixas.
- `audio_features_available`: 49.865 / 50.000 faixas.
- `language_of_performance`: 49.914 / 50.000 faixas.

## `br_curated_lyrics.parquet`

Grao: uma linha por faixa com letra consolidada.

Origem: join entre `br_curated_tracks.parquet` e o cache local de letras
(`data/derived/lyrics_cache.sqlite`), via `python -m music_search.scripts.build_curated_corpus`.

Colunas principais:

| Coluna | Descricao |
| --- | --- |
| `id` | Mesmo ID Spotify de `track_id`. |
| `track_name`, `primary_artist_name`, `artist_names`, `artist_genres`, `macro_genre`, `album_name` | Metadados usados na busca textual. |
| `release_date`, `release_year`, `track_popularity`, `duration_ms`, `explicit` | Metadados adicionais para exibicao/filtros. |
| `lyrics_source`, `lyrics_source_url` | Fonte de onde a letra foi obtida. |
| `lyrics` | Texto completo da letra usado no indice BM25/TF-IDF. |

## Entidades enriquecidas pela Wikipedia

As dimensoes `br_artists.parquet`, `br_albums.parquet`, `br_genres.parquet` e
`br_composers.parquet` sao produzidas em outro passo:

Para `genres`, o default usa seeds detalhadas derivadas de `artist_genres`; use
`--seed-mode macro` se quiser restringir a coleta aos macro-generos curados.

```powershell
uv run python -m music_search.enrichment artists --limit 500 --concurrency 4
uv run python -m music_search.enrichment albums --limit 500 --concurrency 4
uv run python -m music_search.enrichment genres --concurrency 4
uv run python -m music_search.enrichment genres --concurrency 4 --seed-mode macro
uv run python -m music_search.enrichment composers --limit 500 --concurrency 4
uv run python -m music_search.scripts.export_entities
uv run python -m music_search.scripts.build_dataset --skip-lyrics

# UI Tk para rodar esse fluxo manualmente
uv run python -m music_search.enrichment.ui_tk
```

A interface mostra o progresso por etapa, logs e os artefatos finais gerados
(parquets de entidades + manifesto), para acompanhar o pipeline completo.

O pipeline baixa texto da Wikipedia PT, limpa esse conteudo e materializa um payload minimo e deterministico por entidade. Os parquets preservam `source`, `source_url` e `raw_text` para busca textual nas dimensoes.
Campos ja estruturados no Spotify, como popularidade, seguidores, imagens, datas, audio features e mercados, devem continuar vindo do pipeline deterministico.

## Como regenerar

```powershell
# Recria a tabela principal de tracks.
uv run python -m music_search.scripts.expand_dataset

# Consolida letras ja existentes no cache.
uv run python -m music_search.scripts.build_curated_corpus

# Atualiza manifest e exporta entidades ja enriquecidas.
uv run python -m music_search.scripts.build_dataset --skip-lyrics
```

Para versionar somente o dataset final:

```powershell
git add data/derived/final/README.md data/derived/final/br_*.parquet data/derived/final/br_dataset_manifest.json
```
