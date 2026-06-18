# Resultados do benchmark (gerado por `benchmark.py`)

- Corpus: `data/derived/final/br_curated_lyrics.parquet`
- Documentos indexados: **36017**
- Tempo de construção do índice (cold build): **84.33 s**
- Tamanho do índice persistido (`data/indexes/br_curated_lyrics.pkl`): **24.9 MB** (24915219 bytes)
- Repetições por consulta: 5 | consultas: 18

## Vocabulário por campo (termos distintos)

| Campo | Termos distintos |
| --- | ---: |
| `track_name` | 9809 |
| `artist_names` | 9212 |
| `artist_genres` | 355 |
| `macro_genre` | 19 |
| `album_name` | 8995 |
| `lyrics` | 61821 |

## Latência de consulta (ms)

| Algoritmo | Média | Mediana | p95 | Mín | Máx |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 34.4 | 22.8 | 89.2 | 0.5 | 106.3 |
| TFIDF | 30.0 | 17.9 | 79.7 | 0.4 | 97.4 |

## Comparação qualitativa BM25 × TF-IDF (top-3)

### Consulta: `amor saudade`

| # | BM25 | TF-IDF |
| --- | --- | --- |
| 1 | Saudade de Falar de Amor — BIN | Budah | Delacruz | Keviin | Malibu | Saudade de Você — Claudia Leitte | Filhos De Jorge |
| 2 | Saudade de Tu — Banda A Loba | Igor Guerra | Saudade de Você — Filhos De Jorge |
| 3 | S de Saudade — Luíza & Maurílio | Zé Neto & Cristiano | Saudade Que Fala - Ao Vivo — Pagode do Adame |

### Consulta: `chega de saudade`

| # | BM25 | TF-IDF |
| --- | --- | --- |
| 1 | Chega de Saudade — João Gilberto | Saudade de Você — Claudia Leitte | Filhos De Jorge |
| 2 | Chega de Saudade — João Gilberto | Saudade Que Fala - Ao Vivo — Pagode do Adame |
| 3 | Saudade — Fi Barreto | Saudade de Você — Filhos De Jorge |

### Consulta: `bossa nova`

| # | BM25 | TF-IDF |
| --- | --- | --- |
| 1 | bossa nova — aupinard | bossa nova — aupinard |
| 2 | Nova Bossa Nova — Marcos Valle | Nova Bossa Nova — Marcos Valle |
| 3 | Rio da Bossa Nova — Ambulante Discos | Beto Villares | Rio da Bossa Nova — Ambulante Discos | Beto Villares |
