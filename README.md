# Music Search Engine

**Busca de músicas brasileiras com BM25, TF-IDF e embeddings vetoriais.**

Trabalho da disciplina ICC222 — Tópicos em Recuperação de Informação (UFAM 2026/1).

📊 Slides: <https://carsio.github.io/music-search-engine/>

## Visão geral

O projeto implementa os algoritmos clássicos de RI (índice invertido, TF-IDF, BM25) e a busca vetorial densa (embeddings + Milvus), e expõe esse motor através de uma API FastAPI consumida por um frontend React. Os dados foram curados a partir do Spotify Metadata e enriquecidos com letras (lyrics.ovh, Vagalume, letras.mus.br, Genius) e biografias da Wikipedia processadas por LLM.

## Como o projeto está organizado

```
┌─────────────────────────────────────────────────────────────────┐
│  Camada 1 — Core de RI (algoritmos do trabalho)                 │
│  preprocessing → indexer → ranking (BM25, TF-IDF) → evaluation  │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│  Camada 2 — Motores de busca                                    │
│  search.SparseSearchEngine   (multi-campo sobre tracks)         │
│  multi_index.MultiEntityIndex (artist/album/genre/composer)     │
│  vector.VectorSearch         (embeddings + Milvus)              │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│  Camada 3 — Apresentação                                        │
│  ui_tk           (GUI Tk para comparar BM25 x TF-IDF)           │
│  vector/ui_tk    (GUI Tk para inspecionar busca vetorial)       │
│  web/app         (API FastAPI)         ← consumida pelo →       │
│  frontend/       (SPA React + Vite)                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Pipeline de dados (rodado offline; gera os parquets versionados)│
│                                                                 │
│  Spotify raw → notebook 04 → tracks brasileiros                 │
│         ↓                                                       │
│  lyrics/  (cascata: letras.mus.br → vagalume → lyrics.ovh →     │
│            genius, com cache SQLite e circuit breaker)          │
│         ↓                                                       │
│  scripts/build_curated_corpus.py                                │
│         ↓                                                       │
│  data/derived/final/br_curated_lyrics.parquet ← CORPUS PRINCIPAL│
│                                                                 │
│  enrichment/ (Wikipedia PT → LLM extrai JSON estruturado)       │
│         ↓                                                       │
│  scripts/export_entities.py                                     │
│         ↓                                                       │
│  data/derived/final/br_{artist,album,genre,composer}s.parquet   │
│                                                                 │
│  Componentes auxiliares:                                        │
│  _async_http/  — http async com cache+throttle+circuit breaker  │
│  llm/          — cliente NIM (intent / rerank / extract JSON)   │
└─────────────────────────────────────────────────────────────────┘
```

## Setup inicial (uma vez)

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12+.

```bash
# 1. Dependências Python (todos os extras: vector + lyrics)
uv sync --all-groups --extra vector --extra lyrics

# 2. Recursos do NLTK (uma vez por máquina)
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('rslp')"

# 3. Frontend (se for usar a UI web)
cd frontend && npm install && cd ..
```

Os datasets finais em `data/derived/final/` vão versionados no repo, então **não é preciso baixar o Spotify bruto** para usar a busca esparsa, a busca vetorial ou a API.

### Variáveis de ambiente

Use `.env.example` como template local. O projeto lê variáveis do ambiente; se você usar um arquivo `.env`, carregue-o no terminal/VSCode antes de executar os comandos. O projeto não define endpoint de LLM por padrão; configure explicitamente uma API OpenAI-compatível quando for rodar enrichment, classificação por LLM ou rerank.

```powershell
$env:NIM_API_KEY="sua-chave"
$env:NIM_BASE_URL="https://seu-endpoint-openai-compativel/v1"
```

## Modos de uso

### A) Rodar a busca esparsa (BM25 / TF-IDF)

```bash
# CLI
uv run python -m music_search.search "amor saudade" --algorithm both --top 5

# GUI Tkinter (compara BM25 x TF-IDF lado a lado)
uv run python -m music_search.ui_tk
```

> Da primeira vez constrói o índice automaticamente em `data/indexes/br_curated_lyrics.pkl` (~10 s). Para forçar reconstrução: `--rebuild-index`.

### B) Rodar a busca vetorial

Precisa de um backend de embedding:
- **Ollama local** (padrão): `ollama pull nomic-embed-text && ollama serve`
- **OpenAI**: `export OPENAI_API_KEY=... && export USE_OLLAMA=false`

```bash
# 1. Indexa (gera embeddings e popula Milvus Lite em data/vector/)
uv run python -m music_search.vector.indexing
# Smoke test: INDEX_LIMIT=1000 uv run python -m music_search.vector.indexing

# 2. Busca via CLI
uv run python -m music_search.vector.search "rock clássico anos 70" --top 5

# 3. GUI Tk para inspeção
uv run python -m music_search.vector.ui_tk
```

### C) Rodar a aplicação web completa (API + Frontend)

```bash
# Terminal 1 — API FastAPI
uv run uvicorn music_search.web.app:app --reload --port 8000

# Terminal 2 — Frontend Vite
cd frontend && npm run dev
```

Abrir <http://localhost:5173>. O proxy do Vite redireciona `/api/*` → `http://127.0.0.1:8000/*`.

Endpoints da API:
- `GET /healthz` — status + contagens de docs
- `GET /search?q=&top=10&algorithm=bm25` — busca roteada por intent (artist / album / genre / lyric / track)
- `GET /search/lyric?q=&top=20` — busca dedicada em letras com snippets numerados
- `GET /artist/{id}` — knowledge panel
- `GET /song/{id}` — letra completa

> **Sobre `multi_index`**: a API tenta carregar parquets de entidades (`data/derived/final/br_{artist,...}s.parquet`) na startup. Se não existirem ainda, a API ainda sobe — só não terá `MultiEntityIndex` populado e o roteamento de intent só vai retornar tracks. Veja a seção **Pipeline de dados** abaixo para gerar.

### D) Rodar testes e checks de qualidade

```bash
uv run pytest                       # todos
uv run pytest tests/test_search.py  # arquivo
uv run ruff check .                 # lint
uv run ruff format .                # format
uv run --extra vector --extra lyrics ty check
```

## Pipeline de dados (offline)

**Quando rodar:** só quando você quiser regenerar os parquets do zero. No dia-a-dia eles já vão versionados em `data/derived/final/`.

### Dataset definitivo versionado

O dataset final fica separado em tabelas Parquet, para evitar repetir dados textuais grandes em cada faixa:

- `data/derived/final/br_curated_tracks.parquet`: tabela principal de 50.000 faixas brasileiras. É determinística e vem do Spotify Metadata original; inclui IDs, artistas, gêneros, álbum, label, popularidades, followers do artista primário, mercados disponíveis, audio features, metadados de arquivo e capas.
- `data/derived/final/br_curated_lyrics.parquet`: corpus de busca com as faixas que já têm letra consolidada.
- `data/derived/final/br_{artist,album,genre,composer}s.parquet`: dimensões enriquecidas por Wikipedia + LLM, quando geradas.
- `data/derived/final/br_dataset_manifest.json`: versão, contagens, tamanho e hash SHA1 dos arquivos.
- `data/derived/final/README.md`: dicionário curto dos arquivos e dos principais grupos de colunas.

Arquivos locais de cache (`*.sqlite`), Spotify bruto (`data/spotify-metadata/`) e exports intermediários continuam ignorados pelo Git.

Para reconstruir a tabela principal de tracks:

```powershell
uv run python scripts/expand_dataset.py --output data/derived/final/br_curated_tracks.parquet
```

Esse passo também puxa os links de capas já existentes no Spotify Metadata. As colunas são `album_image_url`, `album_image_width`, `album_image_height`, além de `primary_artist_image_url`, `primary_artist_image_width`, `primary_artist_image_height`.

Para versionar os datasets finais:

```powershell
git add data/derived/final/README.md data/derived/final/br_*.parquet data/derived/final/br_dataset_manifest.json
```

### Letras (gera `br_curated_lyrics.parquet`)

```bash
# Sanity check (uma faixa)
uv run python -m music_search.lyrics probe "Anitta" "Envolver"

# Baixar (limite opcional)
uv run python -m music_search.lyrics fetch --limit 100 --concurrency 8

# Status do cache
uv run python -m music_search.lyrics stats

# Consolidar tracks + letras em parquet
uv run python scripts/build_curated_corpus.py
```

Variáveis opcionais: `VAGALUME_API_KEY`, `GENIUS_TOKEN`. Sem elas, ainda funciona via letras.mus.br + lyrics.ovh.

### Enriquecimento (Wikipedia + LLM → entidades)

Requer uma LLM exposta por API OpenAI-compatível. Configure `NIM_API_KEY` e `NIM_BASE_URL` no seu ambiente local; use `.env.example` como referência de nomes de variáveis.

A LLM entra somente para estruturar texto não estruturado. O fluxo é:

```text
br_curated_tracks.parquet
  → seeds de artistas/álbuns/gêneros/compositores
  → WikipediaPTSource baixa HTML
  → llm.tasks.extract_*_json transforma HTML em JSON
  → data/derived/enrichment_cache.sqlite
  → scripts/export_entities.py
  → br_artists.parquet / br_albums.parquet / br_genres.parquet / br_composers.parquet
```

Use LLM para biografia, origem, descrição, obras e relações entre entidades. Não use LLM para campos que já vêm estruturados do Spotify, como popularidade, followers, audio features, datas, label, mercados e capas.

```bash
# Buscar HTML da Wikipedia + extrair JSON via LLM (1 entidade por vez)
uv run python -m music_search.enrichment artists --limit 500 --concurrency 4
uv run python -m music_search.enrichment albums  --limit 500 --concurrency 4
uv run python -m music_search.enrichment genres --concurrency 4
uv run python -m music_search.enrichment composers --limit 500 --concurrency 4

# Cache → parquets
uv run python scripts/export_entities.py

# Manifest final, sem reprocessar letras
uv run python scripts/build_dataset.py --skip-lyrics
```

### Spotify raw (apenas se for re-curar)

```bash
./scripts/download_spotify_metadata.sh --truncated   # ~344 MB (default)
./scripts/download_spotify_metadata.sh --full        # ~5.5 GB via Kaggle
```

## Estrutura

```
src/music_search/
├── preprocessing.py    # Tokenização, stemming
├── indexer.py          # Índice invertido multi-campo
├── ranking.py          # BM25, TF-IDF
├── evaluation.py       # ⚠️ stub: métricas de RI ainda a implementar
├── datasets.py         # Loaders Spotify + corpus curado
├── search.py           # SparseSearchEngine (motor esparso de tracks) + CLI
├── multi_index.py      # MultiEntityIndex (tracks + artist/album/genre/composer)
├── ui_tk.py            # GUI Tk: compara BM25 x TF-IDF
│
├── vector/             # Busca densa (embeddings + Milvus)
├── lyrics/             # Pipeline de extração de letras
├── enrichment/         # Wikipedia → LLM → entidades estruturadas
├── llm/                # Cliente NIM (intent / rerank / extract JSON)
├── _async_http/        # http async reusável (cache + throttle + circuit breaker)
└── web/
    ├── app.py          # API FastAPI
    ├── schemas.py      # Pydantic
    └── snippets.py     # Extração de trechos de letra com highlight

frontend/               # SPA React + Vite (consome /api/*)
notebooks/              # 4 notebooks: EDA + curadoria do dataset BR
scripts/                # build_curated_corpus, build_index, export_entities, ...
data/derived/final/     # Datasets finais versionados + README do dataset
data/derived/           # Caches e intermediarios locais (gitignored)
data/spotify-metadata/  # Spotify raw (gitignored, baixado on-demand)
```

## Atalhos no VSCode

`launch.json` tem configs prontas (F5 → escolher):

- **Sparse: CLI search** / **Sparse: GUI Tk**
- **Vector: CLI search** / **Vector: indexar** / **Vector: GUI Tk**
- **Web: API (uvicorn reload)**
- **Full-stack** (compound: API + Vite juntos)
- **Lyrics: probe / fetch / stats**
- **Enrichment: artists / albums**
- **Build: corpus / index / dataset completo**
- **Pytest: arquivo atual** / **Pytest: todos**

`tasks.json` tem `npm: dev` / `npm: build` para o frontend e `uv: sync` para deps.

## Status — feito vs falta

### ✅ Pronto

- Pipeline RI clássico: preprocessing, índice invertido multi-campo, BM25, TF-IDF
- Motor esparso multi-campo com pesos configuráveis (`SparseSearchEngine`)
- Busca vetorial com Ollama/OpenAI + Milvus Lite
- GUI Tk comparativa (sparse e vector)
- Dataset curado de **50.000 faixas brasileiras** com metadados enriquecidos e **36.017 músicas com letra** versionadas
- Pipeline de letras com cache, retries, circuit breaker e cascata de fontes
- API FastAPI com endpoints `/search`, `/search/lyric`, `/artist`, `/song`
- Frontend React+Vite com painéis de artista/música/lyric matches
- Suíte de testes (preprocessing, indexer, ranking, search, vector, lyrics)

### 🟡 Em andamento

- **Enrichment de entidades** (`enrichment/` + `llm/`): pipeline funciona, mas os parquets `br_{artist,album,genre,composer}s.parquet` ainda não estão gerados/commitados. A API roda sem eles, mas `/artist/{id}` cai num fallback derivado dos tracks.
- **MultiEntityIndex** depende dos parquets acima; está integrado mas só populado parcialmente.
- **Frontend**: estrutura e rotas montadas; alguns painéis mockados aguardam dados de enrichment.

### ❌ Falta

- **`evaluation.py`** está vazio (só docstring). Precisa implementar Precision, Recall, MAP e nDCG e construir um conjunto de queries com julgamento (golden set) para comparar BM25 × TF-IDF × vetorial.
- **Reranking por LLM**: `web/app.py` já tem o gancho (`?rerank=true`), mas só ativa se `NIM_API_KEY` estiver setada. Falta avaliar impacto.
- **CI**: não há workflow ainda no `.github/workflows/` para rodar lint+ty+pytest em PRs (havia um antes, conferir se ainda está ativo).

## Equipe

- [Carsio Eddyo](https://github.com/carsio)
- [Carlos Alexandre](https://github.com/alexandrecarloss)
- [Raquel de Sá](https://github.com/raqueldesa)
- [Lelson Nascimento](https://github.com/lelsonln)

## Licença

MIT
