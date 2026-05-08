# Music Search Engine

**Busca de músicas brasileiras com BM25, TF-IDF e embeddings vetoriais.**

Trabalho da disciplina ICC222 — Tópicos em Recuperação de Informação (UFAM 2026/1).

📊 Slides: <https://carsio.github.io/music-search-engine/>

> 📖 Para entender o sistema do começo ao fim — conceitos de RI, pipeline de
> dados, BM25/TF-IDF (com fórmulas), motores, intent, LLM e o trajeto de uma
> query — comece por **[`docs/GUIA.md`](docs/GUIA.md)**. Este README é
> referência rápida de comandos.

## Visão geral

O projeto implementa os algoritmos clássicos de RI (índice invertido, TF-IDF, BM25) e a busca vetorial densa (embeddings + Milvus), e expõe esse motor através de uma API FastAPI consumida por um frontend React. Os dados foram curados a partir do Spotify Metadata e enriquecidos com letras (lyrics.ovh, Vagalume, letras.mus.br, Genius) e conteúdo da Wikipedia PT materializado de forma determinística para as entidades.

O fluxo principal hoje é: a API em `music_search.web.app` recebe a query, classifica a intent via heurística ou LLM opcional, delega o roteamento para `multi_index.MultiEntityIndex`, consulta o índice esparso ou a dimensão apropriada e devolve a resposta para a SPA em `frontend/`.

Snapshot atual do dataset versionado (`data/derived/final/br_dataset_manifest.json`, versão `0.3.0`, gerado em `2026-05-08`):

- `br_curated_tracks.parquet`: 50.000 faixas brasileiras.
- `br_curated_lyrics.parquet`: 36.017 faixas com letra consolidada.
- `br_artists.parquet`: 7.255 artistas enriquecidos.
- `br_genres.parquet`: 42 gêneros enriquecidos.
- `br_albums.parquet` e `br_composers.parquet`: ainda opcionais no snapshot atual.

## Como o projeto está organizado

```text
┌─────────────────────────────────────────────────────────────────┐
│  Camada 1 — Core de RI (algoritmos do trabalho)  [core/]        │
│  preprocessing → indexer → ranking (BM25, TF-IDF) → evaluation  │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│  Camada 2 — Motores de busca  [motors/, data/, vector/]         │
│  motors.search.SparseSearchEngine   (multi-campo sobre tracks)  │
│  motors.multi_index.MultiEntityIndex (track/artist/album/...)   │
│  data.albums              (catálogo derivado para /album/{id})  │
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
│  python -m music_search.scripts.build_curated_corpus            │
│         ↓                                                       │
│  data/derived/final/br_curated_lyrics.parquet ← CORPUS PRINCIPAL│
│                                                                 │
│  enrichment/ (Wikipedia PT text/API → materializacao local)     │
│         ↓                                                       │
│  python -m music_search.scripts.export_entities                 │
│         ↓                                                       │
│  data/derived/final/br_{artist,album,genre,composer}s.parquet   │
│                                                                 │
│  Componentes auxiliares:                                        │
│  _async_http/  — http async com cache+throttle+circuit breaker  │
│  llm/          — cliente NIM opcional (intent / rerank)         │
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

Use `.env.example` como template local. O projeto lê variáveis do ambiente; se você usar um arquivo `.env`, carregue-o no terminal/VSCode antes de executar os comandos. O projeto não define endpoint de LLM por padrão; configure explicitamente uma API OpenAI-compatível apenas quando for rodar classificação por LLM ou rerank.

```powershell
$env:NIM_API_KEY="sua-chave"
$env:NIM_BASE_URL="https://seu-endpoint-openai-compativel/v1"
$env:NIM_RATE="1.0"          # reduza para 0.5 se receber muitos 429
$env:NIM_MAX_RETRIES="4"     # retries automaticos no cliente NIM
```

## Modos de uso

### A) Rodar a busca esparsa (BM25 / TF-IDF)

```bash
# CLI
uv run python -m music_search.motors.search "amor saudade" --algorithm both --top 5

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

Abrir <http://localhost:5173>. O proxy do Vite encaminha `/api/*` → `http://127.0.0.1:8000/api/*`.

Endpoints da API:

- `GET /healthz` — status + contagens de docs (healthcheck do serviço)
- `GET /api/healthz` — mesmo payload de saúde para clientes HTTP
- `GET /api/search?q=&top=10&algorithm=bm25` — busca roteada por intent (artist / album / genre / lyric / track)
- `GET /api/search/lyric?q=&top=20` — busca dedicada em letras com snippets numerados
- `GET /api/artist/{id}` — knowledge panel
- `GET /api/album/{id}` — página de álbum derivada do dataset curado
- `GET /api/song/{id}` — letra completa

> **Sobre `multi_index`**: a API tenta carregar os parquets de entidades (`data/derived/final/br_{artist,...}s.parquet`) na startup. No snapshot versionado atual, `br_artists.parquet` (7.255 docs) e `br_genres.parquet` (42 docs) já carregam normalmente; `br_albums.parquet` e `br_composers.parquet` seguem opcionais. Mesmo sem alguma dimensão, a API continua subindo com fallback gracioso para tracks e catálogos derivados.

### D) Rodar com Docker Compose

```bash
docker compose up --build
```

Abrir <http://localhost:8000>. Nesse modo o frontend React é buildado dentro da imagem e servido pelo mesmo processo FastAPI; a API fica em `/api/*` e o volume nomeado `search-indexes` persiste `data/indexes/` entre reinícios.

Para parar o stack:

```bash
docker compose down
```

Se quiser limpar também o índice persistido:

```bash
docker compose down -v
```

### E) Rodar testes e checks de qualidade

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

O dataset final fica separado em tabelas Parquet para evitar repetir dados textuais grandes em cada faixa. Snapshot atual do manifesto:

| Artefato | Registros | Situação atual | Uso |
| --- | ---: | --- | --- |
| `data/derived/final/br_curated_tracks.parquet` | 50.000 | versionado | tabela principal de faixas brasileiras, com metadados estruturados do Spotify |
| `data/derived/final/br_curated_lyrics.parquet` | 36.017 | versionado | corpus textual usado na indexação BM25/TF-IDF |
| `data/derived/final/br_artists.parquet` | 7.255 | versionado | dimensão enriquecida de artistas via Wikipedia PT |
| `data/derived/final/br_genres.parquet` | 42 | versionado | dimensão enriquecida de gêneros |
| `data/derived/final/br_albums.parquet` | 0 | ainda não gerado no snapshot atual | dimensão enriquecida de álbuns |
| `data/derived/final/br_composers.parquet` | 0 | ainda não gerado no snapshot atual | dimensão enriquecida de compositores/letristas |
| `data/derived/final/br_dataset_manifest.json` | - | versionado | versão, contagens, tamanho e hash SHA1 dos arquivos |
| `data/derived/final/README.md` | - | versionado | dicionário curto dos arquivos e dos principais grupos de colunas |

Arquivos locais de cache (`*.sqlite`), Spotify bruto (`data/spotify-metadata/`) e exports intermediários continuam ignorados pelo Git.

Para reconstruir a tabela principal de tracks:

```powershell
uv run python -m music_search.scripts.expand_dataset --output data/derived/final/br_curated_tracks.parquet
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
uv run python -m music_search.scripts.build_curated_corpus
```

Variáveis opcionais: `VAGALUME_API_KEY`, `GENIUS_TOKEN`. Sem elas, ainda funciona via letras.mus.br + lyrics.ovh.

### Enriquecimento (Wikipedia → entidades)

O enrichment offline nao requer LLM. O fluxo usa apenas seeds do corpus curado, coleta texto da Wikipedia PT e materializa um payload minimo e deterministico por entidade. O fluxo e:

```text
br_curated_tracks.parquet
  → seeds de artistas/álbuns/gêneros/compositores
  → WikipediaPTSource baixa texto (Wikipedia-API; fallback HTML)
  → materializacao local transforma texto em payload pesquisavel
  → data/derived/enrichment_cache.sqlite
  → python -m music_search.scripts.export_entities
  → br_artists.parquet / br_albums.parquet / br_genres.parquet / br_composers.parquet
```

No snapshot atual do repositório, `br_artists.parquet` e `br_genres.parquet` já estão versionados. `br_albums.parquet` e `br_composers.parquet` continuam sendo os próximos exports naturais desse pipeline.

Os parquets resultantes preservam `source`, `source_url` e texto bruto limpo (`raw_text`) para indexacao leve. Campos ja estruturados do Spotify, como popularidade, followers, audio features, datas, label, mercados e capas, continuam fora desse enrichment e seguem vindo do pipeline deterministico principal.

Para `genres`, o default agora usa seeds detalhadas derivadas de `artist_genres`; use `--seed-mode macro` para voltar à taxonomia curada de macro-gêneros.

```bash
# Buscar conteudo da Wikipedia e materializar payload local (1 entidade por vez)
uv run python -m music_search.enrichment artists --limit 500 --concurrency 4
uv run python -m music_search.enrichment albums  --limit 500 --concurrency 4
uv run python -m music_search.enrichment genres --concurrency 4
uv run python -m music_search.enrichment genres --concurrency 4 --seed-mode macro
uv run python -m music_search.enrichment composers --limit 500 --concurrency 4

# Cache → parquets
uv run python -m music_search.scripts.export_entities

# Manifest final, sem reprocessar letras
uv run python -m music_search.scripts.build_dataset --skip-lyrics

# UI Tk para rodar o mesmo fluxo em batches manuais
uv run python -m music_search.enrichment.ui_tk
```

A UI exibe andamento por etapa (enrichment/export/build), cache com faltantes/erros, logs em
tempo real (com painel maior) e
painel de artefatos gerados (`br_*.parquet` e `br_dataset_manifest.json`).

### Spotify raw (apenas se for re-curar)

```bash
./src/music_search/scripts/download_spotify_metadata.sh --truncated   # ~344 MB (default)
./src/music_search/scripts/download_spotify_metadata.sh --full        # ~5.5 GB via Kaggle
```

## Estrutura do repositório

### Núcleo de RI (`src/music_search/core/`)

- `preprocessing.py`: tokenização, normalização, stopwords e stemming.
- `indexer.py`: índice invertido multi-campo e estruturas auxiliares.
- `ranking.py`: scoring compartilhado entre BM25 e TF-IDF.
- `evaluation.py`: reservado para métricas de RI e golden set.

### Motores de busca (`src/music_search/motors/`)

- `search.py`: `SparseSearchEngine`, CLI e carga do índice principal.
- `multi_index.py`: busca roteada entre tracks, artists, albums, genres e composers.
- `tuning.py`: perfis `balanced`, `lyrics` e `metadata` usados pelos motores.

### Datasets e loaders (`src/music_search/data/`)

- `datasets.py`: loaders dos parquets curados e dos dados-base do projeto.
- `albums.py`: catálogo de álbuns derivado de `br_curated_tracks.parquet` para a rota `/album/{id}`.

### Pipelines de build (`src/music_search/scripts/`)

- `build_curated_corpus.py`, `build_dataset.py`, `build_index.py`,
  `build_entity_indexes.py`, `expand_dataset.py`, `export_entities.py`,
  `download_spotify_metadata.sh`. Todos rodáveis via
  `python -m music_search.scripts.<nome>`.

### Serviços Python e interfaces (`src/music_search/`)

- `web/`: FastAPI (`app.py`), schemas Pydantic e extração de snippets.
- `vector/`: indexação de embeddings, busca vetorial e UI Tk dedicada.
- `lyrics/`: coleta de letras, estatísticas e consolidação do corpus.
- `enrichment/`: seeds, coleta da Wikipedia PT, materialização local e UI Tk do pipeline.
- `llm/`: cliente NIM e tarefas opcionais de intent/rerank.
- `_async_http/`: cache SQLite, throttle, retries e circuit breaker compartilhados.
- `ui_tk.py`: interface desktop para comparar BM25 x TF-IDF.

### Frontend (`frontend/src/`)

- `api/`: cliente HTTP, tipos e contratos consumidos da API.
- `hooks/`: fluxo de busca, debounce, query params e preferências da UI.
- `components/`: blocos reutilizáveis organizados em `home`, `layout`, `panels`, `primitives`, `search` e `states`.
- `views/`: composição das telas principais.
- `styles/`: tokens, reset, animações e estilos globais.
- `utils/`: highlight, intent, formatação e helpers gerais.

### Dados, automação e suporte

- `data/derived/final/`: snapshot versionado do dataset final e manifesto.
- `data/derived/`: caches e intermediários locais não versionados.
- `data/spotify-metadata/`: fonte bruta opcional para recurar o dataset.
- `src/music_search/scripts/`: geração offline de tracks, corpus, índices, exports e manifesto.
- `tests/`: suíte por subsistema (`search`, `multi_index`, `web`, `lyrics`, `vector`, etc.).
- `notebooks/`: EDA e curadoria do dataset brasileiro.
- `.vscode/`: atalhos de debug e tasks para backend, frontend e full stack.
- `.github/workflows/`: CI e deploy dos slides.
- `reference-web/`: protótipos e referências de UI.
- `slides/`: apresentação publicada do projeto.
- `docs/`: guia didático completo (`GUIA.md`) e documentação auxiliar.

## Atalhos no VSCode

`launch.json` hoje expõe:

- `Backend · FastAPI (uvicorn)`
- `Frontend · Vite (npm run dev)`
- `Full stack (backend + frontend)`

`tasks.json` hoje expõe:

- `backend: dev`
- `frontend: dev`
- `dev: full stack`

## Status — feito vs falta

### ✅ Pronto

- Pipeline RI clássico: preprocessing, índice invertido multi-campo, BM25, TF-IDF
- Motor esparso multi-campo com pesos configuráveis (`SparseSearchEngine`)
- Roteamento via `MultiEntityIndex` com fallback gracioso quando alguma dimensão ainda não existe
- Busca vetorial com Ollama/OpenAI + Milvus Lite
- GUI Tk comparativa (sparse e vector)
- Catálogo de álbuns derivado do dataset de tracks e endpoint `/album/{id}`
- Dataset versionado com **50.000 faixas brasileiras**, **36.017 músicas com letra**, **7.255 artistas** e **42 gêneros**
- Pipeline de letras com cache, retries, circuit breaker e cascata de fontes
- Pipeline de enrichment determinístico da Wikipedia PT já exportando artistas e gêneros
- API FastAPI com endpoints `/search`, `/search/lyric`, `/artist`, `/album`, `/song`
- Frontend React+Vite com painéis de artista/música/lyric matches
- CI em `.github/workflows/ci.yml` para lint, format check, type check e pytest
- Suíte de testes (preprocessing, indexer, ranking, search, vector, lyrics)

### 🟡 Em andamento

- **Álbuns e compositores enriquecidos**: `br_albums.parquet` e `br_composers.parquet` ainda não entram no snapshot versionado atual.
- **Cobertura do `MultiEntityIndex`**: artist e genre já carregam do manifesto atual, mas album/composer ainda dependem dos exports restantes.
- **Avaliação de qualidade**: rerank por LLM e os perfis de busca ainda precisam de medição comparativa controlada.

### ❌ Falta

- **`evaluation.py`** está vazio (só docstring). Precisa implementar Precision, Recall, MAP e nDCG e construir um conjunto de queries com julgamento (golden set) para comparar BM25 × TF-IDF × vetorial.
- **Benchmarking consolidado**: falta fechar uma comparação reproduzível entre sparse, vetorial e rerank.

## Equipe

- [Carsio Eddyo](https://github.com/carsio)
- [Carlos Alexandre](https://github.com/alexandrecarloss)
- [Raquel de Sá](https://github.com/raqueldesa)
- [Lelson Nascimento](https://github.com/lelsonln)

## Licença

MIT
