# Music Search Engine

**Um Estudo de Técnicas de Indexação e Ranking em Busca de Músicas**

Trabalho da disciplina ICC222 — Tópicos em Recuperação de Informação (UFAM 2026/1).

📊 **Slides da apresentação:** https://carsio.github.io/music-search-engine/

## Sobre

Sistema de busca de músicas que implementa e compara diferentes técnicas de indexação e ranking textual, utilizando o dataset [Spotify Metadata](https://www.kaggle.com/datasets/lordpatil/spotify-metadata-by-annas-archive) como base de dados.

### Técnicas implementadas

- **Indexação:** Índice invertido com suporte a diferentes esquemas de pesos
- **Ranking esparso:** TF-IDF, BM25
- **Ranking denso:** Embeddings + similaridade de cosseno em Milvus (opcional — ver abaixo)
- **Avaliação:** Precision, Recall, MAP, nDCG
- **Interface:** Aplicação web com FastAPI

## Setup

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12+.

```bash
# Instalar dependências
uv sync

# Baixar dados do NLTK (primeira vez)
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# Baixar o dataset truncado (padrão, ~344 MB)
./scripts/download_spotify_metadata.sh --truncated

# Ou o dataset completo via Kaggle (~5.5 GB)
./scripts/download_spotify_metadata.sh --full

# Rodar testes
uv run pytest

# Iniciar servidor web
uv run uvicorn music_search.web.app:app --reload

# Lint e formatação
uv run ruff check .
uv run ruff format .

# Type checking
uv run ty check
```

## Dados

Código e notebooks **assumem** que `data/spotify-metadata/` já está populado com os parquets do
dataset — nenhum download implícito. O diretório `data/` em si é versionado, mas o conteúdo
(`data/*`) fica fora do git.

Há dois modos de bootstrap, com o mesmo layout final:

- **Truncado** (padrão, ~344 MB): subset empacotado como asset da release `v0.1-data` deste
  repositório. Rápido o suficiente para iterar local.
- **Full** (~5.5 GB): dataset completo via Kaggle CLI.

```bash
# Truncado (padrão)
./scripts/download_spotify_metadata.sh --truncated

# Full via Kaggle
./scripts/download_spotify_metadata.sh --full
```

Troca entre os modos é transparente: o layout final é sempre
`data/spotify-metadata/spotify_clean_parquet/*.parquet` + audio features. Notebooks e código
de indexação não mudam.

### Pré-requisitos por modo

**Truncado**: precisa de `gh` autenticado *ou* `curl`. Se você já tiver o zip em
`data/spotify-metadata-by-annas-archive-truncated-300mb.zip`, o script usa ele direto e pula o
download.

**Full**: precisa da credencial do Kaggle em `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

### Dataset fora do repositório

Se quiser armazenar os arquivos extraídos fora do repo, passe um caminho posicional. O script
recria `data/spotify-metadata` como symlink:

```bash
./scripts/download_spotify_metadata.sh --truncated /caminho/para/datasets
./scripts/download_spotify_metadata.sh --full /caminho/para/datasets/spotify-metadata
```

## Busca vetorial (opcional)

Modelo denso complementar ao BM25/TF-IDF: converte cada track em um embedding e
recupera por similaridade de cosseno no [Milvus](https://milvus.io/). Útil para
queries semânticas (`"música animada para treinar"`, `"rock clássico anos 70"`)
onde os termos literais raramente aparecem no título da faixa.

### Pré-requisitos

Um dos dois backends de embedding:

- **Ollama local** (padrão, sem custo): `ollama pull nomic-embed-text` e
  `ollama serve` rodando.
- **OpenAI API**: variável `OPENAI_API_KEY` (modelo `text-embedding-3-small`).

### Instalação

```bash
# Dependências opcionais (pymilvus, openai, tqdm)
uv sync --all-groups --extra vector
```

### Pipeline

```bash
# 1. Gera embeddings de cada track e popula o Milvus.
#    Artefatos (Milvus Lite .db, checkpoint, log) vão para data/vector/.
uv run python -m music_search.vector.indexing

# Smoke test (limita número de tracks indexadas):
INDEX_LIMIT=1000 uv run python -m music_search.vector.indexing

# 2. Busca semântica via CLI:
uv run python -m music_search.vector.search "rock clássico anos 70" --top 5

# 3. Ou, como biblioteca:
uv run python -c "from music_search.vector import search_tracks; \
    print(search_tracks('música animada para treinar', top_k=5))"

# 4. (Opcional) UI Tk para inspeção interativa — ferramenta de debug local:
uv run python -m music_search.vector.ui_tk
```

### Variáveis de ambiente

| Variável         | Padrão                            | Descrição                                        |
|------------------|-----------------------------------|--------------------------------------------------|
| `USE_OLLAMA`     | `true`                            | `false` para usar OpenAI                         |
| `OLLAMA_URL`     | `http://localhost:11434/v1`       | Endpoint OpenAI-compatível do Ollama             |
| `EMBED_MODEL`    | `nomic-embed-text`                | Modelo de embedding do Ollama                    |
| `OPENAI_API_KEY` | —                                 | Chave da OpenAI (necessária se `USE_OLLAMA=false`) |
| `MILVUS_URI`     | `./data/vector/milvus_spotify.db` | URI do Milvus (Lite local ou servidor remoto)    |
| `INDEX_LIMIT`    | —                                 | Limita número de tracks (apenas na indexação)    |

**Importante**: use o mesmo modelo para indexar e buscar. `nomic-embed-text`
gera vetores de 768 dim; `text-embedding-3-small`, 1536 dim. Misturar os dois
na mesma coleção quebra a busca.

## Extração de letras (opcional)

Pipeline para enriquecer cada faixa do dataset curado (`notebooks/04_dataset_curado_brasileiro.ipynb` → `data/derived/br_curated_tracks.parquet`) com a letra correspondente, usado depois pela indexação BM25/TF-IDF/vetorial.

### Características

- **Cascata de fontes**: tenta letras.mus.br → Vagalume (se houver chave) → lyrics.ovh → Genius (se houver token). Para na primeira que devolve uma letra.
- **Cache SQLite persistente** (`data/derived/lyrics_cache.sqlite`): execução é **idempotente** — re-rodar não reprocessa faixas resolvidas.
- **Async com httpx**: concorrência configurável (`--concurrency`), retries com backoff exponencial e jitter, semáforo global.
- **Normalização de query**: remove `feat. X`, `(Ao Vivo)`, `- Remix`, `[Slowed]` etc. antes de bater na API.
- **Status terminais separados**: `hit | miss | error | blocked`. `error` é re-tentável via `--retry-errors`.
- **Anti-blocking** (ver `throttle.py` e `user_agents.py`):
  - **Token bucket assíncrono** por fonte (letras.mus.br 1 rps, lyrics.ovh 5 rps, Vagalume 2 rps, Genius 1 rps).
  - **Honor `Retry-After`**: na resposta 429/503 o limiter é penalizado pelo período pedido pelo servidor.
  - **Circuit breaker**: após N falhas consecutivas (3 para scrapers HTML, 5 para APIs), a fonte é desligada por 60–120s e o pipeline cascateia para a próxima.
  - **Pool de User-Agents** rotacionado para scraping de HTML (letras.mus.br/Genius), com headers realistas (`Accept`, `Accept-Language`, `Sec-Fetch-*`).
  - `BLOCKED` agora significa "fonte exausta — pula"; só `ERROR` (transitório) é retentado.

### Instalação

```bash
uv sync --all-groups --extra lyrics
```

### Variáveis de ambiente

| Variável           | Necessária?                               | Como obter                                           |
|--------------------|-------------------------------------------|------------------------------------------------------|
| `VAGALUME_API_KEY` | Opcional (fallback via API, quando disponível) | Cadastro gratuito em https://auth.vagalume.com.br/   |
| `GENIUS_TOKEN`     | Opcional (fallback de cauda longa)        | Cadastro gratuito em https://genius.com/api-clients  |

`letras.mus.br` e `lyrics.ovh` são livres, sem cadastro, e ficam ligados por padrão.

### Comandos

```bash
# Sanity check em uma faixa avulsa (testa todas as fontes configuradas)
uv run python -m music_search.lyrics probe "Anitta" "Envolver"

# Baixa letras (limite opcional para teste)
uv run python -m music_search.lyrics fetch --limit 100 --concurrency 8

# Interface Tk para baixar batches manualmente
uv run python -m music_search.lyrics.ui_tk

# Run completo (22k faixas)
uv run python -m music_search.lyrics fetch

# Status do cache
uv run python -m music_search.lyrics stats

# Onde as letras ficam salvas e como abrir no SQLite
uv run python -m music_search.lyrics where

# Ver amostras recentes ou aleatórias do cache
uv run python -m music_search.lyrics sample -n 10 --status hit
uv run python -m music_search.lyrics sample -n 10 --random

# Mostrar a letra completa de uma faixa pelo track_id
uv run python -m music_search.lyrics show <track_id>

# Reprocessar só as faixas que terminaram em erro
uv run python -m music_search.lyrics fetch --retry-errors

# Exportar hits para parquet (pronto para indexação)
uv run python -m music_search.lyrics export
```

### Estrutura do módulo

```
src/music_search/lyrics/
├── cli.py                # subcomandos: fetch, stats, export, probe
├── ui_tk.py              # interface Tk para baixar batches manualmente
├── pipeline.py           # orquestrador async + cache + retries
├── cache.py              # SQLite com WAL e upsert idempotente
├── normalize.py          # limpeza de título/artista
├── throttle.py           # token bucket + circuit breaker + Retry-After
├── user_agents.py        # pool de UAs realistas para HTML scraping
└── sources/
    ├── base.py           # protocolo LyricsSource + Status
    ├── letras_mus_br.py  # scraping HTML do letras.mus.br (sem chave)
    ├── lyrics_ovh.py     # API pública gratuita
    ├── vagalume.py       # API com chave gratuita (foco BR)
    └── genius.py         # API + scraping de HTML (com UA pool)
```

## Estrutura do projeto

```
src/music_search/
├── __init__.py
├── preprocessing.py    # Tokenização, stemming, normalização
├── indexer.py          # Construção de índices invertidos
├── ranking.py          # Modelos de ranking esparsos (TF-IDF, BM25)
├── search.py           # Motor de busca / query processing
├── evaluation.py       # Métricas de avaliação de RI
├── datasets.py         # ETL dos parquets do Spotify
├── vector/             # Busca vetorial (opcional, extra `vector`)
│   ├── __init__.py
│   ├── config.py       # EmbeddingConfig + paths
│   ├── indexing.py     # Pipeline de embeddings → Milvus
│   ├── search.py       # Cliente de busca semântica + CLI
│   └── ui_tk.py        # UI Tk de debug (opcional)
└── web/
    ├── __init__.py
    └── app.py          # Interface web (FastAPI)
```

## Equipe

- [Carsio Eddyo](https://github.com/carsio)
- [Carlos Alexandre](https://github.com/alexandrecarloss)
- [Raquel de Sá](https://github.com/raqueldesa)
- [Lelson Nascimento](https://github.com/lelsonln)

## Licença

MIT
