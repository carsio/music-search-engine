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
