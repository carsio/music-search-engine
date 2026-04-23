# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sobre o projeto

Sistema de busca de músicas que implementa e compara técnicas de indexação e ranking textual (índice invertido, TF-IDF, BM25) e vetorial (embeddings + similaridade de cosseno em Milvus), com métricas de avaliação de RI (Precision, Recall, MAP, nDCG). Trabalho da disciplina ICC222 — Tópicos em Recuperação de Informação (UFAM 2026/1). Dataset: Spotify Metadata (Kaggle), exposto localmente em `data/spotify-metadata`.

## Setup após clonar

Assuma que o usuário acabou de rodar `git clone` e está no diretório raiz do projeto. Ordem:

```bash
# 1. uv (se ainda não tiver): https://docs.astral.sh/uv/
#    macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sincronizar dependências (runtime + dev + notebooks)
uv sync --all-groups

# 2b. (Opcional) Dependências da busca vetorial (pymilvus, openai, tqdm)
#     Só necessário se for rodar o pipeline em src/music_search/vector/.
# uv sync --all-groups --extra vector

# 3. Dados do NLTK (só na primeira vez da máquina)
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('rslp')"

# 4. Baixar e extrair o dataset truncado (~344 MB, padrão)
./scripts/download_spotify_metadata.sh --truncated
#    → precisa de 'gh' autenticado OU 'curl'.
#    → se já houver data/spotify-metadata-by-annas-archive-truncated-300mb.zip
#      localmente, o script pula o download e só extrai (fast path).
#    → para o dataset completo via Kaggle: --full, precisa de ~/.kaggle/kaggle.json.

# 5. Sanidade: lint, tipos e testes devem passar
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

Depois disso o notebook `notebooks/01_eda_spotify.ipynb` roda fim-a-fim e o servidor web sobe com `uv run uvicorn music_search.web.app:app --reload`.

Se o usuário não tiver `gh`, instalar via `brew install gh` (macOS) e `gh auth login`; ou simplesmente ter `curl` no PATH (fallback automático do script).

## Comandos

```bash
# Instalar dependências (runtime + dev + notebooks)
uv sync --all-groups

# Testes
uv run pytest                        # todos os testes
uv run pytest tests/test_foo.py      # arquivo específico
uv run pytest -k "test_name"         # teste por nome

# Lint e formatação
uv run ruff check .                  # lint
uv run ruff check . --fix            # lint com autofix
uv run ruff format .                 # formatação

# Type checking
uv run ty check

# Servidor web (dev)
uv run uvicorn music_search.web.app:app --reload

# Setup inicial (NLTK, só precisa rodar uma vez)
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('rslp')"

# Dataset truncado (padrão, ~344 MB, via GitHub release v0.1-data)
./scripts/download_spotify_metadata.sh --truncated

# Dataset completo (~5.5 GB, via Kaggle CLI)
./scripts/download_spotify_metadata.sh --full

# Destino alternativo (recria symlink em data/spotify-metadata)
./scripts/download_spotify_metadata.sh --truncated /caminho/para/datasets
```

## Arquitetura

O pacote principal é `src/music_search/` (importado como `music_search`). O fluxo de dados segue o pipeline clássico de RI:

1. **preprocessing** — tokenização, stemming, normalização de texto
2. **indexer** — construção de índices invertidos a partir dos documentos
3. **ranking** — scoring de documentos (TF-IDF, BM25) dado um índice
4. **search** — processamento de queries, orquestra preprocessing + indexer + ranking
5. **evaluation** — métricas de avaliação (Precision, Recall, MAP, nDCG)
6. **web/app** — interface web FastAPI que expõe o motor de busca
7. **vector** (opcional, extra `vector`) — pipeline denso: embeddings + Milvus. `vector.indexing` gera os vetores a partir de `SpotifyTracksLoader.iter_rich_docs()`; `vector.search.VectorSearch` responde queries por similaridade de cosseno. Artefatos em `data/vector/`.

## Convenções

- Python 3.12+, gerenciado com `uv`
- Ruff para lint e formatação (line-length=100, regras: E, W, F, I, UP, B, SIM, RUF)
- ty para checagem de tipos
- Testes em `tests/` com pytest (flags: `-v --tb=short`)
- Notebooks exploratórios em `notebooks/`
- Todo código em português (docstrings, variáveis de domínio), APIs e nomes técnicos em inglês
- `data/` é versionado mas `data/*` é gitignored; datasets locais ficam em `data/spotify-metadata/`
- Código e notebooks **assumem que os parquets já estão em `data/spotify-metadata/`**. O download é função do script de apoio, não do pipeline.
- Dois modos de bootstrap (`--truncated` default / `--full`) produzem o mesmo layout final — troca é transparente.
- Scripts de download devem preservar o layout `data/<dataset>`; symlinks são opcionais quando o usuário quiser armazenar dados fora do repositório
