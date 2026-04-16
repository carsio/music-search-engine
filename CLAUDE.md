# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sobre o projeto

Sistema de busca de músicas que implementa e compara técnicas de indexação e ranking textual (índice invertido, TF-IDF, BM25) com métricas de avaliação de RI (Precision, Recall, MAP, nDCG). Trabalho da disciplina ICC222 — Tópicos em Recuperação de Informação (UFAM 2026/1). Dataset: Spotify Metadata (Kaggle), exposto localmente em `data/spotify-metadata`.

## Comandos

```bash
# Instalar dependências
uv sync

# Testes
uv run pytest                        # todos os testes
uv run pytest tests/test_foo.py      # arquivo específico
uv run pytest -k "test_name"         # teste por nome

# Lint e formatação
uv run ruff check .                  # lint
uv run ruff check . --fix            # lint com autofix
uv run ruff format .                 # formatação

# Type checking
uv run mypy src/

# Servidor web (dev)
uv run uvicorn music_search.web.app:app --reload

# Setup inicial (NLTK, só precisa rodar uma vez)
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# CLI do Kaggle no contexto do projeto
uv run kaggle --version

# Download do dataset em data/spotify-metadata
./scripts/download_spotify_metadata.sh

# Download em outro caminho e recriacao do symlink local
./scripts/download_spotify_metadata.sh /caminho/para/datasets
```

## Arquitetura

O pacote principal é `src/music_search/` (importado como `music_search`). O fluxo de dados segue o pipeline clássico de RI:

1. **preprocessing** — tokenização, stemming, normalização de texto
2. **indexer** — construção de índices invertidos a partir dos documentos
3. **ranking** — scoring de documentos (TF-IDF, BM25) dado um índice
4. **search** — processamento de queries, orquestra preprocessing + indexer + ranking
5. **evaluation** — métricas de avaliação (Precision, Recall, MAP, nDCG)
6. **web/app** — interface web FastAPI que expõe o motor de busca

## Convenções

- Python 3.12+, gerenciado com `uv`
- Ruff para lint e formatação (line-length=100, regras: E, W, F, I, UP, B, SIM, RUF)
- mypy em modo strict
- Testes em `tests/` com pytest (flags: `-v --tb=short`)
- Notebooks exploratórios em `notebooks/`
- Todo código em português (docstrings, variáveis de domínio), APIs e nomes técnicos em inglês
- `data/` deve continuar versionado; datasets locais ficam em subpastas, como `data/spotify-metadata`
- Scripts de download devem preservar o layout `data/<dataset>`; symlinks são opcionais quando o usuário quiser armazenar dados fora do repositório
