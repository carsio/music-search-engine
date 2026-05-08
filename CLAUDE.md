# CLAUDE.md

Guia para o Claude Code trabalhar neste repositório. Para visão geral, comandos completos e status, leia o `README.md` — este arquivo cobre só o que é específico de assistência por IA.

## O que é o projeto

Sistema de busca de músicas brasileiras (BM25, TF-IDF, vetorial) construído sobre um dataset curado de 50k tracks e um corpus de letras consolidado. Trabalho da disciplina ICC222 (UFAM 2026/1). A entrega final é uma webapp (FastAPI + React) que reusa o motor de RI no backend.

## Arquitetura em camadas

1. **Core RI** (`core/preprocessing`, `core/indexer`, `core/ranking`, `core/evaluation`) — algoritmos puros do trabalho.
2. **Motores** (`motors/search`, `motors/multi_index`, `motors/tuning`, `vector/`) — abstrações que aplicam o core sobre o corpus curado.
3. **Datasets** (`data/datasets`, `data/albums`) — loaders e estruturas tipadas dos parquets curados.
4. **Apresentação** (`ui_tk`, `vector/ui_tk`, `web/app` + `frontend/`) — interfaces.
5. **Pipeline de dados** (`lyrics`, `enrichment`, `llm`, `_async_http`, `scripts/`, todos em `src/music_search/`) — gera os parquets versionados em `data/derived/final/`. Não roda no caminho crítico de uma busca; só quando se quer regenerar dados.

Para um guia gradual de cada peça (com fórmulas de BM25/TF-IDF e o trajeto ponta a ponta de uma query), ver `docs/GUIA.md`.

Fluxo runtime de uma query na API:

```
HTTP /search?q=… → web/app classifica intent (LLM ou heurística)
                 → motors.multi_index.search_routed() escolhe SparseSearchEngine ou EntityIndex
                 → SparseSearchEngine usa core.indexer + core.ranking sobre br_curated_lyrics.parquet
                 → web.snippets.extract_snippets() recorta letra (quando intent=lyric)
                 → opcional: llm.tasks.rerank()
```

## Comandos essenciais

```bash
uv sync --all-groups --extra vector --extra lyrics       # setup
uv run pytest                                             # testes
uv run ruff check . && uv run ruff format --check .       # lint+format
uv run --extra vector --extra lyrics ty check             # tipos
uv run python -m music_search.motors.search "amor saudade" # smoke esparso
uv run uvicorn music_search.web.app:app --reload          # API
cd frontend && npm run dev                                 # frontend
```

Comandos por subsistema (lyrics fetch, enrichment, vector indexing, `python -m music_search.scripts.build_curated_corpus`, ...) estão no `README.md`.

## Convenções

- Python 3.12+, `uv` para deps, **ruff** (line-length=100, regras E/W/F/I/UP/B/SIM/RUF), **ty** para tipos, **pytest** com `-v --tb=short`.
- Português em docstrings e variáveis de domínio; inglês em APIs e nomes técnicos.
- `data/` versionado, `data/*` gitignored com exceções para os datasets finais em `data/derived/final/`.
- Código **assume** que `data/derived/final/br_curated_lyrics.parquet` existe (vem no repo). Os parquets de entidades (`br_{artist,...}s.parquet`) podem não existir — código deve degradar graciosamente quando ausentes.
- O dataset Spotify bruto (`data/spotify-metadata/`) é opcional — só notebooks de EDA e o pipeline de letras tocam nele.

## Pontos de atenção ao editar

- **Não duplicar lógica de scoring**: `motors/search.py` e `motors/multi_index.py` reusam `BM25` e `TFIDF` de `core/ranking.py`. Mudanças de scoring devem ir no core.
- **Async + cache**: `lyrics` e `enrichment` compartilham infra em `_async_http/` (cache SQLite, token bucket, circuit breaker). Novas fontes devem implementar o protocolo em `sources/base.py`.
- **LLM é opcional**: `web/app.py` e `enrichment/pipeline.py` checam `NIM_API_KEY` e caem em fallbacks heurísticos quando ausente. Não introduzir dependência dura.
- **CORS do frontend**: o Vite dev server roda em `:5173` e o proxy reescreve `/api` → `http://127.0.0.1:8000`. CORS está liberado só para essa origem.
- **Scripts dentro do pacote**: builds rodam via `python -m music_search.scripts.<nome>`. A pasta `scripts/` da raiz não existe mais — o `download_spotify_metadata.sh` mora em `src/music_search/scripts/`.
- **`core/evaluation.py` está vazio** — quando for implementar métricas (Precision, Recall, MAP, nDCG), criar fixtures de golden set em `tests/data/`.

## Setup após clonar (resumo)

```bash
uv sync --all-groups --extra vector --extra lyrics
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('rslp')"
cd frontend && npm install && cd ..
uv run pytest  # sanidade
```

Não precisa baixar o Spotify bruto para a maioria das tarefas — os datasets finais em `data/derived/final/` já vêm no repo.
