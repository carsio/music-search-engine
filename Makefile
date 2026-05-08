SHELL := /bin/bash
.DEFAULT_GOAL := help

SEARCH_QUERY ?= amor saudade
SEARCH_TOP ?= 5
SEARCH_ALGORITHM ?= both
VECTOR_QUERY ?= rock classico anos 70
VECTOR_TOP ?= 5
LYRICS_ARTIST ?= Anitta
LYRICS_TITLE ?= Envolver
LYRICS_FETCH_ARGS ?=
WEB_HOST ?= 127.0.0.1
WEB_PORT ?= 8000

.PHONY: help setup setup-vector setup-lyrics setup-all nltk download download-truncated \
	download-full corpus index search ui web lint format format-check typecheck test \
	check vector-index vector-search vector-ui lyrics-probe lyrics-fetch lyrics-stats \
	lyrics-export lyrics-ui enrichment-ui

help: ## List available targets
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets disponiveis:\n\n"} /^[a-zA-Z0-9_.-]+:.*##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Install runtime, dev and notebook dependencies
	uv sync --all-groups

setup-vector: ## Install dependencies with the vector extra
	uv sync --all-groups --extra vector

setup-lyrics: ## Install dependencies with the lyrics extra
	uv sync --all-groups --extra lyrics

setup-all: ## Install dependencies with vector and lyrics extras
	uv sync --all-groups --extra vector --extra lyrics

nltk: ## Download required NLTK data
	uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('rslp')"

download: download-truncated ## Download the truncated Spotify dataset

download-truncated: ## Download the truncated Spotify dataset
	./src/music_search/scripts/download_spotify_metadata.sh --truncated

download-full: ## Download the full Spotify dataset via Kaggle
	./src/music_search/scripts/download_spotify_metadata.sh --full

corpus: ## Rebuild the curated corpus parquet
	uv run python -m music_search.scripts.build_curated_corpus

index: ## Build the sparse index for the default corpus
	uv run python -m music_search.scripts.build_index

search: ## Run sparse CLI search (use SEARCH_QUERY/SEARCH_TOP/SEARCH_ALGORITHM)
	uv run python -m music_search.motors.search "$(SEARCH_QUERY)" --algorithm $(SEARCH_ALGORITHM) --top $(SEARCH_TOP)

ui: ## Open the sparse Tkinter UI
	uv run python -m music_search.ui_tk

web: ## Start the FastAPI dev server
	uv run uvicorn music_search.web.app:app --reload --host $(WEB_HOST) --port $(WEB_PORT)

lint: ## Run Ruff lint
	uv run ruff check .

format: ## Apply Ruff formatting
	uv run ruff format .

format-check: ## Check formatting without changing files
	uv run ruff format --check .

typecheck: ## Run ty type checks
	uv run --extra vector --extra lyrics ty check

test: ## Run the pytest suite
	uv run pytest

check: lint format-check typecheck test ## Run the non-mutating quality checks

vector-index: ## Build or update the vector index
	uv run --extra vector python -m music_search.vector.indexing

vector-search: ## Run vector CLI search (use VECTOR_QUERY/VECTOR_TOP)
	uv run --extra vector python -m music_search.vector.search "$(VECTOR_QUERY)" --top $(VECTOR_TOP)

vector-ui: ## Open the vector Tkinter UI
	uv run --extra vector python -m music_search.vector.ui_tk

lyrics-probe: ## Probe lyrics sources (use LYRICS_ARTIST/LYRICS_TITLE)
	uv run --extra lyrics python -m music_search.lyrics probe "$(LYRICS_ARTIST)" "$(LYRICS_TITLE)"

lyrics-fetch: ## Fetch lyrics into the cache (use LYRICS_FETCH_ARGS='--limit 100 --concurrency 8')
	uv run --extra lyrics python -m music_search.lyrics fetch $(LYRICS_FETCH_ARGS)

lyrics-stats: ## Show lyrics cache stats
	uv run --extra lyrics python -m music_search.lyrics stats

lyrics-export: ## Export cached lyrics hits to parquet
	uv run --extra lyrics python -m music_search.lyrics export

lyrics-ui: ## Open the lyrics Tkinter UI
	uv run --extra lyrics python -m music_search.lyrics.ui_tk

enrichment-ui: ## Open the enrichment Tkinter UI
	uv run python -m music_search.enrichment.ui_tk
