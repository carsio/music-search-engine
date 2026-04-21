# Music Search Engine

**Um Estudo de Técnicas de Indexação e Ranking em Busca de Músicas**

Trabalho da disciplina ICC222 — Tópicos em Recuperação de Informação (UFAM 2026/1).

## Sobre

Sistema de busca de músicas que implementa e compara diferentes técnicas de indexação e ranking textual, utilizando o dataset [Spotify Metadata](https://www.kaggle.com/datasets/lordpatil/spotify-metadata-by-annas-archive) como base de dados.

### Técnicas implementadas

- **Indexação:** Índice invertido com suporte a diferentes esquemas de pesos
- **Ranking:** TF-IDF, BM25
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

## Estrutura do projeto

```
src/music_search/
├── __init__.py
├── preprocessing.py    # Tokenização, stemming, normalização
├── indexer.py          # Construção de índices invertidos
├── ranking.py          # Modelos de ranking (TF-IDF, BM25)
├── search.py           # Motor de busca / query processing
├── evaluation.py       # Métricas de avaliação de RI
└── web/
    ├── __init__.py
    └── app.py          # Interface web (FastAPI)
```

## Equipe

- [Carsio Eddyo](https://github.com/carsio)
- [Carlos Alexandre](https://github.com/alexandrecarloss)
- Raquel de Sá
- [Lelson Nascimento](https://github.com/lelsonln)

## Licença

MIT
