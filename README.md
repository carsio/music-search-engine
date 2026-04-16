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

# Rodar testes
uv run pytest

# Iniciar servidor web
uv run uvicorn music_search.web.app:app --reload

# Lint e formatação
uv run ruff check .
uv run ruff format .
```

## Dados

Baixe o dataset do Kaggle e coloque os arquivos CSV em `data/`:

```bash
# Via Kaggle CLI
kaggle datasets download -d lordpatil/spotify-metadata-by-annas-archive -p data/ --unzip
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
- Lelson Nascimento

## Licença

MIT
