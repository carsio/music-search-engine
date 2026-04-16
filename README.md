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

# Verificar a CLI local do Kaggle
uv run kaggle --version

# Baixar o dataset no local padrão
./scripts/download_spotify_metadata.sh

# Rodar testes
uv run pytest

# Iniciar servidor web
uv run uvicorn music_search.web.app:app --reload

# Lint e formatação
uv run ruff check .
uv run ruff format .
```

## Dados

O diretório `data/` é versionado pelo repositório. Cada dataset deve ficar em uma subpasta dentro
dele. Por padrão, o script deste projeto baixa o dataset do Spotify em
`data/spotify-metadata`.

Antes do download, gere sua credencial em `Kaggle > Settings > Create New Token` e salve o arquivo
em `~/.kaggle/kaggle.json` com permissão restrita:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

Depois, use o script do projeto para baixar o dataset no local padrão:

```bash
./scripts/download_spotify_metadata.sh
```

Se você quiser armazenar o dataset fora do repositório, passe um caminho explicitamente. Nesse
caso, o script recria `data/spotify-metadata` como symlink para preservar o layout esperado pelo
projeto:

```bash
./scripts/download_spotify_metadata.sh /caminho/para/datasets
```

O argumento pode ser a pasta base ou o diretório final do dataset:

```bash
./scripts/download_spotify_metadata.sh /caminho/para/datasets
./scripts/download_spotify_metadata.sh /caminho/para/datasets/spotify-metadata
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
