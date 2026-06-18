# Music Search Engine

**Busca de músicas brasileiras com BM25, TF-IDF e embeddings vetoriais.**

Trabalho da disciplina ICC222 — Tópicos em Recuperação de Informação (UFAM 2026/1).

| | |
| --- | --- |
| 🌐 **Sistema (webapp)** | **<https://musicabr.online/>** |
| 📊 Slides (apresentação) | <https://carsio.github.io/music-search-engine/> |
| 📄 Relatório técnico (PDF) | [main.pdf](https://github.com/carsio/music-search-engine/releases/download/report-latest/main.pdf) |
| 📚 Wiki (setup, uso, pipeline) | <https://github.com/carsio/music-search-engine/wiki> |
| 📖 Guia conceitual (RI do zero) | [`docs/GUIA.md`](docs/GUIA.md) |

> Quer **rodar** ou entender a operação do sistema? Setup, modos de uso, pipeline
> de dados e estrutura do repositório estão na
> **[Wiki](https://github.com/carsio/music-search-engine/wiki)**. Para entender o
> sistema do começo ao fim (conceitos de RI, BM25/TF-IDF com fórmulas, motores e
> o trajeto de uma query), comece por **[`docs/GUIA.md`](docs/GUIA.md)**.

## Visão geral

O projeto implementa os algoritmos clássicos de RI (índice invertido, TF-IDF, BM25) e a busca vetorial densa (embeddings + Milvus), e expõe esse motor através de uma API FastAPI consumida por um frontend React. Os dados foram curados a partir do Spotify Metadata e enriquecidos com letras (lyrics.ovh, Vagalume, letras.mus.br, Genius) e conteúdo da Wikipedia PT materializado de forma determinística para as entidades.

O fluxo principal hoje é: a API em `music_search.web.app` recebe a query, classifica a intent via heurística determinística, delega o roteamento para `multi_index.MultiEntityIndex`, consulta o índice esparso ou a dimensão apropriada e devolve a resposta para a SPA em `frontend/`.

A solução foi avaliada com uma coleção de referência estilo TREC (50 consultas, 1.604 julgamentos graduados): os métodos esparsos (BM25 e TF-IDF) superam a busca densa de forma estatisticamente significativa neste domínio, com o BM25 como linha de base dominante. Os detalhes estão no [relatório técnico](https://github.com/carsio/music-search-engine/releases/download/report-latest/main.pdf).

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
│  Spotify raw → tracks BR → lyrics/ → corpus → enrichment/        │
│  Detalhes em: Wiki › Pipeline de Dados                          │
└─────────────────────────────────────────────────────────────────┘
```

> Detalhamento de cada subsistema e dos atalhos do VSCode:
> [Wiki › Estrutura do Repositório](https://github.com/carsio/music-search-engine/wiki/Estrutura-do-Repositorio).

## Status — feito vs falta

### ✅ Pronto

- Pipeline RI clássico: preprocessing, índice invertido multi-campo, BM25, TF-IDF
- Motor esparso multi-campo com pesos configuráveis (`SparseSearchEngine`)
- Roteamento via `MultiEntityIndex` com fallback gracioso quando alguma dimensão ainda não existe
- Busca vetorial com Ollama/OpenAI + Milvus Lite
- Catálogo de álbuns derivado do dataset de tracks e endpoint `/album/{id}`
- Dataset versionado com **50.000 faixas brasileiras**, **36.017 músicas com letra**, **7.255 artistas** e **42 gêneros**
- Pipeline de letras com cache, retries, circuit breaker e cascata de fontes
- Pipeline de enrichment determinístico da Wikipedia PT já exportando artistas e gêneros
- API FastAPI (`/search`, `/search/lyric`, `/artist`, `/album`, `/song`) + frontend React/Vite
- Avaliação estilo TREC (50 consultas, 1.604 julgamentos) com métricas MRR, MAP, nDCG@10, P@10, Bpref e testes de Wilcoxon
- CI (`.github/workflows/ci.yml`) para lint, format, type check e pytest

### 🟡 Em andamento

- **Álbuns e compositores enriquecidos**: `br_albums.parquet` e `br_composers.parquet` ainda não entram no snapshot versionado atual.
- **Cobertura do `MultiEntityIndex`**: artist e genre já carregam do manifesto atual, mas album/composer ainda dependem dos exports restantes.

## Equipe

- [Carsio Eddyo](https://github.com/carsio)
- [Carlos Alexandre](https://github.com/alexandrecarloss)
- [Raquel de Sá](https://github.com/raqueldesa)
- [Lelson Nascimento](https://github.com/lelsonln)

## Licença

MIT
