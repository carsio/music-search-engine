# Guia gradual do Music Search Engine

Este documento explica o projeto **do começo ao fim**, em ordem didática, para
quem quer entender como uma busca de música brasileira é construída e
respondida — das letras cruas em parquet até o JSON que chega no frontend.

Se você quer só rodar, vá para o `README.md`. Se você quer entender, comece
aqui e leia em ordem.

> **Convenção:** referências como `motors/search.py:120` apontam para arquivo
> e linha. Os caminhos são relativos a `src/music_search/`.

---

## Sumário

1. [Introdução: o problema e os algoritmos](#1-introdução-o-problema-e-os-algoritmos)
2. [Visão geral em camadas](#2-visão-geral-em-camadas)
3. [Pipeline de dados (offline)](#3-pipeline-de-dados-offline)
4. [Core de RI: pré-processamento, índice e ranking](#4-core-de-ri-pré-processamento-índice-e-ranking)
5. [Motores: como o core vira busca real](#5-motores-como-o-core-vira-busca-real)
6. [Intent classification e LLM (opcional)](#6-intent-classification-e-llm-opcional)
7. [Snippets de letras](#7-snippets-de-letras)
8. [Apresentação: API, frontend e UIs Tk](#8-apresentação-api-frontend-e-uis-tk)
9. [Trajeto ponta a ponta de uma query](#9-trajeto-ponta-a-ponta-de-uma-query)
10. [Como rodar e desenvolver](#10-como-rodar-e-desenvolver)
11. [Glossário](#11-glossário)

---

## 1. Introdução: o problema e os algoritmos

**Recuperação de Informação (RI)** é a área que estuda como encontrar
documentos relevantes em uma coleção a partir de uma consulta em linguagem
natural. A maioria dos buscadores que você usa todo dia é, em algum nível, um
sistema de RI.

Este projeto resolve o problema da disciplina **ICC222** (UFAM 2026/1):
construir um buscador para um catálogo curado de **50.000 faixas brasileiras**
do Spotify, das quais cerca de **36.000 têm letra completa**. As consultas
podem ser de naturezas diferentes:

- **Busca por letra** — `"saudade do meu amor"` → quero a música cuja letra
  contém esse trecho.
- **Busca por artista** — `"ana carolina"` → quero o artista (panel com bio).
- **Busca por gênero** — `"samba de raiz"` → quero faixas do gênero ou o
  próprio panel do gênero.
- **Busca por álbum/composição** — `"acabou chorare"` → quero a página do
  álbum.

Para resolver isso o sistema combina três famílias de técnicas, cada uma
forte em um regime:

| Técnica | Onde brilha | Onde falha |
|---|---|---|
| **TF-IDF** (cosseno) | termos exatos da query aparecem nos documentos | sinônimos, paráfrases |
| **BM25** Okapi | igual ao TF-IDF, mas com saturação de tf e normalização por tamanho — padrão da indústria | sinônimos, paráfrases |
| **Busca vetorial** (embeddings + Milvus) | semântica: "rock pesado anos 70" acha matches sem termos exatos | termos raros, nomes próprios |
| **LLM** (rerank/intent) | refinar top-k semanticamente, classificar a intenção da query | latência, custo, depende de chave |

A ideia é que **cada técnica complementa as outras**, e a apresentação final
roteia a consulta para o motor mais adequado.

---

## 2. Visão geral em camadas

O código é organizado em quatro camadas físicas, todas em `src/music_search/`:

```
┌─────────────────────────────────────────────────────────────────┐
│  Camada 1 — Core de RI (algoritmos puros)         [core/]       │
│  preprocessing → indexer → ranking (BM25, TF-IDF) → evaluation  │
└─────────────────────────────────────────────────────────────────┘
                              ↑ usa
┌─────────────────────────────────────────────────────────────────┐
│  Camada 2 — Motores e datasets             [motors/, data/]     │
│  motors.search.SparseSearchEngine    (multi-campo + boosts)     │
│  motors.multi_index.MultiEntityIndex (rota por intent)          │
│  motors.tuning             (perfis e pesos)                     │
│  data.datasets / data.albums         (loaders dos parquets)     │
│  vector.VectorSearch                 (embeddings + Milvus)      │
└─────────────────────────────────────────────────────────────────┘
                              ↑ usa
┌─────────────────────────────────────────────────────────────────┐
│  Camada 3 — Apresentação                                        │
│  ui_tk            (GUI Tk para comparar BM25 x TF-IDF)          │
│  vector/ui_tk     (GUI Tk para inspecionar busca vetorial)      │
│  web/app          (API FastAPI)        ← consumida pelo →       │
│  frontend/        (SPA React + Vite)                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Pipeline de dados (offline; não corre na busca)                │
│  scripts/         (build_curated_corpus, export_entities, ...)  │
│  lyrics/          (cascata de fontes para baixar letras)        │
│  enrichment/      (Wikipedia PT → payload local)                │
│  llm/             (cliente NIM opcional p/ intent e rerank)     │
│  _async_http/     (cache SQLite + throttle + circuit breaker)   │
└─────────────────────────────────────────────────────────────────┘
```

A ideia central é que a camada 1 (`core/`) **não conhece o domínio do
projeto** — ela trabalha com qualquer coleção de documentos. O domínio
("são tracks, têm letra e artista, vamos pesar letra mais que título")
mora na camada 2 (`motors/`).

Isso é importante por dois motivos: (i) o trabalho da disciplina é avaliado
pela camada 1, então ela tem que ser limpa e auditável; (ii) qualquer
mudança no scoring entra no core uma única vez e propaga.

---

## 3. Pipeline de dados (offline)

Antes de qualquer busca acontecer, é preciso ter um **corpus** — uma
coleção de documentos prontos para indexar. Este passo roda offline (não
na hora da query) e gera os parquets em `data/derived/final/` que já vêm
versionados no repositório. Você só precisa rodar isso se quiser
re-curar do zero.

### 3.1. Spotify bruto

A fonte primária é o **Spotify Metadata** (Anna's Archive / Kaggle), com
metadados estruturados de milhões de faixas: nome, artistas, álbum,
gêneros, popularidade, audio features (danceability, valence, tempo...),
imagens. Vai para `data/spotify-metadata/` (ignorado pelo Git).

Os notebooks `01..04` em `notebooks/` fazem a EDA e a curadoria — filtram
para faixas brasileiras (gênero contendo "brazilian", "samba", "bossa
nova", etc.; artistas com mercado BR; popularidade mínima) e produzem
`data/derived/final/br_curated_tracks.parquet`.

### 3.2. Coleta de letras: cascata de fontes

Saber o nome da música não é o suficiente — precisamos do **texto da
letra** para indexar. Isso vem de seis fontes públicas, em cascata:

```
LRCLib  →  Lyrics.ovh  →  Vagalume  →  Letras.mus.br  →  Genius  →  LyricFind
```

Cada uma é um arquivo em `lyrics/sources/`. Todas implementam o protocolo
`LyricsSource` definido em `lyrics/sources/base.py`:

```python
async def fetch(self, artist: str, title: str) -> LyricsResult:
    """Devolve HIT/MISS/ERROR/BLOCKED + texto se houver."""
```

A cascata em `lyrics/pipeline.py` tenta a primeira fonte; se ela retorna
MISS, tenta a próxima — com **variantes do título** (ex.: remover
`"(Remastered)"` ou `"- Single Version"`) e retry com backoff
exponencial. Tudo memoizado em SQLite (`data/derived/lyrics_cache.sqlite`)
via `lyrics/cache.py`, então rodar de novo é idempotente.

A infra anti-bloqueio mora em `_async_http/`:

- **`AsyncRateLimiter`** (token bucket) — limita chamadas por segundo por
  fonte, com `penalize()` que respeita `Retry-After` ao receber 429.
- **`CircuitBreaker`** — depois de N falhas consecutivas, "abre" e nega
  chamadas por um cooldown, evitando martelar uma fonte que caiu.
- **`KeyValueCache`** — cache SQLite genérico com `(key, status, payload,
  trace, error)`, reusado também pelo enrichment e pelo LLM.
- **`random_browser_headers`** — pool de User-Agents reais para fontes
  que filtram bots.

Ao final, `python -m music_search.scripts.build_curated_corpus` faz o join
entre tracks curadas e o cache de letras, produzindo
**`br_curated_lyrics.parquet`** — o corpus principal usado pela busca
esparsa.

### 3.3. Enrichment de entidades (Wikipedia PT)

Para responder `/artist/{id}` ou `/genre/{id}` com painéis ricos, são
geradas **dimensões de entidades** em parquets separados. O pipeline
está em `enrichment/`:

1. **Seeds** (`enrichment/seeds.py`): a partir do parquet de tracks,
   extrai a lista distinta de artistas, álbuns, gêneros e compositores a
   buscar.
2. **Cascade fetch** (`enrichment/sources/wikipedia_pt.py`): consulta a
   Wikipedia-API e, em fallback, scraping HTML — usando a mesma infra
   `_async_http/`.
3. **Materialização** (`enrichment/pipeline.py`): transforma o texto bruto
   em payload determinístico (nome, descrição/bio curta, gêneros
   relacionados, `raw_text` completo). Sem LLM por padrão; opcionalmente
   o LLM pode estruturar campos quando `NIM_API_KEY` está configurado.
4. **Export**
   (`python -m music_search.scripts.export_entities`):
   consolida o cache em `br_artists.parquet`, `br_genres.parquet`,
   `br_albums.parquet` e `br_composers.parquet`.

### 3.4. Manifest e versionamento

`python -m music_search.scripts.build_dataset` fecha o pipeline gerando
**`br_dataset_manifest.json`** com versão, contagens, tamanhos e hashes
SHA1 de cada parquet, e os parquets são versionados em
`data/derived/final/`. Quem clona o repo já tem dataset pronto para
indexar, sem precisar do Spotify bruto.

---

## 4. Core de RI: pré-processamento, índice e ranking

Esta é **a camada do trabalho**. Tudo aqui é puro: recebe documentos e
queries genéricas, devolve scores. Não conhece o nome `lyrics` nem
`spotify`.

### 4.1. Pré-processamento (`core/preprocessing.py`)

Antes de qualquer comparação, texto vira lista de **termos** normalizados.
A função canônica é `preprocess(text: str) -> list[str]`, que aplica em
ordem:

1. **`normalize`** — minúsculas, remove acentos (NFKD), troca pontuação
   por espaços. `"Canção de Ninar"` → `"cancao de ninar"`.
2. **`tokenize`** — quebra por palavra (NLTK `punkt_tab`).
3. **`remove_stopwords`** — descarta palavras vazias em PT e EN ("a",
   "de", "the", ...). Mantém termos de conteúdo.
4. **`stem`** — reduz à raiz: `"cantando"` → `"cant"`. **RSLP** para
   português (NLTK) e **Snowball** para inglês.

Por que stemar? Porque queremos que `"cantar"`, `"cantando"`,
`"cantamos"` casem com a query `"canta"`. Sem isso, BM25 trataria como
termos diferentes.

### 4.2. Índice invertido (`core/indexer.py`)

Para responder uma query rápido, **não** percorremos todos os documentos
a cada busca. Construímos uma estrutura inversa: para cada termo, qual a
lista de documentos onde ele aparece e com que frequência.

Esquema:

```python
postings[field][term] = [(doc_id, tf), ...]   # ordenado por doc_id
doc_lengths[field][doc_id] = int              # nº de tokens após preprocess
doc_ids[doc_id] = external_id                 # mapeia int interno → string
```

Três decisões importantes:

- **Multi-campo**: cada documento tem campos (`track_name`, `lyrics`,
  `artist_names`, ...) e o índice mantém postings separados por campo.
  Isso permite, na hora do ranking, dar peso diferente a um match em
  título vs. em letra.
- **IDs internos densos**: `doc_id` é um inteiro contíguo atribuído na
  ordem de inserção. Memória pequena, lookup O(1), e simplifica os
  rankers.
- **Persistência via pickle**: o índice (`data/indexes/br_curated_lyrics.pkl`)
  é regenerado uma vez (~10s) e reusado. Pickle é opaco, mas o índice
  não é contrato público — só um cache.

A construção é incremental: `IndexBuilder().add(doc_id, {field: text})`.

### 4.3. Ranking (`core/ranking.py`)

Dois modelos, ambos sobre o mesmo índice. Recebem uma query (lista de
termos) e devolvem score por documento.

#### TF-IDF com cosseno

Cada documento e a query viram vetores esparsos no espaço dos termos. O
peso de um termo $t$ no documento $d$ é:

$$w(t, d) = \mathrm{tf\_weight}(\mathrm{tf}(t,d)) \cdot \mathrm{idf}(t)$$

com IDF clássico:

$$\mathrm{idf}(t) = \ln\!\left(\frac{N}{\mathrm{df}(t)}\right)$$

onde $N$ é o número total de documentos e $\mathrm{df}(t)$ o número que
contém $t$. Termos raros pesam mais; termos universais pesam zero.

A similaridade é o cosseno entre os vetores:

$$\mathrm{cos\_sim}(d, q) = \frac{\mathbf{v}_d \cdot \mathbf{v}_q}{\|\mathbf{v}_d\|\,\|\mathbf{v}_q\|}$$

O `tf_weight` tem **três variantes** suportadas (Literal `TfScheme`):

- `"raw"` — `tf(t,d)` direto.
- `"log"` — `1 + ln(tf)`. Reduz dominância de palavras repetidas. **Padrão.**
- `"augmented"` — `0.5 + 0.5 · tf / max_tf_no_doc`. Normaliza por documento.

#### Okapi BM25

BM25 abandona o cosseno e calcula um score aditivo, com **saturação** da
contagem de termos e **normalização** pelo tamanho do documento:

$$\mathrm{BM25}(D, Q) = \sum_{t \in Q} \mathrm{IDF}_{\mathrm{bm25}}(t) \cdot \frac{\mathrm{tf}(t,D)\,(k_1 + 1)}{\mathrm{tf}(t,D) + k_1\!\left(1 - b + b \cdot \frac{|D|}{\mathrm{avgdl}}\right)}$$

com IDF suavizado:

$$\mathrm{IDF}_{\mathrm{bm25}}(t) = \ln\!\left(\frac{N - \mathrm{df}(t) + 0{,}5}{\mathrm{df}(t) + 0{,}5} + 1\right)$$

Hiperparâmetros (defaults seguindo Elasticsearch/Lucene):

- $k_1 = 1{,}5$ — controla quão rápido a contribuição de tf satura.
  $k_1 = 0$ ignora tf; $k_1 \to \infty$ vira contagem pura.
- $b = 0{,}75$ — peso da normalização por tamanho. $b = 0$ ignora tamanho;
  $b = 1$ normaliza totalmente.

Por que BM25 é melhor que TF-IDF na prática? Porque a saturação evita que
um documento que repete um termo 50 vezes domine; e a normalização por
$|D|/\mathrm{avgdl}$ não pune injustamente documentos longos legítimos
(letras grandes).

#### Implementação

Ambos os modelos pré-calculam IDFs e normas uma única vez via
`@cached_property`. O custo de uma query é $O(|Q| \cdot |\mathrm{candidatos}|)$,
onde candidatos = documentos que contêm pelo menos um termo da query.

### 4.4. Evaluation (`core/evaluation.py`)

Atualmente **vazio** — placeholder. Quando for medir qualidade comparada
(BM25 vs TF-IDF vs vetorial), aqui entram **Precision@k**, **Recall@k**,
**Mean Average Precision (MAP)** e **nDCG**, com fixtures de golden set
em `tests/data/`. É o cartão pendente da disciplina.

---

## 5. Motores: como o core vira busca real

O core não sabe que existem letras e nomes de artistas. Os motores
aplicam o core ao corpus específico e adicionam o que o domínio precisa:
boosts, roteamento, perfis.

### 5.1. SparseSearchEngine (`motors/search.py`)

O motor esparso sobre tracks. Na startup carrega
`br_curated_lyrics.parquet` em memória, constrói (ou lê do disco) o
índice persistido, e expõe:

```python
engine.search(query, algorithm="bm25", top_k=10) -> list[SearchHit]
```

A grande sacada é o **multi-campo com boosts**. Pesos default
(`DEFAULT_FIELD_WEIGHTS`):

| Campo | Peso | Por quê |
|---|---:|---|
| `lyrics` | 4.0 | conteúdo principal — quem canta o que importa |
| `track_name` | 2.5 | título é forte sinal de relevância |
| `artist_genres` | 1.5 | gênero ajuda a filtrar |
| `artist_names` | 1.0 | nome do artista |
| `macro_genre` | 1.0 | macro-categoria curada |
| `album_name` | 0.75 | sinal mais fraco |

O motor calcula score BM25 (ou TF-IDF) **por campo separadamente**, normaliza
por campo (para combinar score em escalas diferentes) e soma ponderado pelo
peso. O `SearchHit` resultante traz, além do score final, a contribuição
de cada campo (`field_contributions`) — útil para debug e para a UI Tk
mostrar por que aquele resultado subiu.

### 5.2. MultiEntityIndex (`motors/multi_index.py`)

Em cima do `SparseSearchEngine`, define uma fachada que sabe sobre
**várias entidades**: tracks, artists, albums, genres, composers.

- Para tracks, delega ao `SparseSearchEngine`.
- Para as demais, define `EntityIndex` — uma versão mais leve que indexa
  payloads (dicts vindos dos parquets de entidades) com o mesmo
  `core/indexer.py` e `core/ranking.py`. Pesos por entidade vêm de
  `motors/tuning.py`.

A função-chave é `search_routed(query, intent, top_k)` — ela recebe a
intenção da query e despacha para o índice certo:

```
intent="track"   → SparseSearchEngine sobre tracks
intent="artist"  → EntityIndex(artist)
intent="album"   → EntityIndex(album) (se existir)
intent="genre"   → EntityIndex(genre)
intent="lyric"   → SparseSearchEngine restrito ao campo lyrics + snippets
```

Se um parquet de entidade não existe (ex.: `br_albums.parquet` ainda não
foi gerado), o índice degrada graciosamente: o roteamento cai em
fallback para tracks. Isso é importante porque o snapshot versionado
nem sempre tem todas as dimensões.

### 5.3. Tuning (`motors/tuning.py`)

Define **perfis** (`SearchProfile`): `balanced`, `lyrics`, `metadata`.
Cada um aplica um multiplicador sobre os pesos default — por exemplo, o
perfil `lyrics` aumenta o peso de `lyrics` e reduz `track_name`. Usado
quando se sabe que a query é de letra.

### 5.4. Vector search (`vector/`)

Motor complementar e opcional. Indexa as tracks como **embeddings** via
Ollama local (`nomic-embed-text`, 768 dims) ou OpenAI
(`text-embedding-3-small`, 1536 dims), persiste em **Milvus Lite**
(`data/vector/milvus_spotify.db`) e responde por similaridade
vetorial. Útil para queries semânticas como `"rock pesado anos 70 com
guitarra"`. Não está no caminho crítico da API — é exposto por CLI e UI
Tk próprias.

---

## 6. Intent classification e LLM (opcional)

A query `"saudade do meu amor"` tem cara de letra. `"ana carolina"` tem
cara de artista. Saber isso muda o motor que vai responder. A
classificação de intent acontece em duas etapas:

### 6.1. Heurística (sempre disponível)

Em `web/app.py` há um classificador determinístico simples:

- contém `"album "`, `"banda "`, ou termos típicos → `album`;
- contém `"letra "`, frases longas com pronomes → `lyric`;
- 1-3 palavras só com nomes próprios → `artist`;
- contém `"samba"`, `"bossa"`, gêneros conhecidos → `genre`;
- caso geral → fallback `lyric`.

Funciona razoável e nunca falha por chave/rede.

### 6.2. LLM (quando `NIM_API_KEY` está setada)

`llm/tasks.py` define `classify_intent(query) -> Intent`, que chama um
LLM (NIM API, OpenAI-compatível) com um system prompt que pede para
classificar entre `artist | album | song | lyric | genre | none`. O
resultado é memoizado em `LLMCache` (SQLite, chave =
`sha1(prompt + input)`), então a segunda vez é instantânea.

Se a chamada falha (sem chave, sem rede, timeout), cai silenciosamente
no classificador heurístico. **Nunca há dependência dura no LLM.**

### 6.3. Rerank LLM (opcional, ?rerank=true)

Depois que o motor esparso retorna top-K (digamos, 20), o cliente pode
pedir `?rerank=true`. Aí `llm/tasks.py rerank()` envia query + os 20
candidatos para o LLM com prompt que pede ranking semântico, e
devolve a nova ordem. Lento (~1s) e caro (custo de tokens), por isso
opt-in.

### 6.4. Materialização de entidades

Há um terceiro uso do LLM: durante o enrichment, opcionalmente, para
transformar texto bruto da Wikipedia em JSON estruturado
(`extract_artist_json`). É offline, então latência não importa. O
enrichment determinístico cobre o caso sem LLM, então o LLM aqui é só
melhoria.

---

## 7. Snippets de letras

Quando a intent é `lyric`, mostrar a faixa inteira não ajuda — queremos
**a parte da letra que casou**.

`web/snippets.py:extract_snippets(lyrics, query)`:

1. Aplica `preprocess(query)` para obter os termos canônicos.
2. Quebra a letra em linhas.
3. Para cada linha, conta quantos termos da query aparecem (em forma
   stemada).
4. Devolve as top-N linhas (default 3) com a maior contagem, junto com
   o número da linha original.

`highlight_terms()` posteriormente envolve cada termo com `<mark>...
</mark>`, e o frontend renderiza o destaque visual sem precisar refazer
o trabalho de matching.

---

## 8. Apresentação: API, frontend e UIs Tk

### 8.1. API FastAPI (`web/app.py`)

A startup carrega `SparseSearchEngine` + `MultiEntityIndex` via
`@asynccontextmanager` lifespan. Os endpoints principais:

| Endpoint | Função |
|---|---|
| `GET /api/healthz` | status + contagens |
| `GET /api/search?q=&top=10&algorithm=bm25&rerank=false` | busca roteada |
| `GET /api/search/lyric?q=&top=20` | busca só em letras com snippets |
| `GET /api/artist/{id}` | knowledge panel de artista |
| `GET /api/album/{id}` | página de álbum |
| `GET /api/song/{id}` | letra completa |

CORS está liberado para `http://localhost:5173` (Vite dev server).

### 8.2. Frontend (`frontend/`)

SPA em **React 18 + Vite + TypeScript**. Estrutura:

- `src/api/` — cliente HTTP (axios + react-query) e tipos espelhando os
  schemas Pydantic do backend.
- `src/hooks/` — hooks de busca, debounce, query params.
- `src/components/` — blocos de UI por contexto (`home`, `panels`,
  `search`, `states`, `primitives`, `layout`).
- `src/views/` — composição das telas.

Em dev, `npm run dev` no `frontend/` sobe o Vite em `:5173` e o proxy
encaminha `/api/*` para `http://127.0.0.1:8000/api/*`.

Em produção (Docker), o React é buildado para `frontend/dist/` e o
próprio FastAPI serve os estáticos, mantendo `/api/*` para JSON.

### 8.3. UIs Tk (uso local sem subir API)

- `ui_tk.py` — GUI Tk principal: busca esparsa com toggle
  BM25/TF-IDF lado a lado e detalhamento de campos.
- `vector/ui_tk.py` — GUI Tk para inspecionar busca vetorial.
- `lyrics/ui_tk.py` — orquestra o pipeline de coleta de letras.
- `enrichment/ui_tk.py` — orquestra o enrichment + export + build_dataset.

Úteis para debug e para apresentar o trabalho sem depender do frontend.

---

## 9. Trajeto ponta a ponta de uma query

Vamos seguir `"saudade do meu amor"` desde o navegador:

1. **Frontend** (`frontend/src/api/`) faz
   `GET /api/search?q=saudade+do+meu+amor&top=10`.
2. **FastAPI** (`web/app.py`) recebe a chamada, monta um
   `SearchRequest`.
3. **Intent classification**: tenta LLM se `NIM_API_KEY` setada (cache
   SQLite hit → instantâneo); senão, heurística → `intent="lyric"`.
4. **Routing**: `multi.search_routed(query, intent="lyric", top_k=10)`
   chama `SparseSearchEngine.search(query)` restrito ao campo `lyrics`.
5. **Pré-processamento**: `preprocess("saudade do meu amor")` →
   `["saudad", "amor"]` (stopwords "do", "meu" caem; stems aplicados).
6. **Lookup no índice**: para cada termo, postings devolve a lista de
   tracks que contêm o termo, com tf por track.
7. **Ranking BM25**: `core/ranking.py` calcula score por track, com
   `k1=1.5`, `b=0.75`. Top 10 sai ordenado.
8. **Field boost**: como só lyrics está em jogo, peso de `lyrics`
   (4.0) é aplicado.
9. **Snippets**: para cada hit, `extract_snippets(lyrics, query)`
   pega 3 linhas da letra que casaram.
10. **Resposta**: serializada em `SearchResponse` (Pydantic) →
    `[{rank, score, track_name, artist, snippets, …}]`.
11. **Frontend** renderiza cards com os snippets já marcados em
    `<mark>`.

Tempo total típico (com cache quente): ~30 ms para a busca, +alguns ms
de serialização. Se `?rerank=true`, mais ~800 ms de chamada LLM.

---

## 10. Como rodar e desenvolver

> Esta seção é só atalho. O `README.md` tem o detalhe completo.

```bash
# Setup
uv sync --all-groups --extra vector --extra lyrics
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('rslp')"

# Smoke test do core esparso
uv run python -m music_search.motors.search "amor saudade" --top 5

# API + frontend
uv run uvicorn music_search.web.app:app --reload   # term 1
cd frontend && npm run dev                          # term 2

# Pipelines de build (offline; só se for re-curar)
uv run python -m music_search.scripts.build_curated_corpus
uv run python -m music_search.scripts.export_entities
uv run python -m music_search.scripts.build_dataset --skip-lyrics

# Qualidade
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run --extra vector --extra lyrics ty check
```

---

## 11. Glossário

- **Corpus** — coleção de documentos a serem indexados.
- **Documento** — uma faixa, no nosso caso. Tem campos como `track_name`
  e `lyrics`.
- **Termo** — token resultante do pré-processamento.
- **TF (term frequency)** — quantas vezes o termo aparece no documento.
- **DF (document frequency)** — em quantos documentos o termo aparece.
- **IDF (inverse document frequency)** — `ln(N/df)`. Inverte: termos
  raros pesam mais.
- **Postings list** — para cada termo, a lista `[(doc_id, tf), ...]` dos
  documentos que o contêm.
- **Índice invertido** — estrutura `term → postings`. Permite responder
  "quem tem este termo?" em O(1).
- **TF-IDF** — modelo de ranking por similaridade de cosseno entre vetores
  ponderados por TF·IDF.
- **BM25** — modelo de ranking aditivo com saturação de TF (`k1`) e
  normalização por tamanho (`b`). Padrão da indústria.
- **Stemming** — reduzir flexões à raiz comum (`"cantando"` → `"cant"`).
  Aqui usamos **RSLP** para PT e **Snowball** para EN.
- **Stopword** — palavra muito comum descartada antes da indexação.
- **Field boost** — multiplicar o score de um campo por um peso para que
  match em campos importantes (ex.: `lyrics`) conte mais que em
  acessórios (ex.: `album_name`).
- **Embedding** — representação densa de um texto em um espaço vetorial
  contínuo. Permite busca por significado.
- **Milvus** — banco vetorial usado para a busca semântica.
- **Intent** — classificação da pergunta do usuário (artist, album, song,
  lyric, genre, none).
- **Rerank** — reordenar o top-K do motor esparso usando um modelo mais
  caro (LLM) por relevância semântica.
- **Snippet** — recorte de letra que contém os termos da query, exibido
  com destaque visual no frontend.
- **Cascade fetch** — tentar várias fontes em ordem até uma responder
  com sucesso, com cache em SQLite para idempotência.
- **Token bucket** — algoritmo de rate limit que permite picos curtos
  até `capacity` mas mantém média de `rate` por segundo.
- **Circuit breaker** — bloqueia chamadas a uma fonte quebrada por um
  tempo após N falhas consecutivas, evitando martelar serviços caídos.
