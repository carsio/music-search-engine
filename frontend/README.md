# Frontend — músicabr

Single-page app em React + Vite + TypeScript que consome a API FastAPI em
`src/music_search/web/app.py`. Substitui o mockup standalone em `music search/`
(que segue versionado como referência visual).

## Setup

```bash
cd frontend
npm install
```

## Rodar em dev

Em um terminal, sobe a API:

```bash
# na raiz do repo
uv run uvicorn music_search.web.app:app --reload --port 8000
```

Em outro, sobe o Vite:

```bash
cd frontend
npm run dev
```

Abre `http://localhost:5173`. O proxy do Vite roteia `/api/*` para
`http://127.0.0.1:8000/*`, então não há CORS no dev.

## Build de produção

```bash
npm run build
```

Gera `frontend/dist/`. A API pode servir esses estáticos via FastAPI
`StaticFiles` se quiser empacotar tudo em um único deploy.

## Estrutura

```
src/
├── api/client.ts          # axios + react-query hooks
├── components/            # SearchBox, Homepage, ResultsList, LyricMatches,
│                          # ArtistPanel, SongPanel
├── routes/                # SearchResultsRoute, ArtistRoute, SongRoute
├── styles/app.css         # estilos baseados em music search/styles.css
├── types.ts               # espelha src/music_search/web/schemas.py
├── App.tsx
└── main.tsx
```

## Painéis MVP

- `LyricMatches`: trecho de letra → BM25/TF-IDF + snippets numerados
- `ArtistPanel`: knowledge panel (bio, gêneros, top tracks, discografia)
- `SongPanel`: letra completa com destaque dos termos da query

`/search` retorna o intent classificado pela API (LLM se `NIM_API_KEY`
estiver setado, ou heurística como fallback) e roteia o card principal.
