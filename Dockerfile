# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    MUSIC_SEARCH_REPO_ROOT=/app

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/derived/final ./data/derived/final
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
RUN mkdir -p ./data/indexes

RUN pip install --upgrade pip \
    && pip install ".[lyrics]" \
    && python -m nltk.downloader -d "$NLTK_DATA" punkt_tab stopwords rslp

# Pre-computa o índice esparso no build para evitar 5–10 min de CPU
# no primeiro boot do container. O .pkl resultante é serializado em
# data/indexes/ e reaproveitado em runtime via load_or_build_default_engine.
RUN python -c "from music_search.motors.search import load_or_build_default_engine; load_or_build_default_engine()"

EXPOSE 8000

CMD ["uvicorn", "music_search.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
