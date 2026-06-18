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
    MUSIC_SEARCH_REPO_ROOT=/app \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/derived/final ./data/derived/final
RUN mkdir -p ./data/indexes

RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[lyrics,dense]" \
    && python -m nltk.downloader -d "$NLTK_DATA" punkt_tab stopwords rslp

# Pre-computa índices esparsos, de entidades e denso (embeddings FAISS) durante
# o build. Ficam em data/indexes/ e são reaproveitados em runtime.
# Copiamos o frontend DEPOIS para que mudanças de UI não invalidem este cache.
RUN python -m music_search.scripts.prepare_search_artifacts

COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["uvicorn", "music_search.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
