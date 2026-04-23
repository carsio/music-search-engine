"""Geração de embeddings do dataset Spotify e inserção no Milvus.

Consome os parquets canônicos em `data/spotify-metadata/spotify_clean_parquet/`
via `SpotifyTracksLoader.iter_rich_docs()` — sem passar por um CSV intermediário.

Backends de embedding (selecionados por `USE_OLLAMA`):
  - Ollama local      → `nomic-embed-text`          (768 dims)   [padrão]
  - OpenAI API        → `text-embedding-3-small`    (1536 dims)

Artefatos (criados sob demanda em `data/vector/`):
  - `milvus_spotify.db`       banco Milvus Lite
  - `embedding_checkpoint.txt` último offset processado (retomada)
  - `embedding_log.txt`       log da execução

Uso:

    # Com Ollama local (default — `ollama serve` rodando):
    uv run python -m music_search.vector.indexing

    # Com OpenAI:
    USE_OLLAMA=false OPENAI_API_KEY=sk-... uv run python -m music_search.vector.indexing

    # Limitando número de tracks (smoke test):
    INDEX_LIMIT=1000 uv run python -m music_search.vector.indexing
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from music_search.datasets import SpotifyTracksLoader
from music_search.vector.config import (
    COLLECTION_NAME,
    VECTOR_DATA_DIR,
    EmbeddingConfig,
    default_milvus_uri,
)

if TYPE_CHECKING:
    from openai import OpenAI
    from pymilvus import MilvusClient

BATCH_SIZE = 100  # textos por chamada de embedding
CHECKPOINT_FILE = VECTOR_DATA_DIR / "embedding_checkpoint.txt"
LOG_FILE = VECTOR_DATA_DIR / "embedding_log.txt"

log = logging.getLogger(__name__)


# ── Helpers de serialização ───────────────────────────────────────────────────


def row_to_text(row: dict[str, Any]) -> str:
    """Monta o texto que será transformado em embedding.

    Campos vazios são omitidos para não poluir o vetor com `field: ` sem valor.
    """
    parts = [
        ("track", row.get("track_name")),
        ("artists", row.get("artist_names")),
        ("album", row.get("album_name")),
        ("genres", row.get("artist_genres")),
        ("album_type", row.get("album_type")),
        ("label", row.get("label")),
        ("release_date", row.get("release_date")),
        ("explicit", row.get("explicit", False)),
        ("track_popularity", row.get("track_popularity", 0)),
        ("album_popularity", row.get("album_popularity", 0)),
        ("duration_ms", row.get("duration_ms", 0)),
    ]
    pieces = []
    for key, value in parts:
        # Omite None, strings vazias e defaults numéricos/booleanos falsos
        # (0 e False são ruído em quase todo documento).
        if value is None or value == "" or value is False or value == 0:
            continue
        pieces.append(f"{key}: {value}")
    return "; ".join(pieces)


def row_to_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, default=str)


def truncate(value: Any, max_len: int) -> str:
    """Trunca em `max_len` bytes UTF-8 sem quebrar caracteres multi-byte."""
    s = "" if value is None else str(value)
    encoded = s.encode("utf-8")
    if len(encoded) <= max_len:
        return s
    return encoded[:max_len].decode("utf-8", errors="ignore")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    try:
        return bool(int(value))
    except (ValueError, TypeError):
        return False


# ── Embedding com retry ───────────────────────────────────────────────────────


def _embed_batch(
    client: OpenAI, model: str, texts: list[str], retries: int = 5
) -> list[list[float]]:
    from openai import APIError, RateLimitError

    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in resp.data]
        except RateLimitError:
            wait = 2**attempt
            log.warning("Rate limit — aguardando %ds (tentativa %d/%d)", wait, attempt + 1, retries)
            time.sleep(wait)
        except APIError as exc:
            log.error("Erro da API: %s", exc)
            raise
    raise RuntimeError("Número máximo de tentativas atingido")


# ── Milvus: criação da coleção ────────────────────────────────────────────────


def _create_collection(client: MilvusClient, embed_dim: int) -> None:
    from pymilvus import DataType

    if client.has_collection(COLLECTION_NAME):
        log.info("Coleção '%s' já existe — reutilizando.", COLLECTION_NAME)
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("track_name", DataType.VARCHAR, max_length=500)
    schema.add_field("artist_names", DataType.VARCHAR, max_length=2000)
    schema.add_field("album_name", DataType.VARCHAR, max_length=500)
    schema.add_field("artist_genres", DataType.VARCHAR, max_length=2000)
    schema.add_field("album_type", DataType.VARCHAR, max_length=50)
    schema.add_field("label", DataType.VARCHAR, max_length=500)
    schema.add_field("release_date", DataType.VARCHAR, max_length=20)
    schema.add_field("track_popularity", DataType.INT64)
    schema.add_field("album_popularity", DataType.INT64)
    schema.add_field("duration_ms", DataType.INT64)
    schema.add_field("explicit", DataType.BOOL)
    schema.add_field("json_data", DataType.VARCHAR, max_length=65_535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=embed_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    log.info("Coleção '%s' criada (dim=%d, métrica=COSINE).", COLLECTION_NAME, embed_dim)


# ── Checkpoint ────────────────────────────────────────────────────────────────


def _read_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        return int(CHECKPOINT_FILE.read_text().strip())
    return 0


def _save_checkpoint(offset: int) -> None:
    CHECKPOINT_FILE.write_text(str(offset))


# ── Factories ─────────────────────────────────────────────────────────────────


def _build_embed_client(cfg: EmbeddingConfig) -> OpenAI:
    from openai import OpenAI

    if cfg.use_ollama:
        return OpenAI(base_url=cfg.ollama_url, api_key="ollama")
    if not cfg.openai_api_key:
        sys.exit("Erro: defina OPENAI_API_KEY ou use USE_OLLAMA=true")
    return OpenAI(api_key=cfg.openai_api_key)


def _build_milvus_client(uri: str) -> MilvusClient:
    from pymilvus import MilvusClient

    return MilvusClient(uri=uri)


def _configure_logging() -> None:
    VECTOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────


def _batched(iterable, size: int):
    """Divide um iterável em lotes de tamanho fixo (o último pode ser menor)."""
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_doc(row: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
    return {
        "track_name": truncate(row.get("track_name"), 500),
        "artist_names": truncate(row.get("artist_names"), 2000),
        "album_name": truncate(row.get("album_name"), 500),
        "artist_genres": truncate(row.get("artist_genres"), 2000),
        "album_type": truncate(row.get("album_type"), 50),
        "label": truncate(row.get("label"), 500),
        "release_date": truncate(row.get("release_date"), 20),
        "track_popularity": _safe_int(row.get("track_popularity")),
        "album_popularity": _safe_int(row.get("album_popularity")),
        "duration_ms": _safe_int(row.get("duration_ms")),
        "explicit": _safe_bool(row.get("explicit")),
        "json_data": truncate(row_to_json(row), 65_535),
        "embedding": embedding,
    }


def main(
    *,
    limit: int | None = None,
    parquet_dir: Path | None = None,
) -> None:
    """Executa a pipeline de indexação vetorial."""
    _configure_logging()

    cfg = EmbeddingConfig.from_env()
    milvus_uri = default_milvus_uri()

    log.info(
        "Backend embedding : %s (%s, dim=%d)",
        "Ollama" if cfg.use_ollama else "OpenAI",
        cfg.model,
        cfg.dim,
    )
    log.info("Milvus URI        : %s", milvus_uri)

    embed_client = _build_embed_client(cfg)
    milvus = _build_milvus_client(milvus_uri)
    _create_collection(milvus, cfg.dim)

    offset_inicial = _read_checkpoint()
    if offset_inicial > 0:
        log.info("Retomando do offset %d", offset_inicial)

    loader = SpotifyTracksLoader(parquet_dir) if parquet_dir else SpotifyTracksLoader()

    total_inseridos = 0
    offset = 0
    for batch in _batched(loader.iter_rich_docs(limit=limit), BATCH_SIZE):
        batch_end = offset + len(batch)

        # Pula lotes já processados
        if batch_end <= offset_inicial:
            offset = batch_end
            continue

        # Se cair no meio de um lote, aproveita só a parte nova
        start = max(0, offset_inicial - offset)
        batch = batch[start:]

        texts = [row_to_text(row) for row in batch]
        embeddings = _embed_batch(embed_client, cfg.model, texts)
        docs = [_build_doc(row, emb) for row, emb in zip(batch, embeddings, strict=True)]

        milvus.insert(collection_name=COLLECTION_NAME, data=docs)
        total_inseridos += len(docs)
        offset = batch_end
        _save_checkpoint(offset)

        if offset % 1000 == 0 or len(batch) < BATCH_SIZE:
            log.info("Progresso: %d processados | inseridos: %d", offset, total_inseridos)

    log.info("Concluído! Total inserido no Milvus: %d", total_inseridos)
    CHECKPOINT_FILE.unlink(missing_ok=True)
    milvus.close()


def _main_cli() -> None:
    limit_env = os.getenv("INDEX_LIMIT")
    limit = int(limit_env) if limit_env else None
    main(limit=limit)


if __name__ == "__main__":
    _main_cli()
