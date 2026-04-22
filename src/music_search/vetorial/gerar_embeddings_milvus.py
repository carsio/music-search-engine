"""
Geração de embeddings do dataset Spotify e armazenamento no Milvus.

Modos de embedding (configure em CONFIGURAÇÃO abaixo):
  - OpenAI API direta : modelo text-embedding-3-small (1536 dims)
                        Requer variável de ambiente: OPENAI_API_KEY
  - Ollama local      : modelo nomic-embed-text      (768 dims)
                        text-embedding-3-small é proprietário e NÃO está
                        disponível no Ollama. Use nomic-embed-text como
                        alternativa open-source de qualidade equivalente.
                        Instale com: ollama pull nomic-embed-text

Milvus:
  - Padrão: Milvus Lite (arquivo local .db, sem Docker necessário)
  - Remoto: defina MILVUS_URI como "http://localhost:19530"

Uso:
  # Com OpenAI API:
  OPENAI_API_KEY=sk-... python gerar_embeddings_milvus.py

  # Com Ollama local (certifique-se que `ollama serve` está rodando):
  USE_OLLAMA=true python gerar_embeddings_milvus.py
"""

import os
import json
import time
import sys
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIError
from pymilvus import MilvusClient, DataType

# ── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
CSV_PATH       = BASE_DIR / "spotify_unificado.csv"
CHECKPOINT_FILE = BASE_DIR / "embedding_checkpoint.txt"  # guarda último offset processado

COLLECTION_NAME = "spotify_tracks"
BATCH_SIZE      = 100    # registros por chamada de embedding (max recomendado: 2048 tokens/item)
CHUNK_SIZE      = 5_000  # linhas lidas do CSV por vez (controla uso de memória)

# Milvus URI: lê da variável de ambiente ou usa Milvus Lite local como padrão
MILVUS_URI = os.getenv("MILVUS_URI", str(BASE_DIR / "milvus_spotify.db"))

# Embedding
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

if USE_OLLAMA:
    EMBED_CLIENT = OpenAI(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # Ollama não valida a chave
    )
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    EMBED_DIM   = 768
else:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Erro: defina OPENAI_API_KEY ou use USE_OLLAMA=true")
    EMBED_CLIENT = OpenAI(api_key=api_key)
    EMBED_MODEL  = "text-embedding-3-small"
    EMBED_DIM    = 1536

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "embedding_log.txt", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def row_to_text(row: dict) -> str:
    """Constrói o texto que será transformado em embedding."""
    parts = [
        f"track: {row.get('track_name') or ''}",
        f"artists: {row.get('artist_names') or ''}",
        f"album: {row.get('album_name') or ''}",
        f"genres: {row.get('artist_genres') or ''}",
        f"album_type: {row.get('album_type') or ''}",
        f"label: {row.get('label') or ''}",
        f"release_date: {row.get('release_date') or ''}",
        f"explicit: {row.get('explicit', False)}",
        f"track_popularity: {row.get('track_popularity', 0)}",
        f"album_popularity: {row.get('album_popularity', 0)}",
        f"duration_ms: {row.get('duration_ms', 0)}",
    ]
    return "; ".join(p for p in parts if not p.endswith(": "))


def row_to_json(row: dict) -> str:
    """Serializa o registro completo como JSON compacto (para armazenamento)."""
    return json.dumps(row, ensure_ascii=False, default=str)


def embed_batch(texts: list[str], retries: int = 5) -> list[list[float]]:
    """Chama a API de embedding com retry em caso de rate limit."""
    for attempt in range(retries):
        try:
            resp = EMBED_CLIENT.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except RateLimitError:
            wait = 2 ** attempt
            log.warning(f"Rate limit — aguardando {wait}s (tentativa {attempt + 1}/{retries})")
            time.sleep(wait)
        except APIError as e:
            log.error(f"Erro da API: {e}")
            raise
    raise RuntimeError("Número máximo de tentativas atingido")


def truncate(value, max_len: int) -> str:
    """Trunca strings para o limite do campo VARCHAR do Milvus (medido em bytes UTF-8)."""
    s = str(value) if value is not None and not pd.isna(value) else ""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_len:
        return s
    # Corta no limite de bytes sem quebrar caracteres multi-byte
    return encoded[:max_len].decode("utf-8", errors="ignore")


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    try:
        return bool(int(value))
    except (ValueError, TypeError):
        return False

# ── MILVUS: CRIAÇÃO DA COLEÇÃO ────────────────────────────────────────────────

def criar_colecao(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION_NAME):
        log.info(f"Coleção '{COLLECTION_NAME}' já existe — reutilizando.")
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field("id",               DataType.INT64,        is_primary=True)
    schema.add_field("track_name",       DataType.VARCHAR,      max_length=500)
    schema.add_field("artist_names",     DataType.VARCHAR,      max_length=2000)
    schema.add_field("album_name",       DataType.VARCHAR,      max_length=500)
    schema.add_field("artist_genres",    DataType.VARCHAR,      max_length=2000)
    schema.add_field("album_type",       DataType.VARCHAR,      max_length=50)
    schema.add_field("label",            DataType.VARCHAR,      max_length=500)
    schema.add_field("release_date",     DataType.VARCHAR,      max_length=20)
    schema.add_field("track_popularity", DataType.INT64)
    schema.add_field("album_popularity", DataType.INT64)
    schema.add_field("duration_ms",      DataType.INT64)
    schema.add_field("explicit",         DataType.BOOL)
    schema.add_field("json_data",        DataType.VARCHAR,      max_length=65_535)
    schema.add_field("embedding",        DataType.FLOAT_VECTOR, dim=EMBED_DIM)

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
    log.info(f"Coleção '{COLLECTION_NAME}' criada (dim={EMBED_DIM}, métrica=COSINE).")


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def ler_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        return int(CHECKPOINT_FILE.read_text().strip())
    return 0


def salvar_checkpoint(offset: int) -> None:
    CHECKPOINT_FILE.write_text(str(offset))

# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────

def main() -> None:
    log.info(f"Modo embedding : {'Ollama (' + EMBED_MODEL + ')' if USE_OLLAMA else 'OpenAI (' + EMBED_MODEL + ')'}")
    log.info(f"Dimensão       : {EMBED_DIM}")
    log.info(f"Milvus URI     : {MILVUS_URI}")
    log.info(f"CSV            : {CSV_PATH}")

    client = MilvusClient(uri=MILVUS_URI)
    criar_colecao(client)

    inicio_offset = ler_checkpoint()
    if inicio_offset > 0:
        log.info(f"Retomando do offset {inicio_offset:,}")

    total_inseridos = 0
    offset_global = 0

    # Lê o CSV em chunks para controlar memória
    reader = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk_df in reader:
        chunk_end = offset_global + len(chunk_df)

        # Pula chunks já processados
        if chunk_end <= inicio_offset:
            offset_global = chunk_end
            continue

        # Recorta para retomar no meio do chunk, se necessário
        skip_in_chunk = max(0, inicio_offset - offset_global)
        chunk_df = chunk_df.iloc[skip_in_chunk:].reset_index(drop=True)

        rows = chunk_df.to_dict(orient="records")

        for i in tqdm(
            range(0, len(rows), BATCH_SIZE),
            desc=f"Chunk {offset_global // CHUNK_SIZE + 1}",
            unit="batch",
        ):
            batch_rows = rows[i : i + BATCH_SIZE]

            # 1. Gera textos para embedding
            texts = [row_to_text(r) for r in batch_rows]

            # 2. Gera embeddings
            embeddings = embed_batch(texts)

            # 3. Monta documentos para o Milvus
            docs = []
            for row, emb in zip(batch_rows, embeddings):
                docs.append({
                    "track_name":       truncate(row.get("track_name"),       500),
                    "artist_names":     truncate(row.get("artist_names"),     2000),
                    "album_name":       truncate(row.get("album_name"),       500),
                    "artist_genres":    truncate(row.get("artist_genres"),    2000),
                    "album_type":       truncate(row.get("album_type"),       50),
                    "label":            truncate(row.get("label"),            500),
                    "release_date":     truncate(row.get("release_date"),     20),
                    "track_popularity": safe_int(row.get("track_popularity")),
                    "album_popularity": safe_int(row.get("album_popularity")),
                    "duration_ms":      safe_int(row.get("duration_ms")),
                    "explicit":         safe_bool(row.get("explicit")),
                    "json_data":        truncate(row_to_json(row),            65_535),
                    "embedding":        emb,
                })

            # 4. Insere no Milvus
            client.insert(collection_name=COLLECTION_NAME, data=docs)
            total_inseridos += len(docs)

        offset_global = chunk_end
        salvar_checkpoint(offset_global)
        log.info(f"Progresso: {offset_global:,} / linhas processadas | inseridos: {total_inseridos:,}")

    log.info(f"Concluído! Total inserido no Milvus: {total_inseridos:,}")
    log.info(f"Arquivo do banco vetorial: {MILVUS_URI}")

    # Remove checkpoint ao finalizar com sucesso
    CHECKPOINT_FILE.unlink(missing_ok=True)

    client.close()


if __name__ == "__main__":
    main()
