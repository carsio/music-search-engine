"""Configuração compartilhada entre indexação e busca vetorial.

Centraliza paths e nomes de recurso para que indexação e busca nunca divirjam
(mesma coleção, mesmo modelo de embedding, mesma dimensão).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Diretório de artefatos da busca vetorial (Milvus Lite, checkpoint, logs).
# Mantido fora de `src/` e do parquet original para facilitar limpeza.
VECTOR_DATA_DIR = Path("data/vector")

COLLECTION_NAME = "spotify_tracks"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Resolve o backend de embedding a partir de variáveis de ambiente.

    Regras (aplicadas na ordem):
    1. Se `USE_OLLAMA=true` (padrão), usa Ollama local.
    2. Senão, se `OPENAI_API_KEY` está definida, usa OpenAI.
    3. Senão, cai no Ollama (comportamento padrão do projeto).

    IMPORTANTE: o modelo usado para indexar deve ser o mesmo usado para buscar —
    dimensões diferentes geram erro no Milvus.
    """

    use_ollama: bool
    model: str
    dim: int
    openai_api_key: str | None
    ollama_url: str

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        use_ollama = os.getenv("USE_OLLAMA", "true").lower() != "false"
        api_key = os.getenv("OPENAI_API_KEY")

        if not use_ollama and api_key:
            return cls(
                use_ollama=False,
                model="text-embedding-3-small",
                dim=1536,
                openai_api_key=api_key,
                ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
            )

        return cls(
            use_ollama=True,
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            dim=768,
            openai_api_key=None,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        )


def default_milvus_uri() -> str:
    """URI padrão do Milvus. Lê `MILVUS_URI` ou usa Milvus Lite local."""
    return os.getenv("MILVUS_URI", str(VECTOR_DATA_DIR / "milvus_spotify.db"))
