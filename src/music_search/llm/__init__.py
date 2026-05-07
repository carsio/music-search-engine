"""Cliente para LLM NIM/OpenAI-compativel com cache de respostas.

Uso:
    from music_search.llm import NimClient, classify_intent, rerank
    from music_search.llm import extract_artist_json

A API e OpenAI-compativel (`POST /v1/chat/completions`). Configure via env:
- NIM_BASE_URL
- NIM_API_KEY
- NIM_MODEL_EXTRACT  (default: nim-coder)
- NIM_MODEL_INTENT   (default: nim-gemma-4-31b-it)
- NIM_MODEL_RERANK   (default: nim-qwen3-next-80b)
"""

from music_search.llm.cache import LLMCache
from music_search.llm.client import NimClient
from music_search.llm.tasks import (
    classify_intent,
    extract_album_json,
    extract_artist_json,
    extract_composer_json,
    extract_genre_json,
    rerank,
)

__all__ = [
    "LLMCache",
    "NimClient",
    "classify_intent",
    "extract_album_json",
    "extract_artist_json",
    "extract_composer_json",
    "extract_genre_json",
    "rerank",
]
