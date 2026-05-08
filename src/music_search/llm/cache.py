"""LLMCache: cache de respostas da LLM em SQLite via KeyValueCache.

Chave: sha1 de (model + prompt_template_id + payload). PROMPT_VERSION em prompts.py
participa da chave para invalidacao automatica quando o template muda.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from music_search._async_http.cache import KeyValueCache

DEFAULT_PATH = Path("data/derived/llm_cache.sqlite")


def _make_key(*, model: str, template_id: str, payload: Any) -> str:
    payload_norm = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    raw = f"{model}::{template_id}::{payload_norm}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class LLMCache:
    """Wrapper sobre KeyValueCache com chaveamento por (model, template, payload)."""

    def __init__(self, path: Path | str = DEFAULT_PATH, *, table: str = "llm_calls"):
        self._kv = KeyValueCache(path, table=table)

    def lookup(self, *, model: str, template_id: str, payload: Any) -> Any:
        """Retorna o payload cacheado ou None."""
        key = _make_key(model=model, template_id=template_id, payload=payload)
        row = self._kv.get(key)
        if row is None or row["status"] != "hit":
            return None
        return row.get("payload")

    def store(
        self,
        *,
        model: str,
        template_id: str,
        payload: Any,
        result: Any,
        status: str = "hit",
        error: str | None = None,
    ) -> None:
        key = _make_key(model=model, template_id=template_id, payload=payload)
        self._kv.upsert(
            key=key,
            kind=template_id,
            status=status,
            source=model,
            payload=result,
            error=error,
        )

    def stats(self) -> dict[str, int]:
        return self._kv.stats()

    def close(self) -> None:
        self._kv.close()
