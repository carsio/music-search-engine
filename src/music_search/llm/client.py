"""Cliente HTTP para uma API NIM/OpenAI-compativel.

Encapsula `POST /chat/completions` com retries, rate limit e timeout. A camada
de cache vive em `cache.py` e e composta separadamente em `tasks.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from music_search._async_http.throttle import AsyncRateLimiter

DEFAULT_BASE_URL = ""
DEFAULT_TIMEOUT = 60.0


@dataclass
class NimConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    rate_per_sec: float = 2.0
    model_extract: str = "nim-coder"
    model_intent: str = "nim-gemma-4-31b-it"
    model_rerank: str = "nim-qwen3-next-80b"

    @classmethod
    def from_env(cls) -> NimConfig:
        return cls(
            base_url=os.environ.get("NIM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=os.environ.get("NIM_API_KEY"),
            timeout=float(os.environ.get("NIM_TIMEOUT", DEFAULT_TIMEOUT)),
            rate_per_sec=float(os.environ.get("NIM_RATE", "2.0")),
            model_extract=os.environ.get("NIM_MODEL_EXTRACT", "nim-coder"),
            model_intent=os.environ.get("NIM_MODEL_INTENT", "nim-gemma-4-31b-it"),
            model_rerank=os.environ.get("NIM_MODEL_RERANK", "nim-qwen3-next-80b"),
        )


class NimClient:
    """Cliente assincrono. Use como context manager para garantir close."""

    def __init__(self, cfg: NimConfig | None = None, client: httpx.AsyncClient | None = None):
        self.cfg = cfg or NimConfig.from_env()
        self._owned_client = client is None
        if client is None:
            if not self.cfg.base_url:
                raise RuntimeError(
                    "NIM_BASE_URL nao configurado. Defina um endpoint OpenAI-compativel "
                    "ou use .env.example como referencia."
                )
            headers: dict[str, str] = {
                "User-Agent": "music-search-engine/0.1 (UFAM-ICC222)",
            }
            if self.cfg.api_key:
                headers["Authorization"] = f"Bearer {self.cfg.api_key}"
            client = httpx.AsyncClient(
                base_url=self.cfg.base_url,
                timeout=httpx.Timeout(self.cfg.timeout, connect=10.0),
                headers=headers,
            )
        self.client = client
        self.limiter = AsyncRateLimiter(rate=self.cfg.rate_per_sec)

    async def __aenter__(self) -> NimClient:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owned_client:
            await self.client.aclose()

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Envia chat completion e retorna a string da primeira choice."""
        await self.limiter.acquire()
        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format is not None:
            body["response_format"] = response_format
        resp = await self.client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"resposta invalida da NIM: {data!r}") from exc
