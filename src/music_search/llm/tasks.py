"""Tarefas alto-nivel sobre a LLM NIM, com cache transparente.

Funcoes:
- extract_artist_json / extract_album_json / extract_genre_json / extract_composer_json:
  recebem HTML cru e devolvem dict com schema definido em prompts.py.
- classify_intent: recebe query string e devolve um Literal de intent.
- rerank: recebe query + candidatos + top_k e devolve a lista reordenada.

Todas usam LLMCache transparente — repetir a mesma chamada nao dispara LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from music_search.llm.cache import LLMCache
from music_search.llm.client import NimClient
from music_search.llm.prompts import (
    CLASSIFY_INTENT_SYSTEM,
    EXTRACT_ALBUM_SYSTEM,
    EXTRACT_ARTIST_SYSTEM,
    EXTRACT_COMPOSER_SYSTEM,
    EXTRACT_GENRE_SYSTEM,
    PROMPT_VERSION,
    RERANK_SYSTEM,
    classify_intent_user_prompt,
    extract_user_prompt,
    rerank_user_prompt,
)

Intent = Literal["artist", "album", "song", "lyric", "genre", "none"]
_VALID_INTENTS: set[str] = {"artist", "album", "song", "lyric", "genre", "none"}


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` ou ``` ... ``` se a LLM emitir mesmo com response_format."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(text: str) -> Any:
    return json.loads(_strip_code_fences(text))


async def _chat_json(
    client: NimClient,
    *,
    system: str,
    user: str,
    model: str,
    template_id: str,
    cache: LLMCache | None,
    cache_payload: Any,
    max_tokens: int | None = None,
) -> Any:
    """Helper: envia chat com response_format=json_object e parseia resposta."""
    if cache is not None:
        cached = cache.lookup(model=model, template_id=template_id, payload=cache_payload)
        if cached is not None:
            return cached
    raw = await client.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    parsed = _parse_json(raw)
    if cache is not None:
        cache.store(model=model, template_id=template_id, payload=cache_payload, result=parsed)
    return parsed


# ---------- Extracao ----------


async def _extract_entity(
    *,
    html: str,
    source_url: str | None,
    system: str,
    template_id: str,
    client: NimClient,
    cache: LLMCache | None,
) -> dict:
    user = extract_user_prompt(html, source_url=source_url)
    payload = {"v": PROMPT_VERSION, "src": source_url, "html_sha": _hash(html)}
    return await _chat_json(
        client,
        system=system,
        user=user,
        model=client.cfg.model_extract,
        template_id=template_id,
        cache=cache,
        cache_payload=payload,
        max_tokens=2048,
    )


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


async def extract_artist_json(
    html: str,
    *,
    source_url: str | None = None,
    client: NimClient,
    cache: LLMCache | None = None,
) -> dict:
    return await _extract_entity(
        html=html,
        source_url=source_url,
        system=EXTRACT_ARTIST_SYSTEM,
        template_id=f"extract_artist_{PROMPT_VERSION}",
        client=client,
        cache=cache,
    )


async def extract_album_json(
    html: str,
    *,
    source_url: str | None = None,
    client: NimClient,
    cache: LLMCache | None = None,
) -> dict:
    return await _extract_entity(
        html=html,
        source_url=source_url,
        system=EXTRACT_ALBUM_SYSTEM,
        template_id=f"extract_album_{PROMPT_VERSION}",
        client=client,
        cache=cache,
    )


async def extract_genre_json(
    html: str,
    *,
    source_url: str | None = None,
    client: NimClient,
    cache: LLMCache | None = None,
) -> dict:
    return await _extract_entity(
        html=html,
        source_url=source_url,
        system=EXTRACT_GENRE_SYSTEM,
        template_id=f"extract_genre_{PROMPT_VERSION}",
        client=client,
        cache=cache,
    )


async def extract_composer_json(
    html: str,
    *,
    source_url: str | None = None,
    client: NimClient,
    cache: LLMCache | None = None,
) -> dict:
    return await _extract_entity(
        html=html,
        source_url=source_url,
        system=EXTRACT_COMPOSER_SYSTEM,
        template_id=f"extract_composer_{PROMPT_VERSION}",
        client=client,
        cache=cache,
    )


# ---------- Intent ----------


async def classify_intent(
    query: str,
    *,
    client: NimClient,
    cache: LLMCache | None = None,
) -> Intent:
    parsed = await _chat_json(
        client,
        system=CLASSIFY_INTENT_SYSTEM,
        user=classify_intent_user_prompt(query),
        model=client.cfg.model_intent,
        template_id=f"classify_intent_{PROMPT_VERSION}",
        cache=cache,
        cache_payload={"v": PROMPT_VERSION, "q": query.strip().lower()},
        max_tokens=64,
    )
    intent = str(parsed.get("intent", "none")).lower()
    return intent if intent in _VALID_INTENTS else "none"  # type: ignore[return-value]


# ---------- Reranking ----------


async def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    *,
    client: NimClient,
    cache: LLMCache | None = None,
) -> list[dict]:
    """Reordena `candidates` por relevancia semantica. `candidates[i]` deve ter 'id'.

    Itens nao listados pela LLM mantem ordem original ao final.
    """
    if not candidates:
        return []
    cap = min(len(candidates), 50)
    head = candidates[:cap]
    parsed = await _chat_json(
        client,
        system=RERANK_SYSTEM,
        user=rerank_user_prompt(query, head, top_k),
        model=client.cfg.model_rerank,
        template_id=f"rerank_{PROMPT_VERSION}",
        cache=cache,
        cache_payload={
            "v": PROMPT_VERSION,
            "q": query.strip().lower(),
            "ids": [c["id"] for c in head],
        },
        max_tokens=512,
    )
    order = parsed.get("order") or []
    by_id = {c["id"]: c for c in head}
    seen: set = set()
    out: list[dict] = []
    for cid in order:
        if cid in by_id and cid not in seen:
            out.append(by_id[cid])
            seen.add(cid)
    for c in head:
        if c["id"] not in seen:
            out.append(c)
    return out[:top_k] + candidates[cap:]
