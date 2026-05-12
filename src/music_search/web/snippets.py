"""Extracao de snippets (linhas de letra contendo termos da query)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from music_search.core.preprocessing import preprocess


def query_terms(query: str) -> set[str]:
    """Tokens preprocessados da query, util para destacar matches."""
    return {t for t in preprocess(query) if t}


def _line_score(line_tokens: list[str], query_tokens: set[str]) -> int:
    return sum(1 for t in line_tokens if t in query_tokens)


def extract_snippets(
    lyrics: str,
    query: str,
    *,
    max_snippets: int = 3,
    context: int = 0,
) -> list[dict]:
    """Devolve as melhores linhas (com seu numero) que contem termos da query.

    `context=1` inclui +-1 linha vizinha (concatenadas).
    Linhas vazias sao ignoradas.
    """
    if not lyrics:
        return []
    qtokens = query_terms(query)
    if not qtokens:
        return []

    raw_lines = [ln.rstrip() for ln in lyrics.splitlines()]
    line_tokens = [preprocess(ln) for ln in raw_lines]

    scored: list[tuple[int, int, int]] = []  # (-score, line_no, idx)
    for i, tokens in enumerate(line_tokens):
        if not raw_lines[i].strip():
            continue
        s = _line_score(tokens, qtokens)
        if s > 0:
            scored.append((-s, i + 1, i))

    scored.sort()
    out: list[dict] = []
    seen_idx: set[int] = set()
    for _neg_score, line_no, idx in scored[: max_snippets * 2]:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        if context > 0:
            start = max(0, idx - context)
            end = min(len(raw_lines), idx + context + 1)
            text = " / ".join(
                raw_lines[j].strip() for j in range(start, end) if raw_lines[j].strip()
            )
        else:
            text = raw_lines[idx].strip()
        out.append({"line": line_no, "text": text})
        if len(out) >= max_snippets:
            break
    return out


def highlight_terms(text: str, query: str) -> str:
    """Destaca termos da query com <mark>...</mark>. Case-insensitive, palavras inteiras."""
    if not text or not query:
        return text
    raw_terms = [t for t in re.split(r"\s+", query.strip()) if t]
    if not raw_terms:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in raw_terms) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", text)


def make_lyric_snippets(
    docs: Iterable,
    query: str,
    *,
    max_snippets: int = 3,
) -> dict[str, list[dict]]:
    """Mapa doc_id -> lista de snippets, para uma colecao de hits."""
    out: dict[str, list[dict]] = {}
    for doc in docs:
        doc_id = doc.id if hasattr(doc, "id") else doc.get("id")
        lyrics = doc.lyrics if hasattr(doc, "lyrics") else doc.get("lyrics", "")
        out[doc_id] = extract_snippets(lyrics, query, max_snippets=max_snippets)
    return out
