"""Protocolo de fonte de letras + tipos compartilhados."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Status(StrEnum):
    HIT = "hit"  # letra encontrada
    MISS = "miss"  # busca completou mas nao achou (404 / not found)
    ERROR = "error"  # erro temporario (timeout, 5xx) — re-tentavel
    BLOCKED = "blocked"  # rate limit / WAF — re-tentavel mas com cuidado


@dataclass
class LyricsResult:
    status: Status
    lyrics: str | None = None
    source: str | None = None
    source_url: str | None = None
    error: str | None = None


@runtime_checkable
class LyricsSource(Protocol):
    """Cada fonte (lyrics.ovh, vagalume, genius...) implementa esse protocolo."""

    name: str

    async def fetch(self, artist: str, title: str) -> LyricsResult: ...
