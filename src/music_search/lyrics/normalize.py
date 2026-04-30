"""Normalizacao de titulo de faixa e nome de artista para matching com APIs de letras."""
# ruff: noqa: RUF001

from __future__ import annotations

import re
import unicodedata

_FEAT_TAIL = re.compile(
    r"\s*[\(\[]?\s*(feat\.?|ft\.?|featuring|com\s+|with\s+)\s*[^\)\]]*[\)\]]?\s*$",
    re.IGNORECASE,
)

_VERSION_TAIL = re.compile(
    r"\s*-\s*("
    r"ao\s+vivo|live|en\s+vivo|"
    r"remix|edit|radio\s+edit|extended|"
    r"acoustic|ac[uú]stic[ao]|"
    r"instrumental|demo|"
    r"remaster(ed)?|"
    r"vers[aã]o.*|"
    r"slowed(\s+(\+|and|&)\s+reverb)?|sped\s*up|"
    r"cover|original\s+mix|club\s+mix|deluxe"
    r").*$",
    re.IGNORECASE,
)

_PAREN_TAGS = re.compile(
    r"\s*[\(\[]\s*("
    r"ao\s+vivo|live|en\s+vivo|"
    r"remix|edit|radio\s+edit|extended|"
    r"acoustic|ac[uú]stic[ao]|"
    r"instrumental|demo|"
    r"remaster(ed)?|"
    r"slowed(\s+(\+|and|&)\s+reverb)?|sped\s*up|"
    r"cover|original\s+mix|club\s+mix|deluxe|bonus\s+track"
    r")\s*[\)\]]\s*",
    re.IGNORECASE,
)

_ARTIST_SPLIT = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring|with|com)\s+|\s*\|\s*|\s*&\s*|,\s*",
    re.IGNORECASE,
)


def normalize_title(title: str | None) -> str:
    """Remove sufixos como 'feat. X', '- Ao Vivo', '(Remix)', '[Live]'."""
    if not title:
        return ""
    cleaned = title
    for _ in range(3):
        new = _PAREN_TAGS.sub(" ", cleaned)
        new = _VERSION_TAIL.sub("", new)
        new = _FEAT_TAIL.sub("", new)
        new = re.sub(r"\s+", " ", new)
        new = new.strip(" -–—")  # ASCII hyphen + en/em dash
        if new == cleaned:
            break
        cleaned = new
    return cleaned.strip()


def normalize_artist(artist: str | None) -> str:
    """Pega o artista primario quando ha colaboracao."""
    if not artist:
        return ""
    primary = _ARTIST_SPLIT.split(artist, maxsplit=1)[0]
    return primary.strip()


def slugify(text: str) -> str:
    """Slug ASCII para matching e URLs."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
