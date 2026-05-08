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


_PUNCT_TAIL = re.compile(r"[\s\-–—:;,.!?]+$")
_ALL_PARENS = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")

# Caracteres especiais a remover na normalizacao agressiva. Mantem letras (com acento),
# numeros, espaco e apostrofo (que aparece em titulos como "I'm", "L'amour").
_SPECIAL_CHARS = re.compile(r"[^\w\s']", re.UNICODE)
# Tags de versao "seguras" — palavras que isoladas ja indicam variante de gravacao
# e dificilmente fazem parte do nome real da musica. "Live" e "ao vivo" ficam fora
# para nao mutilar titulos legitimos (ex: "Live and Let Die"); eles sao tratados
# por _VERSION_CONTEXT que exige delimitador.
_VERSION_ANYWHERE = re.compile(
    r"(?:[\(\[\{]\s*)?\b("
    r"remix|radio\s+edit|extended(?:\s+mix)?|"
    r"ac[uú]stic[ao]|acoustic|unplugged|"
    r"instrumental|karaoke|demo|"
    r"remaster(?:ed)?(?:\s+\d{4})?|"
    r"vers[aã]o[^\)\]\}\-]*|"
    r"slowed(?:\s+(?:\+|and|&)\s+reverb)?|sped\s*up|"
    r"original\s+mix|club\s+mix|deluxe|bonus\s+track|"
    r"single\s+version|album\s+version|mono\s+version|stereo\s+version"
    r")\b[^\)\]\}]*?(?:\s*[\)\]\}])?",
    re.IGNORECASE,
)
# Tags que exigem contexto pra nao casar palavras legitimas do titulo:
# - precedidas por inicio/parens/colchetes/chaves/hifen/barra; ou
# - dentro de parens/colchetes/chaves explicitos.
_VERSION_CONTEXT = re.compile(
    r"(?:[\(\[\{]\s*|(?:^|[\s])[\-–—/]\s*)("
    r"live(?:\s+at[^\)\]\}]*|\s+in[^\)\]\}]*|\s+from[^\)\]\}]*)?|"
    r"ao\s+vivo|en\s+vivo|"
    r"cover|edit|edited|explicit|clean"
    r")\b[^\)\]\}]*?(?:\s*[\)\]\}])?",
    re.IGNORECASE,
)
# "Pt. 2", "Part II", "Vol. 1" — costumam ser ruido pra letras.
_PART_MARKER = re.compile(
    r"\s*[\-–—,]?\s*\b(?:pt\.?|part|vol\.?|volume)\s*[ivxlcdm0-9]+\b\s*",
    re.IGNORECASE,
)


def aggressive_normalize(title: str | None) -> str:
    """Normalizacao mais agressiva: remove tags de versao em qualquer posicao,
    remove caracteres especiais, colapsa espacos.

    Diferente de `normalize_title`, que so corta sufixos/parens conhecidos, esta
    funcao trata o titulo como um saco de palavras e remove qualquer ocorrencia
    de "ao vivo", "remix", etc., onde quer que apareca, alem de remover pontuacao.
    Use como variante para retry quando o match exato falhou.
    """
    if not title:
        return ""
    cleaned = normalize_title(title)
    # remove tags "seguras" (remix, instrumental, ...) em qualquer posicao
    cleaned = _VERSION_ANYWHERE.sub(" ", cleaned)
    # remove tags ambiguas (live, cover, ...) so quando ha contexto delimitador
    cleaned = _VERSION_CONTEXT.sub(" ", cleaned)
    # remove "Pt 2", "Part II"
    cleaned = _PART_MARKER.sub(" ", cleaned)
    # remove parens/colchetes/chaves residuais (junto com o conteudo)
    cleaned = _ALL_PARENS.sub(" ", cleaned)
    # remove caracteres especiais (mantem unicode letters/digits/space/apos)
    cleaned = _SPECIAL_CHARS.sub(" ", cleaned)
    # colapsa espacos
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—'")
    return cleaned


def title_variants(title: str | None) -> list[str]:
    """Gera variantes progressivamente mais agressivas do titulo para retry em MISS.

    A ideia: se o casamento exato falhar (ex: catalogo da fonte tem o titulo escrito
    sem o '(Remastered 2011)' ou sem o '/ Subtitulo'), a gente tenta versoes cada vez
    mais simples antes de desistir. Ordem do mais especifico para o mais generico,
    sem repeticoes:

    1. titulo normalizado (com tags ja removidas por `normalize_title`);
    2. sem qualquer parens/colchetes/chaves;
    3. parte antes do primeiro " - " / " / " / " : " / " | ";
    4. versao ASCII pura (decompoe acentos), util para sites que normalizam slug;
    5. agressiva: tags de versao removidas em qualquer posicao + sem caracteres
       especiais (cobre "Song *Live*", "Song! (Remastered)", "Song / Pt. 2");
    6. agressiva + ASCII (combinacao das duas anteriores).
    """
    if not title:
        return []
    base = normalize_title(title)
    if not base:
        return []

    candidates: list[str] = [base]

    def _add(value: str) -> None:
        v = (value or "").strip()
        if not v:
            return
        # comparacao case-insensitive para evitar duplicatas tipo "foo" vs "Foo"
        existing = {c.lower() for c in candidates}
        if v.lower() not in existing:
            candidates.append(v)

    # 2. sem parens
    no_parens = _ALL_PARENS.sub(" ", base)
    no_parens = re.sub(r"\s+", " ", no_parens)
    no_parens = _PUNCT_TAIL.sub("", no_parens).strip()
    _add(no_parens)

    # 3. cortes em separadores comuns
    for sep in (" - ", " / ", " : ", " | "):
        head = base.split(sep, 1)[0].strip()
        _add(head)

    # 4. ASCII puro
    ascii_v = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    ascii_v = re.sub(r"\s+", " ", ascii_v).strip()
    _add(ascii_v)

    # 5. normalizacao agressiva
    aggressive = aggressive_normalize(title)
    _add(aggressive)

    # 6. agressiva + ASCII
    aggressive_ascii = (
        unicodedata.normalize("NFKD", aggressive).encode("ascii", "ignore").decode("ascii")
    )
    aggressive_ascii = re.sub(r"\s+", " ", aggressive_ascii).strip()
    _add(aggressive_ascii)

    return candidates
