"""
src/music_search/preprocessing.py
==================================
Pré-processamento textual centralizado.

Usado pelo indexer.py (BM25) e, futuramente, pelo embedder da Parte 2.
Garante que query e documentos passem pelo mesmo pipeline.

Dependências:
    pip install nltk
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
"""

from __future__ import annotations

import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download silencioso dos recursos (só faz efeito na primeira execução)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_STOPWORDS_NLTK = set(stopwords.words("english"))

_STOPWORDS_MUSICAIS = {
    # Colaborações
    "feat", "ft", "featuring", "with", "vs", "versus",
    "presents", "pres", "prod", "produced",
    # Versões e edições
    "remix", "remaster", "remastered", "version", "ver",
    "edit", "extended", "radio", "original", "official",
    "explicit", "instrumental", "acoustic", "live",
    "unplugged", "session", "demo", "single", "bonus",
    "deluxe", "edition", "expanded", "anniversary",
    "special", "limited", "collector", "reissue",
    # Estrutura de álbum
    "part", "pt", "vol", "volume", "side", "disc",
    "disk", "cd", "intro", "outro", "interlude",
    "skit", "reprise", "medley",
}

STOPWORDS = _STOPWORDS_NLTK | _STOPWORDS_MUSICAIS

# ---------------------------------------------------------------------------
# Stemmer (instância única — é thread-safe)
# ---------------------------------------------------------------------------

_stemmer = PorterStemmer()

# Pré-compilados para performance
_RE_PARENS  = re.compile(r"[\(\[\{].*?[\)\]\}]")
_RE_NOISE   = re.compile(
    r"\b(feat\.?|ft\.?|featuring|prod\.?|remix|remaster(?:ed)?|"
    r"version|edit|mix|extended|radio|live|acoustic|"
    r"instrumental|deluxe|edition|bonus|anniversary|reissue|"
    r"explicit|official)\b.*",
    flags=re.IGNORECASE,
)
_RE_SPACES  = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def normalizar(texto: str, remover_versoes: bool = True) -> str:
    """
    Normalização textual:
      1. Normaliza encoding (NFC)
      2. Remove espaços Unicode invisíveis
      3. Lowercase
      4. Remove conteúdo entre parênteses/colchetes
      5. Remove marcadores de versão/feat (opcional)
      6. Remove pontuação
      7. Colapsa espaços
    """
    if not isinstance(texto, str):
        return ""

    texto = unicodedata.normalize("NFC", texto)
    texto = texto.strip("\u200b\u00a0\ufeff ")
    texto = texto.lower()
    texto = _RE_PARENS.sub(" ", texto)

    if remover_versoes:
        texto = _RE_NOISE.sub("", texto)

    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = _RE_SPACES.sub(" ", texto).strip()
    return texto


def tokenizar(texto: str) -> list[str]:
    """Tokeniza usando word_tokenize do NLTK e remove tokens de 1 char."""
    if not texto:
        return []
    return [t for t in word_tokenize(texto) if len(t) > 1]


def remover_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords NLTK + domínio musical."""
    return [t for t in tokens if t not in STOPWORDS]


def aplicar_stemming(tokens: list[str]) -> list[str]:
    """Aplica PorterStemmer em cada token."""
    return [_stemmer.stem(t) for t in tokens]


def preprocessar_tokens(
    texto:        str,
    use_stemming: bool = True,
) -> list[str]:
    """
    Pipeline completo: normalização → tokenização → stopwords → stemming.

    É a função principal chamada pelo indexer (documentos e queries).
    Ambos DEVEM usar os mesmos parâmetros para garantir compatibilidade.

    Parâmetros
    ----------
    texto        : texto bruto (track_name, artist, album concatenados)
    use_stemming : aplica PorterStemmer (recomendado para BM25)

    Retorno
    -------
    Lista de tokens prontos para BM25Okapi.
    """
    texto   = normalizar(texto)
    tokens  = tokenizar(texto)
    tokens  = remover_stopwords(tokens)

    if use_stemming:
        tokens = aplicar_stemming(tokens)

    return tokens