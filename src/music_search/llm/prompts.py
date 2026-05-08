"""Templates de prompt em PT-BR. PROMPT_VERSION participa da chave do cache.

Toda mudanca de template incrementa PROMPT_VERSION para invalidar respostas antigas.
"""

from __future__ import annotations

PROMPT_VERSION = "v2"


# ---------- Extracao de entidades a partir de conteudo textual/Wikipedia ----------

EXTRACT_ARTIST_SYSTEM = """Voce e um extrator de dados estruturados sobre artistas musicais
brasileiros. Receba conteudo textual de uma pagina (Wikipedia ou similar) e devolva APENAS um objeto
JSON com os campos especificados, sem comentarios e sem texto extra.

Schema:
{
  "name": string,
  "tagline": string | null,         // 1 frase curta (ate 120 chars) descrevendo o artista
  "bio": string | null,              // 2-4 paragrafos resumindo carreira/estilo
  "genres": string[],                // ate 8 generos
  "origin": string | null,           // cidade/estado/pais
  "year_started": number | null,     // ano de inicio da carreira
  "monthly_listeners": string | null,// formato livre (ex.: "12M") ou null se nao mencionado
  "popularity": number | null,       // 0-100, estimativa subjetiva ou null
  "albums": [{"title": string, "year": number | null, "tracks": number | null}]
}

Se um campo nao puder ser extraido com confianca, use null.
"""

EXTRACT_ALBUM_SYSTEM = """Extraia dados estruturados de um album musical brasileiro.
Devolva APENAS um objeto JSON, sem texto extra.

Schema:
{
  "title": string,
  "artist": string,
  "year": number | null,
  "description": string | null,    // 2-3 paragrafos
  "tracks": [{"position": number | null, "title": string, "duration": string | null}]
}

Use null para campos nao confirmados pelo conteudo.
"""

EXTRACT_GENRE_SYSTEM = """Extraia dados estruturados de um genero musical (pode ser pagina geral
ou de subgenero). Devolva APENAS um objeto JSON.

Schema:
{
  "name": string,
  "description": string | null,         // 1-3 paragrafos
  "origin": string | null,              // pais/regiao/decada de origem
  "decade": string | null,              // ex.: "1970s"
  "representative_artists": string[],   // ate 12 artistas
  "related_genres": string[]            // ate 8 generos relacionados
}
"""

EXTRACT_COMPOSER_SYSTEM = """Extraia dados estruturados de um compositor / letrista brasileiro.
Devolva APENAS um objeto JSON.

Schema:
{
  "name": string,
  "bio": string | null,
  "genres": string[],
  "origin": string | null,
  "year_started": number | null,
  "notable_works": [{"title": string, "year": number | null, "performer": string | null}]
}
"""


def extract_user_prompt(source_text: str, *, source_url: str | None = None) -> str:
    """Compoe a parte user com o conteudo truncado (LLM tem janela limitada)."""
    truncated = source_text[:35000]
    suffix = "" if len(source_text) <= 35000 else "\n[...truncado...]"
    src = f"\n\nFonte: {source_url}" if source_url else ""
    return f"Conteudo da fonte:\n{truncated}{suffix}{src}"


# ---------- Classificacao de intent da query ----------

CLASSIFY_INTENT_SYSTEM = """Voce classifica queries de busca de musica em PT-BR.
Devolva APENAS um JSON com a forma {"intent": "<categoria>"} sem texto extra.

Categorias:
- "artist": nome de artista (ex.: "anitta", "caetano veloso", "djavan")
- "album": titulo de album, geralmente prefixado por "album", "disco" (ex.: "album acabou chorare")
- "song": titulo de musica especifica (ex.: "envolver", "garota de ipanema")
- "lyric": trecho de letra ou multipalavras tipicas de letra (ex.: "amor que dorme em mim",
  "andar com fe eu vou", "garota de ipanema")
- "genre": nome de genero (ex.: "samba", "bossa nova", "sertanejo universitario")
- "none": ambiguo demais ou irrelevante

Se hesitar entre "song" e "lyric", prefira "lyric" quando a query tem >2 palavras
descrevendo conteudo. Prefira "artist" se for so 1-3 palavras que parecem nome.
"""


def classify_intent_user_prompt(query: str) -> str:
    return f'Query: "{query}"\nResponda apenas com {{"intent": "..."}}'


# ---------- Reranking ----------

RERANK_SYSTEM = """Voce e um reordenador semantico de resultados de busca de musica.
Receba uma query e uma lista de candidatos (cada um com id, titulo, artista e snippet).
Devolva APENAS um JSON com a forma:
  {"order": [<id1>, <id2>, ...]}
listando os ids do candidato MAIS relevante para o MENOS, sem texto extra.
Use seu conhecimento do contexto musical brasileiro para desempatar.
"""


def rerank_user_prompt(query: str, candidates: list[dict], top_k: int) -> str:
    items = []
    for c in candidates:
        items.append(
            {
                "id": c["id"],
                "title": c.get("title") or c.get("track_name") or "",
                "artist": c.get("artist") or c.get("primary_artist_name") or "",
                "snippet": (c.get("snippet") or c.get("preview") or "")[:300],
            }
        )
    return (
        f'Query: "{query}"\n'
        f"Top-{top_k} candidatos a reordenar:\n"
        f"{items}\n"
        'Responda com {"order": [...ids...]}'
    )
