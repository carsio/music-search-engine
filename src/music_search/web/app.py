"""API FastAPI para a UI: rotea queries por intent, retorna painel + lista de hits.

Endpoints:
- GET /healthz                              status simples
- GET /search?q=&top=10&algorithm=bm25      busca generica com classificacao de intent
- GET /search/lyric?q=&top=20               busca exclusivamente em letras com snippets
- GET /artist/{id}                          knowledge panel de artista
- GET /album/{id}                           pagina de album derivada do dataset atual
- GET /song/{id}                            pagina de musica com letra
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from music_search.multi_index import MultiEntityIndex
from music_search.ranking import TfScheme
from music_search.search import SparseSearchEngine, load_or_build_default_engine
from music_search.search_tuning import SearchProfile
from music_search.web.schemas import (
    AlbumArtistSummary,
    AlbumRef,
    AlbumResponse,
    AlbumTrack,
    ArtistResponse,
    LyricMatch,
    LyricSearchResponse,
    LyricSnippet,
    SearchResponse,
    SearchResultItem,
    SongResponse,
    TrackRef,
)
from music_search.web.snippets import extract_snippets

logger = logging.getLogger(__name__)

# LLM e opcional. Se NIM_API_KEY nao estiver setado, intent classification e rerank
# silenciosamente caem para fallback heuristico.
_NIM_AVAILABLE = bool(os.environ.get("NIM_API_KEY"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando motor de busca esparso...")
    track_engine = load_or_build_default_engine()
    logger.info("Indice de tracks: %d docs", track_engine.num_docs)

    multi = MultiEntityIndex.from_parquets(track_engine=track_engine)
    logger.info(
        "Entidades carregadas: %s",
        {k: v.num_docs for k, v in multi.entity_indexes.items()},
    )

    nim_client = None
    llm_cache = None
    if _NIM_AVAILABLE:
        from music_search.llm.cache import LLMCache
        from music_search.llm.client import NimClient

        nim_client = NimClient()
        llm_cache = LLMCache()
        logger.info(
            "Cliente NIM ativo (modelos: extract=%s, intent=%s, rerank=%s)",
            nim_client.cfg.model_extract,
            nim_client.cfg.model_intent,
            nim_client.cfg.model_rerank,
        )
    else:
        logger.info("NIM_API_KEY ausente — usando fallback heuristico para intent.")

    app.state.track_engine = track_engine
    app.state.multi = multi
    app.state.nim_client = nim_client
    app.state.llm_cache = llm_cache
    try:
        yield
    finally:
        if nim_client is not None:
            await nim_client.aclose()
        if llm_cache is not None:
            llm_cache.close()


app = FastAPI(
    title="Music Search Engine",
    description="Buscador de musicas brasileiras (BM25/TF-IDF + LLM rerank).",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# -------------------- helpers --------------------


_LYRIC_HINT_TOKENS = {
    "letra",
    "lyrics",
    "verso",
    "que",
    "quando",
    "como",
    "porque",
    "para",
}
_GENRE_TOKENS = {
    "samba",
    "bossa",
    "sertanejo",
    "funk",
    "pagode",
    "forro",
    "rap",
    "trap",
    "rock",
    "mpb",
    "axé",
    "axe",
    "gospel",
    "reggae",
}


def heuristic_intent(query: str) -> str:
    q = query.strip().lower()
    if not q:
        return "none"
    words = q.split()
    if len(words) == 1 and q in _GENRE_TOKENS:
        return "genre"
    if any(g in words for g in _GENRE_TOKENS) and len(words) <= 3:
        return "genre"
    if q.startswith(("album ", "disco ")):
        return "album"
    if q.startswith(("letra ", "lyrics ")):
        return "lyric"
    if any(t in words for t in _LYRIC_HINT_TOKENS) and len(words) > 3:
        return "lyric"
    if 1 <= len(words) <= 3:
        return "artist"
    return "lyric"


async def classify_intent_with_fallback(query: str, app: FastAPI) -> tuple[str, bool]:
    """Retorna (intent, used_llm)."""
    if app.state.nim_client is None:
        return heuristic_intent(query), False
    try:
        from music_search.llm.tasks import classify_intent

        intent = await classify_intent(
            query, client=app.state.nim_client, cache=app.state.llm_cache
        )
        return intent, True
    except Exception as exc:
        logger.warning("classify_intent fallback: %s", exc)
        return heuristic_intent(query), False


def hit_to_item(hit: dict[str, Any]) -> SearchResultItem:
    """Converte hit generico (de MultiEntityIndex.search_routed) em SearchResultItem."""
    payload = hit.get("payload") or {}
    intent = hit.get("kind") or hit.get("intent") or "track"
    if intent == "track":
        title = payload.get("track_name") or hit.get("track_name") or ""
        subtitle = payload.get("artist_names") or payload.get("primary_artist_name") or ""
        snippet = payload.get("lyrics_preview") or hit.get("lyrics_preview") or ""
    elif intent == "artist":
        title = payload.get("name", "")
        subtitle = payload.get("tagline") or ", ".join(payload.get("genres", [])[:3])
        snippet = (payload.get("bio") or "")[:240]
    elif intent == "album":
        title = payload.get("title", "")
        subtitle = payload.get("artist", "")
        snippet = (payload.get("description") or "")[:240]
    elif intent == "genre":
        title = payload.get("name", "")
        subtitle = payload.get("origin") or ""
        snippet = (payload.get("description") or "")[:240]
    else:
        title = str(payload.get("name") or payload.get("title") or "")
        subtitle = ""
        snippet = ""
    return SearchResultItem(
        id=str(hit.get("id", "")),
        rank=int(hit.get("rank", 0)),
        score=float(hit.get("score", 0.0)),
        intent=intent,
        title=title,
        subtitle=subtitle or None,
        snippet=snippet or None,
        payload=payload if intent != "track" else _slim_track_payload(hit, payload),
    )


def _slim_track_payload(hit: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Track hits ja vem ricos via SearchHit.to_dict; retornamos um subset enxuto."""
    keys = (
        "track_name",
        "primary_artist_name",
        "artist_names",
        "artist_genres",
        "macro_genre",
        "album_name",
        "release_year",
        "lyrics_preview",
        "lyrics_source",
        "lyrics_source_url",
    )
    src = payload or hit
    return {k: src.get(k) for k in keys if k in src}


# -------------------- endpoints --------------------


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    tracks = app.state.track_engine.num_docs if hasattr(app.state, "track_engine") else 0
    entities = (
        {k: v.num_docs for k, v in app.state.multi.entity_indexes.items()}
        if hasattr(app.state, "multi")
        else {}
    )
    return {"ok": True, "tracks_indexed": tracks, "entities": entities, "llm": _NIM_AVAILABLE}


@app.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=1, max_length=200)],
    top: Annotated[int, Query(ge=1, le=50)] = 10,
    algorithm: Annotated[str, Query(pattern="^(bm25|tfidf)$")] = "bm25",
    rerank: bool = False,
    profile: Annotated[SearchProfile, Query()] = "balanced",
    bm25_k1: Annotated[float, Query(gt=0.0, le=3.0)] = 1.5,
    bm25_b: Annotated[float, Query(ge=0.0, le=1.0)] = 0.75,
    tf_scheme: Annotated[TfScheme, Query()] = "log",
) -> SearchResponse:
    started = time.time()
    intent, _used_llm = await classify_intent_with_fallback(q, app)

    multi: MultiEntityIndex = app.state.multi
    routed = multi.search_routed(
        q,
        intent,
        algorithm=algorithm,  # type: ignore[arg-type]
        top_k=top,
        profile=profile,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        tf_scheme=tf_scheme,
    )
    intent_used = routed["intent_used"]

    # Para tracks, o hit vem do SparseSearchEngine ja como dict completo —
    # injetamos um campo `kind` para o conversor.
    hits = []
    for h in routed["hits"]:
        if intent_used == "track":
            h = {**h, "kind": "track"}
        hits.append(hit_to_item(h))

    rerank_used = False
    if rerank and app.state.nim_client is not None and hits:
        try:
            from music_search.llm.tasks import rerank as llm_rerank

            payload_for_rerank = [
                {
                    "id": h.id,
                    "title": h.title,
                    "artist": h.subtitle or "",
                    "snippet": h.snippet or "",
                }
                for h in hits
            ]
            ordered = await llm_rerank(
                q,
                payload_for_rerank,
                top,
                client=app.state.nim_client,
                cache=app.state.llm_cache,
            )
            order = {item["id"]: i for i, item in enumerate(ordered)}
            hits = sorted(hits, key=lambda h: order.get(h.id, 1_000))
            for new_rank, h in enumerate(hits, start=1):
                h.rank = new_rank
            rerank_used = True
        except Exception as exc:
            logger.warning("rerank fallback: %s", exc)

    elapsed = int((time.time() - started) * 1000)
    return SearchResponse(
        query=q,
        intent_requested=intent,  # type: ignore[arg-type]
        intent_used=intent_used,  # type: ignore[arg-type]
        algorithm=algorithm,  # type: ignore[arg-type]
        items=hits,
        rerank_used=rerank_used,
        elapsed_ms=elapsed,
    )


@app.get("/search/lyric", response_model=LyricSearchResponse)
def search_lyric(
    q: Annotated[str, Query(min_length=1, max_length=200)],
    top: Annotated[int, Query(ge=1, le=50)] = 20,
    algorithm: Annotated[str, Query(pattern="^(bm25|tfidf)$")] = "bm25",
    profile: Annotated[SearchProfile, Query()] = "balanced",
    bm25_k1: Annotated[float, Query(gt=0.0, le=3.0)] = 1.5,
    bm25_b: Annotated[float, Query(ge=0.0, le=1.0)] = 0.75,
    tf_scheme: Annotated[TfScheme, Query()] = "log",
    max_snippets: Annotated[int, Query(ge=1, le=10)] = 3,
) -> LyricSearchResponse:
    started = time.time()
    engine: SparseSearchEngine = app.state.track_engine
    hits = engine.search(
        q,
        algorithm=algorithm,  # type: ignore[arg-type]
        top_k=top,
        profile=profile,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        tf_scheme=tf_scheme,
    )
    matches: list[LyricMatch] = []
    for hit in hits:
        snippets = extract_snippets(hit.lyrics, q, max_snippets=max_snippets)
        matches.append(
            LyricMatch(
                song_id=hit.id,
                title=hit.track_name,
                artist=hit.artist_names or hit.primary_artist_name,
                score=hit.score,
                snippets=[LyricSnippet(line=s["line"], text=s["text"]) for s in snippets],
            )
        )
    return LyricSearchResponse(
        query=q,
        algorithm=algorithm,  # type: ignore[arg-type]
        matches=matches,
        elapsed_ms=int((time.time() - started) * 1000),
    )


@app.get("/artist/{artist_id}", response_model=ArtistResponse)
def get_artist(artist_id: str) -> ArtistResponse:
    multi: MultiEntityIndex = app.state.multi
    idx = multi.entity_indexes.get("artist")
    payload = idx.documents.get(artist_id) if idx else None

    # Top tracks: se tiver track_engine, busca por nome no campo artist_names
    track_engine: SparseSearchEngine = app.state.track_engine
    top_tracks: list[TrackRef] = []
    if payload:
        name = payload.get("name", "")
        if name:
            try:
                hits = track_engine.search(name, algorithm="bm25", top_k=5)
                top_tracks = [
                    TrackRef(title=h.track_name, album=h.album_name or None) for h in hits
                ]
            except Exception as exc:
                logger.warning("top_tracks fail for %s: %s", name, exc)

    if payload is None:
        # Fallback: artista nao enriquecido — derive do parquet de tracks
        try:
            hits = track_engine.search(artist_id.replace("-", " "), algorithm="bm25", top_k=5)
        except Exception:
            hits = []
        if not hits:
            raise HTTPException(404, f"artista {artist_id!r} nao encontrado")
        return ArtistResponse(
            id=artist_id,
            name=hits[0].primary_artist_name or hits[0].artist_names,
            tagline=None,
            bio=None,
            genres=[],
            top_tracks=[TrackRef(title=h.track_name, album=h.album_name or None) for h in hits],
        )

    return ArtistResponse(
        id=str(payload.get("id") or artist_id),
        name=str(payload.get("name") or ""),
        tagline=payload.get("tagline"),
        bio=payload.get("bio"),
        genres=list(payload.get("genres") or []),
        origin=payload.get("origin"),
        year_started=payload.get("year_started"),
        monthly_listeners=payload.get("monthly_listeners"),
        popularity=payload.get("popularity"),
        albums=[AlbumRef(**a) for a in (payload.get("albums") or []) if isinstance(a, dict)],
        top_tracks=top_tracks,
        source=payload.get("source"),
        source_url=payload.get("source_url"),
    )


@app.get("/album/{album_id}", response_model=AlbumResponse)
def get_album(album_id: str) -> AlbumResponse:
    multi: MultiEntityIndex = app.state.multi
    payload = multi.get_album(album_id)
    if payload is None:
        raise HTTPException(404, f"album {album_id!r} nao encontrado")
    return _album_response_from_payload(payload)


@app.get("/song/{song_id}", response_model=SongResponse)
def get_song(song_id: str) -> SongResponse:
    engine: SparseSearchEngine = app.state.track_engine
    doc = engine.documents.get(song_id)
    if doc is None:
        raise HTTPException(404, f"musica {song_id!r} nao encontrada")
    return SongResponse(
        id=song_id,
        title=str(doc.get("track_name") or ""),
        artist=str(doc.get("artist_names") or doc.get("primary_artist_name") or ""),
        artist_id=None,
        album=str(doc.get("album_name") or "") or None,
        year=int(doc.get("release_year") or 0) or None,
        duration=_format_duration(int(doc.get("duration_ms") or 0)),
        plays=None,
        composers=[],
        lyrics=str(doc.get("lyrics") or "") or None,
        lyrics_source=str(doc.get("lyrics_source") or "") or None,
        lyrics_source_url=str(doc.get("lyrics_source_url") or "") or None,
        genres=[g.strip() for g in str(doc.get("artist_genres") or "").split(",") if g.strip()],
        macro_genre=str(doc.get("macro_genre") or "") or None,
    )


def _album_response_from_payload(payload: dict[str, Any]) -> AlbumResponse:
    artist_summary = payload.get("artist_summary") or {}
    tracks = [
        AlbumTrack(**track) for track in (payload.get("tracks") or []) if isinstance(track, dict)
    ]
    top_tracks = [
        TrackRef(
            title=str(track.get("title") or ""),
            album=str(track.get("album") or "") or None,
            plays=str(track.get("plays") or "") or None,
        )
        for track in (artist_summary.get("top_tracks") or [])
        if isinstance(track, dict)
    ]
    albums = [
        AlbumRef(
            title=str(album.get("title") or ""),
            year=album.get("year"),
            tracks=album.get("tracks"),
        )
        for album in (artist_summary.get("albums") or [])
        if isinstance(album, dict)
    ]
    summary = AlbumArtistSummary(
        id=str(artist_summary.get("id") or payload.get("artist_id") or ""),
        name=str(artist_summary.get("name") or payload.get("artist") or ""),
        image_url=str(artist_summary.get("image_url") or "") or None,
        genres=list(artist_summary.get("genres") or []),
        popularity=artist_summary.get("popularity"),
        followers_total=artist_summary.get("followers_total"),
        top_tracks=top_tracks,
        albums=albums,
    )
    return AlbumResponse(
        id=str(payload.get("id") or ""),
        title=str(payload.get("title") or ""),
        artist=str(payload.get("artist") or ""),
        artist_id=str(payload.get("artist_id") or "") or None,
        year=payload.get("year"),
        description=str(payload.get("description") or "") or None,
        tags=list(payload.get("tags") or []),
        tracks_count=payload.get("tracks_count"),
        cover_url=str(payload.get("cover_url") or "") or None,
        artist_image_url=str(payload.get("artist_image_url") or "") or None,
        album_type=str(payload.get("album_type") or "") or None,
        label=str(payload.get("label") or "") or None,
        duration=str(payload.get("duration") or "") or None,
        total_duration_ms=payload.get("total_duration_ms"),
        tracks=tracks,
        artist_summary=summary,
    )


def _format_duration(ms: int) -> str | None:
    if not ms:
        return None
    seconds = ms // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"
