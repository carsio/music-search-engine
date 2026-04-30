from pathlib import Path

import pytest

from music_search.lyrics.cache import LyricsCache
from music_search.lyrics.normalize import normalize_artist, normalize_title, slugify


def test_normalize_title_remove_versions_and_features() -> None:
    assert normalize_title("Evidências - Ao Vivo (Remix) feat. Fulano") == "Evidências"
    assert normalize_title("Garota de Ipanema [Slowed + Reverb]") == "Garota de Ipanema"


def test_normalize_artist_keeps_primary_artist() -> None:
    assert normalize_artist("Gilberto Gil feat. Caetano Veloso") == "Gilberto Gil"
    assert normalize_artist("Anavitória & Vitor Kley") == "Anavitória"


def test_slugify_normalizes_accents() -> None:
    assert slugify("Não Quero Dinheiro") == "nao-quero-dinheiro"


def test_browser_headers_do_not_force_brotli() -> None:
    from music_search.lyrics.user_agents import random_browser_headers

    headers = random_browser_headers()

    assert "Accept-Encoding" not in headers


def test_lyrics_cache_upsert_and_retry_errors(tmp_path: Path) -> None:
    cache = LyricsCache(tmp_path / "lyrics.sqlite")
    cache.upsert(
        track_id="trk1",
        isrc="BR123",
        artist="Artist",
        title="Song",
        status="error",
        error="timeout",
    )

    assert cache.has_resolved("trk1")
    assert not cache.has_resolved("trk1", retry_errors=True)

    cache.upsert(
        track_id="trk1",
        isrc="BR123",
        artist="Artist",
        title="Song",
        status="hit",
        source="fixture",
        lyrics="linha 1\nlinha 2",
        attempts=2,
    )

    row = cache.get("trk1")
    cache.close()

    assert row is not None
    assert row["status"] == "hit"
    assert row["attempts"] == 3
    assert row["lyrics"] == "linha 1\nlinha 2"


def test_letras_mus_br_parse_lyrics_from_html() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")
    from music_search.lyrics.sources.letras_mus_br import LetrasMusBrSource

    html = """
    <html><body>
      <div class="lyric-original">
        <p>Primeira linha<br>Segunda linha</p>
        <p>Outro verso</p>
      </div>
    </body></html>
    """

    assert LetrasMusBrSource._parse_lyrics(html) == "Primeira linha\nSegunda linha\n\nOutro verso"


def test_genius_parse_lyrics_from_data_containers() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")
    from music_search.lyrics.sources.genius import GeniusSource

    html = """
    <html><body>
      <div data-lyrics-container="true">Verso A<br/>Verso B</div>
      <div data-lyrics-container="true">Refrão</div>
    </body></html>
    """

    assert GeniusSource._parse_lyrics(html) == "Verso A\nVerso B\nRefrão"
