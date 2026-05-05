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


def test_genius_extract_hits_handles_api_and_public_payloads() -> None:
    from music_search.lyrics.sources.genius import GeniusSource

    api_payload = {
        "response": {
            "hits": [
                {"type": "song", "result": {"url": "u1", "title": "t1"}},
                {"type": "lyric", "result": {"url": "u2"}},
            ]
        }
    }
    public_payload = {
        "response": {
            "sections": [
                {"type": "top_hit", "hits": [{"type": "song", "result": {"url": "u3"}}]},
                {
                    "type": "song",
                    "hits": [
                        {"type": "song", "result": {"url": "u4"}},
                        {"type": "album", "result": {"url": "ignored"}},
                    ],
                },
            ]
        }
    }

    api_hits = GeniusSource._extract_hits(api_payload)
    pub_hits = GeniusSource._extract_hits(public_payload)
    assert [h["result"]["url"] for h in api_hits] == ["u1"]
    assert [h["result"]["url"] for h in pub_hits] == ["u4"]


def test_lrclib_strip_timestamps_keeps_only_text() -> None:
    from music_search.lyrics.sources.lrclib import LrcLibSource

    synced = (
        "[00:12.34]Linha um\n[00:15.00]Linha dois\n[01:02.5]Linha tres\n  \nLinha sem timestamp\n"
    )
    assert LrcLibSource._strip_timestamps(synced) == (
        "Linha um\nLinha dois\nLinha tres\nLinha sem timestamp"
    )


def test_lrclib_extract_lyrics_prefers_plain_falls_back_to_synced() -> None:
    from music_search.lyrics.sources.lrclib import LrcLibSource

    plain_only = {"plainLyrics": "verso 1\nverso 2", "syncedLyrics": ""}
    synced_only = {"plainLyrics": "", "syncedLyrics": "[00:01.00]oi"}
    instrumental = {"instrumental": True, "plainLyrics": "ignorar"}
    empty = {"plainLyrics": "", "syncedLyrics": ""}

    assert LrcLibSource._extract_lyrics(plain_only) == "verso 1\nverso 2"
    assert LrcLibSource._extract_lyrics(synced_only) == "oi"
    assert LrcLibSource._extract_lyrics(instrumental) is None
    assert LrcLibSource._extract_lyrics(empty) is None


def test_lyrics_cache_retry_misses(tmp_path: Path) -> None:
    cache = LyricsCache(tmp_path / "lyrics.sqlite")
    cache.upsert(
        track_id="trk-miss",
        isrc=None,
        artist="A",
        title="B",
        status="miss",
    )
    cache.upsert(
        track_id="trk-blocked",
        isrc=None,
        artist="A",
        title="C",
        status="blocked",
    )
    cache.upsert(
        track_id="trk-hit",
        isrc=None,
        artist="A",
        title="D",
        status="hit",
        lyrics="...",
    )

    # default: tudo resolvido
    assert cache.has_resolved("trk-miss")
    assert cache.has_resolved("trk-blocked")
    assert cache.has_resolved("trk-hit")

    # retry_misses libera apenas miss
    assert not cache.has_resolved("trk-miss", retry_misses=True)
    assert cache.has_resolved("trk-blocked", retry_misses=True)
    assert cache.has_resolved("trk-hit", retry_misses=True)

    # retry_blocked libera apenas blocked
    assert cache.has_resolved("trk-miss", retry_blocked=True)
    assert not cache.has_resolved("trk-blocked", retry_blocked=True)

    # hit nunca e retentado
    assert cache.has_resolved("trk-hit", retry_errors=True, retry_misses=True, retry_blocked=True)
    cache.close()
