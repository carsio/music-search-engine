from __future__ import annotations

from pathlib import Path

import duckdb

from music_search.data.albums import build_album_search_records, load_album_catalog_from_tracks


def _write_tracks_parquet(path: Path) -> None:
    con = duckdb.connect()
    try:
        target = path.as_posix()
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM (
                    VALUES
                        (
                            'trk-2', 'Faixa Dois', '', 2, 1,
                            'artist-1', 'Artista A', 5000, 84,
                            'mpb | samba', 'mpb',
                            'album-1', 'Raizes do Norte', 'album', 'Selo Azul', 2,
                            2024, 78, 180000, FALSE,
                            'https://example.test/album.jpg', 'https://example.test/artist.jpg'
                        ),
                        (
                            'trk-1', 'Faixa Um', 'https://example.test/preview.mp3', 1, 1,
                            'artist-1', 'Artista A', 5000, 84,
                            'mpb | samba', 'mpb',
                            'album-1', 'Raizes do Norte', 'album', 'Selo Azul', 2,
                            2024, 90, 200000, FALSE,
                            'https://example.test/album.jpg', 'https://example.test/artist.jpg'
                        ),
                        (
                            'trk-3', 'Outra Faixa', '', 1, 1,
                            '', 'Artista B', 1000, 65,
                            'forro', 'forro',
                            '', 'Sem ID', 'single', '', 1,
                            2023, 50, 150000, TRUE,
                            '', ''
                        )
                ) AS t(
                    track_id, track_name, preview_url, track_number, disc_number,
                    primary_artist_id, primary_artist_name, primary_artist_followers_total,
                    primary_artist_popularity, artist_genres, macro_genre,
                    album_id, album_name, album_type, album_label, album_total_tracks,
                    release_year, track_popularity, duration_ms, explicit,
                    album_image_url, primary_artist_image_url
                )
            )
            TO '{target}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def test_load_album_catalog_from_tracks_agrega_e_ordena(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.parquet"
    _write_tracks_parquet(tracks)

    catalog = load_album_catalog_from_tracks(tracks)

    assert set(catalog) == {"album-1", "album-sem-id-artista-b-2023"}
    album = catalog["album-1"]
    assert album["title"] == "Raizes do Norte"
    assert album["artist"] == "Artista A"
    assert album["year"] == 2024
    assert album["tracks_count"] == 2
    assert album["cover_url"] == "https://example.test/album.jpg"
    assert album["tags"] == ["mpb", "samba"]
    assert [track["id"] for track in album["tracks"]] == ["trk-1", "trk-2"]
    assert album["tracks"][0]["duration"] == "3:20"
    assert album["description"].startswith("Raizes do Norte: Album de Artista A")
    assert album["artist_summary"]["id"] == "artist-1"
    assert [track["id"] for track in album["artist_summary"]["top_tracks"]] == ["trk-1", "trk-2"]


def test_build_album_search_records_reduz_payload(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.parquet"
    _write_tracks_parquet(tracks)

    catalog = load_album_catalog_from_tracks(tracks)
    records = build_album_search_records(catalog.values())

    record = next(item for item in records if item["id"] == "album-1")
    assert record["title"] == "Raizes do Norte"
    assert record["tracks_count"] == 2
    assert record["cover_url"] == "https://example.test/album.jpg"
    assert "tracks" not in record
