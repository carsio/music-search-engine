"""Le `enrichment_cache.sqlite` e exporta um parquet por entidade.

Uso:
    uv run python scripts/export_entities.py [--kinds artist,album,genre,composer]
    uv run python scripts/export_entities.py --kinds artist --output-dir data/derived/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from music_search.datasets import DEFAULT_FINAL_DATASET_DIR
from music_search.enrichment.pipeline import (
    DEFAULT_CACHE_PATH,
    DEFAULT_CACHE_TABLE,
    slugify,
)

KINDS = ("artist", "album", "genre", "composer")
DEFAULT_OUTPUT_DIR = DEFAULT_FINAL_DATASET_DIR


def _output_path_for(kind: str, base: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return base / f"br_{kind}s.parquet"


def export_kind(
    kind: str,
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    cache_table: str = DEFAULT_CACHE_TABLE,
    output: Path | None = None,
) -> Path:
    if not cache_path.exists():
        raise FileNotFoundError(f"cache vazio: {cache_path}")
    output = output or _output_path_for(kind)
    output.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(
            f"INSTALL sqlite_scanner; LOAD sqlite_scanner; "
            f"ATTACH '{cache_path.as_posix()}' AS db (TYPE sqlite, READ_ONLY)"
        )
        rows = con.execute(
            f"""
            SELECT key, kind, status, source, source_url, payload_json
            FROM db.{cache_table}
            WHERE kind = ? AND status = 'hit' AND payload_json IS NOT NULL
            """,
            [kind],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        print(f"[{kind}] sem hits no cache — pulando.")
        return output

    records: list[dict] = []
    for row in rows:
        key, _kind, _status, source, source_url, payload_json = row
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except json.JSONDecodeError:
            continue
        record = dict(payload)
        record.setdefault("id", key.split(":", 1)[1] if ":" in key else slugify(key))
        record.setdefault("source", source)
        record.setdefault("source_url", source_url)
        records.append(_normalize_record(record, kind))

    con = duckdb.connect()
    try:
        con.register("records", pd.DataFrame.from_records(records))
        con.execute(
            f"COPY (SELECT * FROM records) TO '{output.as_posix()}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        con.close()

    size_kb = output.stat().st_size / 1024
    print(f"[{kind}] {len(records):,} registros -> {output} ({size_kb:.1f} KB)")
    return output


def _normalize_record(record: dict, kind: str) -> dict:
    """Garante que campos esperados existem, mesmo que vazios.

    Permite que o parquet tenha schema estavel quando a LLM omite campos.
    DuckDB infere o schema do primeiro registro entao mantemos a forma.
    """
    base = {
        "id": str(record.get("id") or ""),
        "source": str(record.get("source") or ""),
        "source_url": str(record.get("source_url") or ""),
        "raw_text": str(record.get("raw_text") or "") or None,
    }
    if kind == "artist":
        return {
            **base,
            "name": str(record.get("name") or ""),
            "tagline": str(record.get("tagline") or "") or None,
            "bio": str(record.get("bio") or "") or None,
            "genres": _list_str(record.get("genres")),
            "origin": str(record.get("origin") or "") or None,
            "year_started": _opt_int(record.get("year_started")),
            "monthly_listeners": str(record.get("monthly_listeners") or "") or None,
            "popularity": _opt_int(record.get("popularity")),
            "albums_json": json.dumps(record.get("albums") or [], ensure_ascii=False),
            "top_tracks_json": json.dumps(record.get("top_tracks") or [], ensure_ascii=False),
        }
    if kind == "album":
        return {
            **base,
            "title": str(record.get("title") or ""),
            "artist": str(record.get("artist") or ""),
            "year": _opt_int(record.get("year")),
            "description": str(record.get("description") or "") or None,
            "tracks_json": json.dumps(record.get("tracks") or [], ensure_ascii=False),
        }
    if kind == "genre":
        return {
            **base,
            "name": str(record.get("name") or ""),
            "description": str(record.get("description") or "") or None,
            "origin": str(record.get("origin") or "") or None,
            "decade": str(record.get("decade") or "") or None,
            "representative_artists": _list_str(record.get("representative_artists")),
            "related_genres": _list_str(record.get("related_genres")),
        }
    if kind == "composer":
        return {
            **base,
            "name": str(record.get("name") or ""),
            "bio": str(record.get("bio") or "") or None,
            "genres": _list_str(record.get("genres")),
            "origin": str(record.get("origin") or "") or None,
            "year_started": _opt_int(record.get("year_started")),
            "notable_works_json": json.dumps(record.get("notable_works") or [], ensure_ascii=False),
        }
    return base


def _list_str(v: object) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x) for x in v if x is not None]


def _opt_int(v: object) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta entidades do enrichment_cache para parquet."
    )
    parser.add_argument("--kinds", default=",".join(KINDS), help="lista separada por virgula")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in KINDS:
            raise SystemExit(f"kind invalido: {k}. Use um de {KINDS}")
    for k in kinds:
        export_kind(
            k,
            cache_path=args.cache_path,
            output=_output_path_for(k, args.output_dir),
        )


if __name__ == "__main__":
    main()
