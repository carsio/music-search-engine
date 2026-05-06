"""CLI: `python -m music_search.lyrics <subcommand>`.

Subcomandos:
- fetch:  baixa letras a partir do parquet curado, escrevendo no cache SQLite.
- stats:  imprime contagens por status e por fonte.
- export: exporta os hits do cache como parquet pronto para indexacao.
- probe:  testa as fontes em uma faixa avulsa (sanity check).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import httpx

from music_search.lyrics.cache import LyricsCache
from music_search.lyrics.normalize import normalize_artist, normalize_title
from music_search.lyrics.pipeline import PipelineConfig, run_pipeline
from music_search.lyrics.sources import (
    GeniusSource,
    LetrasMusBrSource,
    LrcLibSource,
    LyricFindSource,
    LyricsOvhSource,
    LyricsSource,
    VagalumeSource,
)

DEFAULT_PARQUET = Path("data/derived/br_curated_tracks.parquet")
DEFAULT_CACHE = Path("data/derived/lyrics_cache.sqlite")
DEFAULT_EXPORT = Path("data/derived/lyrics.parquet")


def _build_sources(client: httpx.AsyncClient) -> list[LyricsSource]:
    """Cascata de fontes. Ordem otimizada para catalogo brasileiro + recuperacao
    de misses internacionais:

    1. letras.mus.br (scraping HTML, melhor cobertura BR, sem chave).
    2. lrclib (API JSON publica, sem chave, ~3M tracks — cobre internacional/pop).
    3. vagalume (API JSON, opt-in via VAGALUME_API_KEY — instavel atualmente).
    4. lyrics.ovh (publico, fallback simples).
    5. lyricfind (scraping HTML do player publico, cobre catalogo licenciado).
    6. genius (API com GENIUS_TOKEN; sem token usa endpoint publico de busca).
    """
    sources: list[LyricsSource] = [
        LetrasMusBrSource(client),
        LrcLibSource(client),
    ]

    vagalume_key = os.environ.get("VAGALUME_API_KEY", "").strip()
    if vagalume_key:
        sources.append(VagalumeSource(client, api_key=vagalume_key))

    sources.append(LyricsOvhSource(client))
    sources.append(LyricFindSource(client))

    # Genius funciona com ou sem token. Com token = mais quota e estabilidade;
    # sem token = endpoint publico do site (cota menor, sujeito a Cloudflare).
    genius_token = os.environ.get("GENIUS_TOKEN", "").strip() or None
    sources.append(GeniusSource(client, token=genius_token))

    return sources


def cmd_fetch(args: argparse.Namespace) -> None:
    cfg = PipelineConfig(
        parquet_path=Path(args.parquet),
        cache_path=Path(args.cache),
        concurrency=args.concurrency,
        request_timeout=args.timeout,
        max_retries=args.max_retries,
        retry_errors=args.retry_errors,
        retry_misses=args.retry_misses,
        retry_blocked=args.retry_blocked,
        limit=args.limit,
    )
    asyncio.run(run_pipeline(cfg, _build_sources))


def cmd_stats(args: argparse.Namespace) -> None:
    cache = LyricsCache(Path(args.cache))
    stats = cache.stats()
    by_source = cache.stats_by_source()
    total = sum(stats.values())
    print(f"Cache: {args.cache}")
    print(f"Total resolvidos: {total:,}")
    for status in ("hit", "miss", "error", "blocked"):
        n = stats.get(status, 0)
        pct = (100 * n / total) if total else 0.0
        print(f"  {status:8s}: {n:>8,}  ({pct:5.1f}%)")
    if by_source:
        print("Hits por fonte:")
        for src, n in by_source.items():
            print(f"  {src:12s}: {n:>8,}")
    cache.close()


def cmd_export(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache = LyricsCache(Path(args.cache))
    rows = list(cache.iter_hits())
    cache.close()
    if not rows:
        print("Cache vazio (nenhum hit). Nada para exportar.")
        return
    table = pa.Table.from_pylist(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd")
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Exportado: {output} ({size_mb:.2f} MB, {len(rows):,} letras)")


def cmd_where(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache).resolve()
    print(f"Cache SQLite: {cache_path}")
    if cache_path.exists():
        size_kb = cache_path.stat().st_size / 1024
        print(f"Tamanho:      {size_kb:,.1f} KB")
        cache = LyricsCache(cache_path)
        print(f"Registros:    {cache.total():,}")
        cache.close()
        print()
        print("Como inspecionar manualmente:")
        print(f"  sqlite3 {cache_path}")
        print("  > .schema lyrics")
        print("  > SELECT artist, title, status, source FROM lyrics LIMIT 10;")
        print("  > SELECT artist, title, lyrics FROM lyrics WHERE status='hit' LIMIT 1;")
    else:
        print("(cache ainda nao existe — rode `fetch` primeiro)")


def cmd_show(args: argparse.Namespace) -> None:
    cache = LyricsCache(Path(args.cache))
    row = cache.get(args.track_id)
    cache.close()
    if not row:
        print(f"track_id '{args.track_id}' nao encontrado no cache.")
        return
    print(f"track_id:   {row['track_id']}")
    print(f"artist:     {row['artist']}")
    print(f"title:      {row['title']}")
    print(f"status:     {row['status']}")
    print(f"source:     {row['source']}")
    print(f"source_url: {row['source_url']}")
    print(f"attempts:   {row['attempts']}")
    if row.get("error"):
        print(f"error:      {row['error']}")
    if row.get("lyrics"):
        print()
        print("--- letra ---")
        print(row["lyrics"])


def cmd_sample(args: argparse.Namespace) -> None:
    cache = LyricsCache(Path(args.cache))
    rows = cache.list_by_status(
        status=args.status, limit=args.n, order="random" if args.random else "recent"
    )
    cache.close()
    if not rows:
        print("Nenhum registro encontrado com esses filtros.")
        return
    for i, row in enumerate(rows, 1):
        preview = (row.get("lyrics") or "")[:120].replace("\n", " / ")
        marker = "*" if row.get("lyrics") else " "
        print(
            f"[{i:>3}] {marker} {row['status']:<7} {row['source'] or '—':<14} "
            f"{row['artist']} — {row['title']}"
        )
        if preview:
            print(f"      {preview}")


def cmd_probe(args: argparse.Namespace) -> None:
    artist = normalize_artist(args.artist)
    title = normalize_title(args.title)
    print(f"Normalizado: artist='{artist}' | title='{title}'")

    async def _run() -> None:
        async with httpx.AsyncClient(
            headers={"User-Agent": "music-search-engine-lyrics/0.1 probe"},
            follow_redirects=True,
            timeout=20.0,
        ) as client:
            sources = _build_sources(client)
            for source in sources:
                print(f"\n>>> {source.name}")
                result = await source.fetch(artist, title)
                preview = (result.lyrics or "")[:300].replace("\n", " / ")
                print(f"  status={result.status.value}")
                print(f"  source_url={result.source_url}")
                print(f"  error={result.error}")
                if preview:
                    print(f"  lyrics[:300]={preview}")

    asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m music_search.lyrics",
        description="Pipeline de extracao de letras com cache persistente.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Baixa letras das faixas no parquet")
    p_fetch.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    p_fetch.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_fetch.add_argument("--concurrency", type=int, default=8)
    p_fetch.add_argument("--timeout", type=float, default=15.0)
    p_fetch.add_argument("--max-retries", type=int, default=2)
    p_fetch.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocessa faixas que terminaram com status=error",
    )
    p_fetch.add_argument(
        "--retry-misses",
        action="store_true",
        help="Reprocessa faixas que terminaram com status=miss "
        "(util ao adicionar novas fontes ao cascade)",
    )
    p_fetch.add_argument(
        "--retry-blocked",
        action="store_true",
        help="Reprocessa faixas que terminaram com status=blocked",
    )
    p_fetch.add_argument("--limit", type=int, default=None)
    p_fetch.set_defaults(func=cmd_fetch)

    p_stats = sub.add_parser("stats", help="Imprime contagens do cache")
    p_stats.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_stats.set_defaults(func=cmd_stats)

    p_export = sub.add_parser("export", help="Exporta hits para parquet")
    p_export.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_export.add_argument("--output", default=str(DEFAULT_EXPORT))
    p_export.set_defaults(func=cmd_export)

    p_probe = sub.add_parser("probe", help="Testa fontes em uma faixa avulsa")
    p_probe.add_argument("artist")
    p_probe.add_argument("title")
    p_probe.set_defaults(func=cmd_probe)

    p_where = sub.add_parser("where", help="Mostra onde o cache esta no disco")
    p_where.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_where.set_defaults(func=cmd_where)

    p_show = sub.add_parser("show", help="Mostra uma entrada especifica do cache")
    p_show.add_argument("track_id", help="track_id do Spotify (coluna 'track_id' no parquet)")
    p_show.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_show.set_defaults(func=cmd_show)

    p_sample = sub.add_parser("sample", help="Lista N entradas do cache (preview da letra)")
    p_sample.add_argument("-n", type=int, default=10)
    p_sample.add_argument(
        "--status",
        choices=["hit", "miss", "error", "blocked"],
        default=None,
        help="Filtrar por status (default: todos)",
    )
    p_sample.add_argument("--random", action="store_true", help="Aleatorio em vez de recente")
    p_sample.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_sample.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
