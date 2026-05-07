"""CLI minima para sanity check da LLM: `python -m music_search.llm intent "carolina"`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from music_search.llm.cache import LLMCache
from music_search.llm.client import NimClient
from music_search.llm.tasks import (
    classify_intent,
    extract_artist_json,
    rerank,
)


async def _run(args: argparse.Namespace) -> None:
    cache = None if args.no_cache else LLMCache()
    async with NimClient() as client:
        if args.cmd == "intent":
            result = await classify_intent(args.query, client=client, cache=cache)
            print(result)
        elif args.cmd == "extract-artist":
            html = sys.stdin.read()
            result = await extract_artist_json(
                html, source_url=args.source_url, client=client, cache=cache
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.cmd == "rerank":
            candidates = json.loads(sys.stdin.read())
            result = await rerank(args.query, candidates, args.top, client=client, cache=cache)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.cmd == "stats":
            if cache is None:
                print("(--no-cache passed; nada a reportar)")
                return
            print(json.dumps(cache.stats(), indent=2))
    if cache is not None:
        cache.close()


def main() -> None:
    parser = argparse.ArgumentParser("music_search.llm")
    parser.add_argument("--no-cache", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_intent = sub.add_parser("intent", help="classifica intent de uma query")
    p_intent.add_argument("query")

    p_extract = sub.add_parser("extract-artist", help="extrai JSON de artista do HTML em stdin")
    p_extract.add_argument("--source-url", default=None)

    p_rerank = sub.add_parser("rerank", help="reranking; candidatos em stdin como JSON")
    p_rerank.add_argument("query")
    p_rerank.add_argument("--top", type=int, default=10)

    sub.add_parser("stats", help="status do cache da LLM")

    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
