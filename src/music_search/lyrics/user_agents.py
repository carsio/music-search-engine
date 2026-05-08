"""Re-export fino de `music_search._async_http.user_agents`.

A logica original mora em `_async_http/user_agents.py` para ser reusada por
enrichment (Wikipedia). Este modulo permanece como ponto de import dos sources de
letras para nao quebrar o pipeline ja rodando.
"""

from music_search._async_http.user_agents import random_browser_headers, random_browser_ua

__all__ = ["random_browser_headers", "random_browser_ua"]
