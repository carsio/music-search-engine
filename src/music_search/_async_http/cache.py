"""KeyValueCache: cache SQLite generico para fetchers cascateados.

Mesma forma do `LyricsCache`, mas com schema agnostico (chave + payload JSON).
Usado por `enrichment/` (Wikipedia) e `llm/` (respostas da LLM).

Nao substitui `LyricsCache` — coexistem. O schema do cache de letras e otimizado
para inspecao via SQL com colunas dedicadas (`artist`, `title`, `lyrics`); aqui
o foco e generalidade, com tudo em `payload_json`.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TERMINAL = ("hit", "miss", "blocked")


def _validate_table_name(name: str) -> str:
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(f"invalid table name: {name!r}")
    return name


class KeyValueCache:
    """Cache SQLite chave-valor com WAL, idempotente e seguro para retomadas.

    Schema (tabela parametrizada):

        key TEXT PRIMARY KEY,
        kind TEXT,                  -- categoria opcional (ex: 'artist', 'album')
        status TEXT NOT NULL,       -- hit | miss | blocked | error
        source TEXT,
        source_url TEXT,
        payload_json TEXT,          -- JSON arbitrario (resultado serializado)
        error TEXT,
        attempts INTEGER NOT NULL DEFAULT 1,
        fetched_at INTEGER NOT NULL,
        trace TEXT                  -- JSON com tentativas (debug)

    A chave deve ser deterministica para a entrada (ex.: sha1 de modelo+prompt+input
    para LLM, ou `<kind>:<slug>` para enrichment).
    """

    def __init__(self, path: Path | str, table: str = "kv_cache"):
        self.path = Path(path)
        self.table = _validate_table_name(table)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(self._schema())
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.commit()

    def _schema(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            key TEXT PRIMARY KEY,
            kind TEXT,
            status TEXT NOT NULL,
            source TEXT,
            source_url TEXT,
            payload_json TEXT,
            error TEXT,
            attempts INTEGER NOT NULL DEFAULT 1,
            fetched_at INTEGER NOT NULL,
            trace TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_{self.table}_status ON {self.table}(status);
        CREATE INDEX IF NOT EXISTS idx_{self.table}_kind ON {self.table}(kind);
        CREATE INDEX IF NOT EXISTS idx_{self.table}_source ON {self.table}(source);
        """

    def has_resolved(
        self,
        key: str,
        *,
        retry_errors: bool = False,
        retry_misses: bool = False,
        retry_blocked: bool = False,
    ) -> bool:
        row = self.conn.execute(
            f"SELECT status FROM {self.table} WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return False
        status = row["status"]
        if retry_errors and status == "error":
            return False
        if retry_misses and status == "miss":
            return False
        if retry_blocked and status == "blocked":
            return False
        return status in _TERMINAL or status == "error"

    def upsert(
        self,
        *,
        key: str,
        status: str,
        kind: str | None = None,
        source: str | None = None,
        source_url: str | None = None,
        payload: Any = None,
        error: str | None = None,
        attempts: int = 1,
        trace: str | None = None,
    ) -> None:
        now = int(time.time())
        payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
        self.conn.execute(
            f"""
            INSERT INTO {self.table} (
                key, kind, status, source, source_url,
                payload_json, error, attempts, fetched_at, trace
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                kind         = excluded.kind,
                status       = excluded.status,
                source       = excluded.source,
                source_url   = excluded.source_url,
                payload_json = excluded.payload_json,
                error        = excluded.error,
                attempts     = {self.table}.attempts + excluded.attempts,
                fetched_at   = excluded.fetched_at,
                trace        = excluded.trace
            """,
            (key, kind, status, source, source_url, payload_json, error, attempts, now, trace),
        )
        self.conn.commit()

    def get(self, key: str) -> dict | None:
        row = self.conn.execute(
            f"""
            SELECT key, kind, status, source, source_url, payload_json,
                   error, attempts, fetched_at, trace
            FROM {self.table} WHERE key = ?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        if out["payload_json"]:
            try:
                out["payload"] = json.loads(out["payload_json"])
            except json.JSONDecodeError:
                out["payload"] = None
        else:
            out["payload"] = None
        return out

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            f"SELECT status, count(*) AS n FROM {self.table} GROUP BY status"
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def stats_by_kind(self) -> dict[str, dict[str, int]]:
        rows = self.conn.execute(
            f"""
            SELECT kind, status, count(*) AS n
            FROM {self.table}
            GROUP BY kind, status
            """
        ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(row["kind"] or "(none)", {})[row["status"]] = row["n"]
        return out

    def iter_hits(self, kind: str | None = None):
        if kind is None:
            cur = self.conn.execute(
                f"""
                SELECT key, kind, source, source_url, payload_json
                FROM {self.table} WHERE status = 'hit'
                """
            )
        else:
            cur = self.conn.execute(
                f"""
                SELECT key, kind, source, source_url, payload_json
                FROM {self.table} WHERE status = 'hit' AND kind = ?
                """,
                (kind,),
            )
        for row in cur:
            out = dict(row)
            if out["payload_json"]:
                try:
                    out["payload"] = json.loads(out["payload_json"])
                except json.JSONDecodeError:
                    out["payload"] = None
            yield out

    def total(self) -> int:
        return self.conn.execute(
            f"SELECT count(*) FROM {self.table}"
        ).fetchone()[0]

    def close(self) -> None:
        self.conn.close()
