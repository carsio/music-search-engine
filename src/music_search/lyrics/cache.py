"""Cache SQLite para letras extraidas. Idempotente e seguro para retomadas."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lyrics (
    track_id TEXT PRIMARY KEY,
    isrc TEXT,
    artist TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT,
    source_url TEXT,
    lyrics TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 1,
    fetched_at INTEGER NOT NULL,
    trace TEXT
);
CREATE INDEX IF NOT EXISTS idx_lyrics_status ON lyrics(status);
CREATE INDEX IF NOT EXISTS idx_lyrics_source ON lyrics(source);
"""

# Status considerados resolvidos e que nao precisam ser reprocessados.
_TERMINAL = ("hit", "miss", "blocked")


class LyricsCache:
    """Camada fina sobre SQLite. WAL ligado para concorrencia leitura/escrita."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # Migration in-place: cache antigo nao tem coluna `trace`. ALTER TABLE
        # com IF NOT EXISTS nao existe em SQLite, entao a gente checa e adiciona.
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(lyrics)").fetchall()}
        if "trace" not in cols:
            self.conn.execute("ALTER TABLE lyrics ADD COLUMN trace TEXT")
        self.conn.commit()

    def has_resolved(
        self,
        track_id: str,
        *,
        retry_errors: bool = False,
        retry_misses: bool = False,
        retry_blocked: bool = False,
    ) -> bool:
        """Retorna True se a faixa ja foi processada com status terminal.

        Os flags `retry_*` permitem forcar reprocessamento por status:
        - retry_errors:  re-tenta linhas com status='error'.
        - retry_misses:  re-tenta linhas com status='miss' (util quando se
          adicionam novas fontes ao cascade — os 'miss' antigos podem virar 'hit').
        - retry_blocked: re-tenta linhas com status='blocked'.
        """
        row = self.conn.execute(
            "SELECT status FROM lyrics WHERE track_id = ?", (track_id,)
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
        track_id: str,
        isrc: str | None,
        artist: str,
        title: str,
        status: str,
        source: str | None = None,
        source_url: str | None = None,
        lyrics: str | None = None,
        error: str | None = None,
        attempts: int = 1,
        trace: str | None = None,
    ) -> None:
        now = int(time.time())
        self.conn.execute(
            """
            INSERT INTO lyrics (
                track_id, isrc, artist, title, status, source, source_url,
                lyrics, error, attempts, fetched_at, trace
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET
                isrc       = excluded.isrc,
                artist     = excluded.artist,
                title      = excluded.title,
                status     = excluded.status,
                source     = excluded.source,
                source_url = excluded.source_url,
                lyrics     = excluded.lyrics,
                error      = excluded.error,
                attempts   = lyrics.attempts + excluded.attempts,
                fetched_at = excluded.fetched_at,
                trace      = excluded.trace
            """,
            (
                track_id,
                isrc,
                artist,
                title,
                status,
                source,
                source_url,
                lyrics,
                error,
                attempts,
                now,
                trace,
            ),
        )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, count(*) AS n FROM lyrics GROUP BY status"
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def stats_by_source(self) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT source, count(*) AS n
            FROM lyrics
            WHERE status = 'hit'
            GROUP BY source
            ORDER BY n DESC
            """
        ).fetchall()
        return {row["source"] or "(none)": row["n"] for row in rows}

    def iter_hits(self):
        cur = self.conn.execute(
            """
            SELECT track_id, isrc, artist, title, source, source_url, lyrics
            FROM lyrics
            WHERE status = 'hit'
            """
        )
        for row in cur:
            yield dict(row)

    def get(self, track_id: str) -> dict | None:
        row = self.conn.execute(
            """
            SELECT track_id, isrc, artist, title, status, source, source_url,
                   lyrics, error, attempts, fetched_at, trace
            FROM lyrics WHERE track_id = ?
            """,
            (track_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_by_status(
        self, status: str | None = None, limit: int = 10, order: str = "recent"
    ) -> list[dict]:
        """Lista entradas filtradas por status. `order`: 'recent' | 'random'."""
        sql = (
            "SELECT track_id, artist, title, status, source, source_url, "
            "lyrics, error, fetched_at FROM lyrics"
        )
        params: list = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY RANDOM()" if order == "random" else " ORDER BY fetched_at DESC"
        sql += " LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def total(self) -> int:
        return self.conn.execute("SELECT count(*) FROM lyrics").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
