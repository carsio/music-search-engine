"""Interface Tkinter para baixar letras em batches manuais.

Uso:
    uv run python -m music_search.lyrics.ui_tk

A UI orquestra o pipeline async em uma thread separada (Tk e asyncio nao convivem
no mesmo loop), e atualiza periodicamente o painel de status lendo direto do
cache SQLite. O download e idempotente: rodar varias vezes nao reprocessa faixas
ja resolvidas — apenas avanca para as proximas pendentes.
"""

from __future__ import annotations

import asyncio
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

import httpx

from music_search.lyrics.cache import LyricsCache
from music_search.lyrics.cli import _build_sources
from music_search.lyrics.normalize import normalize_artist, normalize_title
from music_search.lyrics.pipeline import read_tracks
from music_search.lyrics.sources.base import LyricsSource, Status

DEFAULT_PARQUET = Path("data/derived/br_curated_tracks.parquet")
DEFAULT_CACHE = Path("data/derived/lyrics_cache.sqlite")

_WINDOW_TITLE = "Letras — Downloader manual"
_WINDOW_SIZE = "980x680"
_REFRESH_INTERVAL_MS = 700


class LyricsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(_WINDOW_TITLE)
        self.geometry(_WINDOW_SIZE)
        self.minsize(820, 560)

        self._parquet_path = DEFAULT_PARQUET
        self._cache_path = DEFAULT_CACHE

        # comunicacao com a worker thread
        self._event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

        self._total_tracks: int | None = None

        self._build_ui()
        self.after(100, self._poll_events)
        self.after(200, self._refresh_status)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        self._build_header()
        self._build_controls()
        self._build_progress()
        self._build_table()
        self._build_log()

    def _build_header(self) -> None:
        frame = ttk.Frame(self, padding=(12, 10, 12, 4))
        frame.pack(fill=tk.X)

        cache_label = ttk.Label(frame, text="Cache:")
        cache_label.grid(row=0, column=0, sticky="w")

        self._cache_var = tk.StringVar(value=str(self._cache_path.resolve()))
        cache_entry = ttk.Entry(frame, textvariable=self._cache_var, width=80)
        cache_entry.grid(row=0, column=1, sticky="we", padx=(6, 6))
        cache_entry.configure(state="readonly")

        ttk.Button(frame, text="Abrir pasta", command=self._open_cache_folder).grid(
            row=0, column=2, padx=(0, 4)
        )
        ttk.Button(frame, text="Copiar caminho", command=self._copy_cache_path).grid(
            row=0, column=3
        )

        # painel de status
        status_frame = ttk.Frame(self, padding=(12, 4, 12, 8))
        status_frame.pack(fill=tk.X)
        self._status_var = tk.StringVar(value="(carregando…)")
        ttk.Label(
            status_frame, textvariable=self._status_var, font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w")

        frame.columnconfigure(1, weight=1)

    def _build_controls(self) -> None:
        frame = ttk.LabelFrame(self, text="Baixar próximo batch", padding=(12, 8))
        frame.pack(fill=tk.X, padx=12, pady=(0, 8))

        ttk.Label(frame, text="Quantidade:").grid(row=0, column=0, sticky="w")
        self._limit_var = tk.IntVar(value=100)
        ttk.Spinbox(
            frame, from_=1, to=22000, increment=50, width=8, textvariable=self._limit_var
        ).grid(row=0, column=1, padx=(6, 16))

        ttk.Label(frame, text="Concurrency:").grid(row=0, column=2, sticky="w")
        self._concurrency_var = tk.IntVar(value=8)
        ttk.Spinbox(
            frame, from_=1, to=32, increment=1, width=5, textvariable=self._concurrency_var
        ).grid(row=0, column=3, padx=(6, 16))

        self._retry_errors_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="Re-tentar erros anteriores", variable=self._retry_errors_var
        ).grid(row=0, column=4, padx=(0, 16))

        self._start_btn = ttk.Button(frame, text="▶ Iniciar", command=self._start_batch)
        self._start_btn.grid(row=0, column=5, padx=(0, 4))

        self._stop_btn = ttk.Button(
            frame, text="■ Parar", command=self._stop_batch, state="disabled"
        )
        self._stop_btn.grid(row=0, column=6)

        frame.columnconfigure(7, weight=1)

    def _build_progress(self) -> None:
        frame = ttk.Frame(self, padding=(12, 0, 12, 8))
        frame.pack(fill=tk.X)
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(frame, variable=self._progress_var, maximum=100)
        self._progress_bar.pack(fill=tk.X)
        self._progress_label = ttk.Label(frame, text="ocioso")
        self._progress_label.pack(anchor="w", pady=(2, 0))

    def _build_table(self) -> None:
        frame = ttk.LabelFrame(self, text="Últimas letras processadas", padding=(8, 4))
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        columns = ("status", "source", "artist", "title")
        self._tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        self._tree.heading("status", text="Status")
        self._tree.heading("source", text="Fonte")
        self._tree.heading("artist", text="Artista")
        self._tree.heading("title", text="Música")
        self._tree.column("status", width=70, anchor="center")
        self._tree.column("source", width=120, anchor="w")
        self._tree.column("artist", width=200, anchor="w")
        self._tree.column("title", width=280, anchor="w")

        scroll = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<Double-1>", self._show_selected_lyric)

    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self, text="Log", padding=(8, 4))
        frame.pack(fill=tk.X, padx=12, pady=(0, 12))
        self._log = scrolledtext.ScrolledText(frame, height=5, wrap=tk.WORD)
        self._log.pack(fill=tk.BOTH, expand=True)
        self._log.configure(state="disabled")

    # --------------------------------------------------------------- actions

    def _open_cache_folder(self) -> None:
        folder = self._cache_path.resolve().parent
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Erro", f"Nao consegui abrir a pasta: {exc}")

    def _copy_cache_path(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(str(self._cache_path.resolve()))
        self._log_line(f"Caminho copiado: {self._cache_path.resolve()}")

    def _start_batch(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Em andamento", "Um batch ja esta rodando.")
            return
        if not self._parquet_path.exists():
            messagebox.showerror(
                "Parquet ausente",
                f"Nao achei {self._parquet_path}.\n"
                "Rode primeiro o notebook 04 para gerar o dataset curado.",
            )
            return

        limit = max(1, int(self._limit_var.get()))
        concurrency = max(1, int(self._concurrency_var.get()))
        retry_errors = bool(self._retry_errors_var.get())

        self._stop_flag.clear()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress_var.set(0)
        self._progress_label.configure(text="iniciando…")
        self._log_line(
            f"▶ Batch iniciado: limite={limit}, concurrency={concurrency}, "
            f"retry_errors={retry_errors}"
        )

        self._worker_thread = threading.Thread(
            target=self._worker_main,
            args=(limit, concurrency, retry_errors),
            daemon=True,
        )
        self._worker_thread.start()

    def _stop_batch(self) -> None:
        self._stop_flag.set()
        self._stop_btn.configure(state="disabled")
        self._progress_label.configure(text="parando após próximas faixas em vôo…")
        self._log_line("■ Parada pedida — aguardando faixas em vôo terminarem.")

    def _show_selected_lyric(self, _event: object) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        tags = self._tree.item(sel[0], "tags")
        track_id = tags[0] if tags else None
        if not track_id:
            return
        cache = LyricsCache(self._cache_path)
        row = cache.get(track_id)
        cache.close()
        if not row:
            return
        win = tk.Toplevel(self)
        win.title(f"{row['artist']} — {row['title']}")
        win.geometry("640x520")
        header = (
            f"Status: {row['status']}    Fonte: {row.get('source') or '—'}\n"
            f"URL: {row.get('source_url') or '—'}\n"
            f"Track ID: {row['track_id']}"
        )
        ttk.Label(win, text=header, padding=12, justify="left").pack(anchor="w")
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        text.insert("1.0", row.get("lyrics") or row.get("error") or "(vazio)")
        text.configure(state="disabled")

    # ----------------------------------------------------- worker / refresh

    def _worker_main(self, limit: int, concurrency: int, retry_errors: bool) -> None:
        try:
            asyncio.run(self._run_async(limit, concurrency, retry_errors))
        except Exception as exc:  # pragma: no cover
            self._event_queue.put(("log", f"erro no worker: {exc!r}"))
        finally:
            self._event_queue.put(("done", None))

    async def _run_async(self, limit: int, concurrency: int, retry_errors: bool) -> None:
        cache = LyricsCache(self._cache_path)
        rows = read_tracks(self._parquet_path, limit=None)
        pending = [
            r
            for r in rows
            if not cache.has_resolved(r["track_id"], retry_errors=retry_errors)
        ][:limit]

        if not pending:
            self._event_queue.put(("log", "Nada pendente — todas as faixas ja foram tentadas."))
            cache.close()
            return

        self._event_queue.put(
            ("progress", {"done": 0, "total": len(pending), "label": "rodando…"})
        )

        timeout = httpx.Timeout(20.0, connect=10.0)
        limits = httpx.Limits(
            max_connections=concurrency * 2,
            max_keepalive_connections=concurrency,
        )
        headers = {"User-Agent": "music-search-engine-lyrics/0.1 (UI Tk)"}

        cache_lock = asyncio.Lock()

        async with httpx.AsyncClient(
            timeout=timeout, limits=limits, headers=headers, follow_redirects=True
        ) as client:
            sources: list[LyricsSource] = list(_build_sources(client))
            self._event_queue.put(
                ("log", "Fontes: " + " → ".join(s.name for s in sources))
            )
            sem = asyncio.Semaphore(concurrency)

            done_count = {"n": 0}

            async def _process(track: dict) -> None:
                if self._stop_flag.is_set():
                    return
                async with sem:
                    if self._stop_flag.is_set():
                        return
                    artist = normalize_artist(track.get("primary_artist_name") or "")
                    title = normalize_title(track.get("track_name") or "")
                    if not artist or not title:
                        async with cache_lock:
                            cache.upsert(
                                track_id=track["track_id"],
                                isrc=track.get("isrc"),
                                artist=artist,
                                title=title,
                                status=Status.MISS.value,
                                error="empty artist or title",
                            )
                        self._after_track(done_count, len(pending), track, "miss", None)
                        return
                    final = None
                    for source in sources:
                        if self._stop_flag.is_set():
                            return
                        result = await source.fetch(artist, title)
                        final = result
                        if result.status == Status.HIT:
                            async with cache_lock:
                                cache.upsert(
                                    track_id=track["track_id"],
                                    isrc=track.get("isrc"),
                                    artist=artist,
                                    title=title,
                                    status=Status.HIT.value,
                                    source=result.source,
                                    source_url=result.source_url,
                                    lyrics=result.lyrics,
                                )
                            self._after_track(
                                done_count, len(pending), track, "hit", result.source
                            )
                            return
                        if result.status == Status.ERROR:
                            # 1 retry simples
                            await asyncio.sleep(1.0)
                            result = await source.fetch(artist, title)
                            final = result
                            if result.status == Status.HIT:
                                async with cache_lock:
                                    cache.upsert(
                                        track_id=track["track_id"],
                                        isrc=track.get("isrc"),
                                        artist=artist,
                                        title=title,
                                        status=Status.HIT.value,
                                        source=result.source,
                                        source_url=result.source_url,
                                        lyrics=result.lyrics,
                                    )
                                self._after_track(
                                    done_count, len(pending), track, "hit", result.source
                                )
                                return
                    status = (final.status if final else Status.ERROR).value
                    async with cache_lock:
                        cache.upsert(
                            track_id=track["track_id"],
                            isrc=track.get("isrc"),
                            artist=artist,
                            title=title,
                            status=status,
                            source=final.source if final else None,
                            source_url=final.source_url if final else None,
                            error=final.error if final else "no source",
                        )
                    self._after_track(
                        done_count, len(pending), track, status, final.source if final else None
                    )

            tasks = [asyncio.create_task(_process(t)) for t in pending]
            await asyncio.gather(*tasks, return_exceptions=True)

        cache.close()

    def _after_track(
        self,
        done_count: dict,
        total: int,
        track: dict,
        status: str,
        source: str | None,
    ) -> None:
        done_count["n"] += 1
        self._event_queue.put(
            (
                "progress",
                {"done": done_count["n"], "total": total, "label": "rodando…"},
            )
        )
        self._event_queue.put(
            (
                "track",
                {
                    "track_id": track["track_id"],
                    "artist": track.get("primary_artist_name") or "",
                    "title": track.get("track_name") or "",
                    "status": status,
                    "source": source or "",
                },
            )
        )

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self._event_queue.get_nowait()
                if kind == "progress":
                    done, total = payload["done"], payload["total"]
                    pct = (100 * done / total) if total else 0
                    self._progress_var.set(pct)
                    self._progress_label.configure(
                        text=f"{payload['label']}  {done}/{total}  ({pct:.1f}%)"
                    )
                elif kind == "track":
                    self._append_track_row(payload)
                elif kind == "log":
                    self._log_line(payload)
                elif kind == "done":
                    self._on_worker_done()
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _refresh_status(self) -> None:
        try:
            cache = LyricsCache(self._cache_path)
            stats = cache.stats()
            cache.close()
        except Exception as exc:
            self._status_var.set(f"(erro lendo cache: {exc!r})")
            self.after(_REFRESH_INTERVAL_MS, self._refresh_status)
            return

        if self._total_tracks is None and self._parquet_path.exists():
            try:
                self._total_tracks = len(read_tracks(self._parquet_path, limit=None))
            except Exception:
                self._total_tracks = None

        resolved = sum(stats.values())
        total_str = f"{self._total_tracks:,}" if self._total_tracks else "?"
        pending = (self._total_tracks - resolved) if self._total_tracks else None
        pending_str = f"{pending:,}" if pending is not None else "?"

        self._status_var.set(
            f"Total no parquet: {total_str}   |   "
            f"Pendentes: {pending_str}   |   "
            f"Hits: {stats.get('hit', 0):,}   "
            f"Miss: {stats.get('miss', 0):,}   "
            f"Erros: {stats.get('error', 0):,}   "
            f"Bloqueios: {stats.get('blocked', 0):,}"
        )
        self.after(_REFRESH_INTERVAL_MS, self._refresh_status)

    def _on_worker_done(self) -> None:
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._stop_flag.clear()
        self._progress_label.configure(text="ocioso")
        self._log_line("✓ Batch terminado.")

    def _append_track_row(self, payload: dict) -> None:
        # mantem so as ultimas 200 linhas
        children = self._tree.get_children()
        if len(children) > 200:
            for cid in children[: len(children) - 200]:
                self._tree.delete(cid)
        self._tree.insert(
            "",
            "end",
            values=(payload["status"], payload["source"], payload["artist"], payload["title"]),
            tags=(payload["track_id"],),
        )
        self._tree.yview_moveto(1.0)

    def _log_line(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")


def main() -> None:
    app = LyricsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
