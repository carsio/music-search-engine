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
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

import httpx

from music_search.lyrics.cache import LyricsCache
from music_search.lyrics.cli import _build_sources
from music_search.lyrics.pipeline import (
    PipelineConfig,
    _process_track,
    read_tracks,
)
from music_search.lyrics.sources.base import LyricsSource

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
        ttk.Checkbutton(frame, text="Re-tentar erros", variable=self._retry_errors_var).grid(
            row=0, column=4, padx=(0, 8)
        )

        self._retry_misses_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Re-tentar misses", variable=self._retry_misses_var).grid(
            row=0, column=5, padx=(0, 8)
        )

        self._retry_blocked_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Re-tentar blocked", variable=self._retry_blocked_var).grid(
            row=0, column=6, padx=(0, 16)
        )

        self._start_btn = ttk.Button(frame, text="▶ Iniciar", command=self._start_batch)
        self._start_btn.grid(row=0, column=7, padx=(0, 4))

        self._stop_btn = ttk.Button(
            frame, text="■ Parar", command=self._stop_batch, state="disabled"
        )
        self._stop_btn.grid(row=0, column=8)

        frame.columnconfigure(9, weight=1)

    def _build_progress(self) -> None:
        frame = ttk.Frame(self, padding=(12, 0, 12, 8))
        frame.pack(fill=tk.X)
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(frame, variable=self._progress_var, maximum=100)
        self._progress_bar.pack(fill=tk.X)
        self._progress_label = ttk.Label(frame, text="ocioso")
        self._progress_label.pack(anchor="w", pady=(2, 0))

    def _build_table(self) -> None:
        frame = ttk.LabelFrame(
            self,
            text="Últimas letras processadas (duplo-clique = log; Ctrl+C / Botão direito = copiar)",
            padding=(8, 4),
        )
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
        # Copia da celula sob o cursor (Ctrl+C copia a selecionada; clique direito
        # copia a celula clicada). Em macOS o usuario costuma usar Cmd em vez de Ctrl.
        self._tree.bind("<Control-c>", self._copy_selected_row)
        self._tree.bind("<Control-C>", self._copy_selected_row)
        self._tree.bind("<Command-c>", self._copy_selected_row)
        self._tree.bind("<Button-3>", self._show_row_context_menu)
        # No macOS o "right click" costuma ser Button-2 ou Control-Click.
        self._tree.bind("<Button-2>", self._show_row_context_menu)
        self._tree.bind("<Control-Button-1>", self._show_row_context_menu)

        self._row_menu = tk.Menu(self._tree, tearoff=0)
        self._row_menu_target: tuple[str, str] | None = None  # (item_id, column)
        self._row_menu.add_command(label="Copiar célula", command=self._copy_menu_cell)
        self._row_menu.add_command(label="Copiar linha (TSV)", command=self._copy_menu_row)
        self._row_menu.add_separator()
        self._row_menu.add_command(label="Abrir log da busca", command=self._open_menu_lyric)

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
        retry_misses = bool(self._retry_misses_var.get())
        retry_blocked = bool(self._retry_blocked_var.get())

        self._stop_flag.clear()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress_var.set(0)
        self._progress_label.configure(text="iniciando…")
        self._log_line(
            f"▶ Batch iniciado: limite={limit}, concurrency={concurrency}, "
            f"retry_errors={retry_errors}, retry_misses={retry_misses}, "
            f"retry_blocked={retry_blocked}"
        )

        self._worker_thread = threading.Thread(
            target=self._worker_main,
            args=(limit, concurrency, retry_errors, retry_misses, retry_blocked),
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
        self._open_lyric_window(sel[0])

    def _open_lyric_window(self, item_id: str) -> None:
        tags = self._tree.item(item_id, "tags")
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
        win.geometry("820x640")

        header_text = (
            f"Status:     {row['status']}\n"
            f"Fonte:      {row.get('source') or '—'}\n"
            f"URL:        {row.get('source_url') or '—'}\n"
            f"Track ID:   {row['track_id']}\n"
            f"Tentativas: {row.get('attempts') or '—'}\n"
            f"Resolvido:  {self._fmt_timestamp(row.get('fetched_at'))}"
        )
        ttk.Label(win, text=header_text, padding=12, justify="left", font=("Courier", 9)).pack(
            anchor="w"
        )

        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # --- Aba 1: letra / erro
        lyrics_frame = ttk.Frame(notebook)
        notebook.add(lyrics_frame, text="Letra")
        lyrics_text = scrolledtext.ScrolledText(lyrics_frame, wrap=tk.WORD)
        lyrics_text.pack(fill=tk.BOTH, expand=True)
        lyrics_text.insert("1.0", row.get("lyrics") or row.get("error") or "(vazio)")
        lyrics_text.configure(state="disabled")

        # --- Aba 2: log da busca (trace)
        trace_frame = ttk.Frame(notebook)
        notebook.add(trace_frame, text="Log da busca")
        trace_text = scrolledtext.ScrolledText(trace_frame, wrap=tk.NONE, font=("Courier", 9))
        trace_text.pack(fill=tk.BOTH, expand=True)
        trace_text.insert("1.0", self._format_trace(row.get("trace"), row))
        trace_text.configure(state="disabled")

        # --- Aba 3: trace cru (JSON)
        raw_frame = ttk.Frame(notebook)
        notebook.add(raw_frame, text="Trace JSON")
        raw_text = scrolledtext.ScrolledText(raw_frame, wrap=tk.NONE, font=("Courier", 9))
        raw_text.pack(fill=tk.BOTH, expand=True)
        raw = row.get("trace") or "[]"
        try:
            pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            pretty = raw
        raw_text.insert("1.0", pretty)
        raw_text.configure(state="disabled")

    @staticmethod
    def _fmt_timestamp(ts: Any) -> str:
        if not ts:
            return "—"
        try:
            return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def _format_trace(trace_json: str | None, row: dict) -> str:
        """Formata o trace JSON num bloco legivel, uma tentativa por bloco."""
        if not trace_json:
            return (
                "(sem trace registrado — provavelmente faixa processada antes da migration "
                "que adicionou a coluna `trace`. Reprocesse com Re-tentar misses/errors para "
                "popular o log.)"
            )
        try:
            trace = json.loads(trace_json)
        except (TypeError, ValueError):
            return f"(trace invalido)\n\n{trace_json}"

        if not isinstance(trace, list) or not trace:
            return "(trace vazio)"

        lines: list[str] = []
        lines.append(f"Total de tentativas registradas: {len(trace)}")
        lines.append("=" * 80)
        for i, raw_entry in enumerate(trace, 1):
            if not isinstance(raw_entry, dict):
                continue
            # cast pra dict[str, Any] — JSON garantiu que e dict, e ty nao
            # consegue inferir os tipos de valor a partir de um isinstance check.
            entry = {str(k): v for k, v in raw_entry.items()}
            ts = entry.get("ts")
            ts_fmt = (
                datetime.fromtimestamp(int(ts)).strftime("%H:%M:%S") if isinstance(ts, int) else "—"
            )
            status = str(entry.get("status", "?")).upper()
            source = entry.get("source") or "(pre-fetch)"
            elapsed = entry.get("elapsed_ms")
            elapsed_str = f"{elapsed} ms" if isinstance(elapsed, int) else "—"
            attempts = entry.get("attempts")
            chars = entry.get("lyrics_chars") or 0

            lines.append(
                f"[{i:>2}] {ts_fmt}  {status:<7}  fonte={source:<14}  "
                f"tempo={elapsed_str:<8}  attempts={attempts}"
            )
            lines.append(f"     query: artist={entry.get('query_artist')!r}")
            lines.append(f"            title ={entry.get('query_title')!r}")
            raw_title = entry.get("raw_title")
            query_title = entry.get("query_title")
            if raw_title and raw_title != query_title:
                lines.append(f"     (raw_title original: {raw_title!r})")
            source_url = entry.get("source_url")
            if source_url:
                lines.append(f"     url:    {source_url}")
            err = entry.get("error")
            if err:
                lines.append(f"     erro:   {err}")
            if status == "HIT":
                lines.append(f"     letra:  {chars} chars")
            lines.append("")
        return "\n".join(lines)

    # ----------------------------------------------------------- copy helpers

    def _column_under_event(self, event: Any) -> str | None:
        """Devolve a coluna ('#1', '#2', ...) na qual o evento ocorreu, ou None."""
        try:
            col = self._tree.identify_column(event.x)
        except (AttributeError, tk.TclError):
            return None
        return col or None

    def _row_under_event(self, event: Any) -> str | None:
        try:
            row = self._tree.identify_row(event.y)
        except (AttributeError, tk.TclError):
            return None
        return row or None

    def _cell_value(self, item_id: str, column: str | None) -> str:
        """Devolve o valor textual de uma celula. column = '#1'..'#N' ou None
        (entao devolve a linha inteira como TSV)."""
        if not item_id:
            return ""
        values = self._tree.item(item_id, "values")
        if not values:
            return ""
        if column and column.startswith("#"):
            try:
                idx = int(column[1:]) - 1
            except ValueError:
                idx = -1
            if 0 <= idx < len(values):
                return str(values[idx])
        return "\t".join(str(v) for v in values)

    def _copy_to_clipboard(self, text: str) -> None:
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        # update() para garantir que a clipboard sobreviva apos o app fechar.
        self.update()
        preview = text if len(text) <= 80 else text[:77] + "…"
        self._log_line(f"📋 Copiado: {preview}")

    def _copy_selected_row(self, _event: Any = None) -> str | None:
        sel = self._tree.selection()
        if not sel:
            return None
        # Quando dispara via Ctrl+C nao temos uma celula especifica — copia a linha.
        text = self._cell_value(sel[0], None)
        self._copy_to_clipboard(text)
        return "break"

    def _show_row_context_menu(self, event: Any) -> None:
        item_id = self._row_under_event(event)
        column = self._column_under_event(event)
        if not item_id:
            return
        # Foca a linha clicada para feedback visual.
        self._tree.selection_set(item_id)
        self._tree.focus(item_id)
        self._row_menu_target = (item_id, column or "")
        try:
            self._row_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._row_menu.grab_release()

    def _copy_menu_cell(self) -> None:
        if not self._row_menu_target:
            return
        item_id, column = self._row_menu_target
        text = self._cell_value(item_id, column or None)
        self._copy_to_clipboard(text)

    def _copy_menu_row(self) -> None:
        if not self._row_menu_target:
            return
        item_id, _ = self._row_menu_target
        text = self._cell_value(item_id, None)
        self._copy_to_clipboard(text)

    def _open_menu_lyric(self) -> None:
        if not self._row_menu_target:
            return
        item_id, _ = self._row_menu_target
        self._open_lyric_window(item_id)

    # ----------------------------------------------------- worker / refresh

    def _worker_main(
        self,
        limit: int,
        concurrency: int,
        retry_errors: bool,
        retry_misses: bool,
        retry_blocked: bool,
    ) -> None:
        try:
            asyncio.run(
                self._run_async(limit, concurrency, retry_errors, retry_misses, retry_blocked)
            )
        except Exception as exc:  # pragma: no cover
            self._event_queue.put(("log", f"erro no worker: {exc!r}"))
        finally:
            self._event_queue.put(("done", None))

    async def _run_async(
        self,
        limit: int,
        concurrency: int,
        retry_errors: bool,
        retry_misses: bool,
        retry_blocked: bool,
    ) -> None:
        cache = LyricsCache(self._cache_path)
        rows = read_tracks(self._parquet_path, limit=None)
        pending = [
            r
            for r in rows
            if not cache.has_resolved(
                r["track_id"],
                retry_errors=retry_errors,
                retry_misses=retry_misses,
                retry_blocked=retry_blocked,
            )
        ][:limit]

        if not pending:
            self._event_queue.put(("log", "Nada pendente — todas as faixas ja foram tentadas."))
            cache.close()
            return

        self._event_queue.put(("progress", {"done": 0, "total": len(pending), "label": "rodando…"}))

        cfg = PipelineConfig(
            parquet_path=self._parquet_path,
            cache_path=self._cache_path,
            concurrency=concurrency,
            request_timeout=20.0,
            max_retries=2,
            retry_errors=retry_errors,
            retry_misses=retry_misses,
            retry_blocked=retry_blocked,
            limit=limit,
        )

        timeout = httpx.Timeout(cfg.request_timeout, connect=10.0)
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
            self._event_queue.put(("log", "Fontes: " + " → ".join(s.name for s in sources)))
            sem = asyncio.Semaphore(concurrency)

            done_count = {"n": 0}

            async def _process(track: dict) -> None:
                # Honra o stop flag antes de pegar o semaphore — evita iniciar
                # novos requests apos o usuario clicar em Parar.
                if self._stop_flag.is_set():
                    return
                # Reutiliza _process_track do pipeline: ele ja faz variantes,
                # retries com backoff e gravacao do trace estruturado no cache.
                status = await _process_track(track, sources, cache, sem, cfg, cache_lock)
                # Le source de volta do cache (o pipeline grava la com trace).
                row = cache.get(track["track_id"]) or {}
                self._after_track(
                    done_count,
                    len(pending),
                    track,
                    status,
                    row.get("source"),
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
