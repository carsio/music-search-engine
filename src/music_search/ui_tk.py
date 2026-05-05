"""GUI Tkinter para comparar BM25 e TF-IDF no corpus curado brasileiro.

Uso:

    uv run python -m music_search.ui_tk
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Literal

from music_search.search import (
    DEFAULT_FIELD_WEIGHTS,
    FIELD_LABELS,
    SearchAlgorithm,
    SearchHit,
    SparseSearchEngine,
    load_or_build_default_engine,
)

_Anchor = Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"]

_WINDOW_TITLE = "Busca Brasileira com Letras — BM25 x TF-IDF"
_WINDOW_SIZE = "1560x860"
_DEFAULT_TOP_K = 10
_ALGORITHMS: tuple[SearchAlgorithm, SearchAlgorithm] = ("bm25", "tfidf")
_COLUMNS: list[tuple[str, str, int, _Anchor]] = [
    ("rank", "#", 44, "center"),
    ("score", "Score", 70, "center"),
    ("track_name", "Música", 210, "w"),
    ("artist_names", "Artistas", 180, "w"),
    ("artist_genres", "Gêneros", 180, "w"),
    ("lyrics_preview", "Trecho", 320, "w"),
]


def _fmt_duration(ms: int) -> str:
    mins = ms // 60_000
    secs = (ms % 60_000) // 1_000
    return f"{mins}:{secs:02d}"


def _row_values(hit: SearchHit) -> tuple[object, ...]:
    return (
        hit.rank,
        f"{hit.score:.4f}",
        hit.track_name,
        hit.artist_names or hit.primary_artist_name,
        hit.artist_genres or hit.macro_genre or "—",
        hit.lyrics_preview or "—",
    )


class SparseSearchApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(_WINDOW_TITLE)
        self.geometry(_WINDOW_SIZE)
        self.minsize(1100, 680)

        self._engine: SparseSearchEngine | None = None
        self._results: dict[SearchAlgorithm, list[SearchHit]] = {"bm25": [], "tfidf": []}
        self._trees: dict[SearchAlgorithm, ttk.Treeview] = {}
        self._field_vars = {field: tk.BooleanVar(value=True) for field in DEFAULT_FIELD_WEIGHTS}

        self._query_var = tk.StringVar()
        self._top_k_var = tk.IntVar(value=_DEFAULT_TOP_K)
        self._status_var = tk.StringVar(value="Carregando corpus curado e índice...")
        self._summary_var = tk.StringVar(value="")

        self._build_ui()
        self._set_busy(True)
        threading.Thread(target=self._load_engine_thread, daemon=True).start()

    def _build_ui(self) -> None:
        self._build_search_bar()
        self._build_field_panel()
        self._build_results_panel()
        self._build_status_bar()

    def _build_search_bar(self) -> None:
        frame = ttk.Frame(self, padding=(14, 12, 14, 8))
        frame.pack(fill=tk.X)

        title = ttk.Label(
            frame,
            text="Corpus curado de músicas brasileiras com letras",
            font=("Segoe UI", 15, "bold"),
        )
        title.grid(row=0, column=0, columnspan=6, sticky="w")

        subtitle = ttk.Label(
            frame,
            text="Compare BM25 e TF-IDF sobre título, gêneros, metadados e principalmente a letra.",
        )
        subtitle.grid(row=1, column=0, columnspan=6, sticky="w", pady=(2, 10))

        ttk.Label(frame, text="Consulta:").grid(row=2, column=0, sticky="w")
        self._entry = ttk.Entry(frame, textvariable=self._query_var, font=("Segoe UI", 11))
        self._entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(8, 10))
        self._entry.bind("<Return>", lambda _e: self._on_search())

        ttk.Label(frame, text="Top:").grid(row=2, column=4, sticky="e")
        self._top_spin = ttk.Spinbox(frame, from_=1, to=100, textvariable=self._top_k_var, width=6)
        self._top_spin.grid(row=2, column=5, sticky="w", padx=(8, 10))

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=6, sticky="w", pady=(10, 0))
        self._search_button = ttk.Button(button_frame, text="Buscar", command=self._on_search)
        self._search_button.pack(side=tk.LEFT)
        self._rebuild_button = ttk.Button(
            button_frame,
            text="Reconstruir índice",
            command=lambda: self._reload_engine(force_rebuild=True),
        )
        self._rebuild_button.pack(side=tk.LEFT, padx=(8, 0))

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=1)

    def _build_field_panel(self) -> None:
        frame = ttk.LabelFrame(self, text="Campos usados na consulta", padding=(14, 10))
        frame.pack(fill=tk.X, padx=14, pady=(0, 8))

        for idx, (field, weight) in enumerate(DEFAULT_FIELD_WEIGHTS.items()):
            text = f"{FIELD_LABELS.get(field, field)}  (peso {weight:.2f})"
            ttk.Checkbutton(frame, text=text, variable=self._field_vars[field]).grid(
                row=idx // 3,
                column=idx % 3,
                sticky="w",
                padx=(0, 16),
                pady=2,
            )

        ttk.Label(
            frame,
            textvariable=self._summary_var,
            foreground="#4a4a4a",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _build_results_panel(self) -> None:
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 12))

        for algorithm in _ALGORITHMS:
            frame = ttk.LabelFrame(paned, text=algorithm.upper(), padding=(8, 6))
            paned.add(frame, weight=1)

            cols = [column[0] for column in _COLUMNS]
            tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")
            for col_id, heading, width, anchor in _COLUMNS:
                tree.heading(col_id, text=heading)
                tree.column(
                    col_id,
                    width=width,
                    minwidth=40,
                    anchor=anchor,
                    stretch=(col_id == "lyrics_preview"),
                )

            vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)

            tree.bind("<Double-1>", lambda _e, algo=algorithm: self._show_selected_hit(algo))
            self._trees[algorithm] = tree

    def _build_status_bar(self) -> None:
        bar = ttk.Label(
            self,
            textvariable=self._status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(8, 2),
        )
        bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _load_engine_thread(self) -> None:
        try:
            engine = load_or_build_default_engine()
            self.after(0, self._on_engine_ready, engine)
        except Exception as exc:
            self.after(0, self._show_error, f"falha ao carregar o motor: {exc}")

    def _reload_engine(self, *, force_rebuild: bool) -> None:
        self._set_busy(True)
        self._status_var.set("Reconstruindo índice do corpus curado...")

        def _worker() -> None:
            try:
                engine = load_or_build_default_engine(rebuild_index=force_rebuild)
                self.after(0, self._on_engine_ready, engine)
            except Exception as exc:
                self.after(0, self._show_error, f"falha ao reconstruir o índice: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_engine_ready(self, engine: SparseSearchEngine) -> None:
        self._engine = engine
        self._summary_var.set(f"Corpus carregado: {engine.num_docs:,} músicas com letra.")
        self._status_var.set("Pronto para buscar.")
        self._set_busy(False)
        self._entry.focus()

    def _on_search(self) -> None:
        query = self._query_var.get().strip()
        if not query:
            messagebox.showwarning("Aviso", "Digite um texto para pesquisar.")
            return
        if self._engine is None:
            messagebox.showinfo("Aguarde", "O índice ainda está carregando.")
            return
        engine = self._engine

        weights = self._selected_field_weights()
        if not weights:
            messagebox.showwarning("Aviso", "Selecione ao menos um campo para buscar.")
            return

        top_k = self._top_k_var.get()
        self._set_busy(True)
        self._status_var.set(f'Buscando "{query}"...')
        self._clear_tables()

        def _worker() -> None:
            started = time.perf_counter()
            try:
                results = engine.search_both(query, top_k=top_k, field_weights=weights)
                elapsed_ms = (time.perf_counter() - started) * 1000
                self.after(0, self._on_search_done, results, query, elapsed_ms)
            except Exception as exc:
                self.after(0, self._show_error, f"falha na busca: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_search_done(
        self,
        results: dict[SearchAlgorithm, list[SearchHit]],
        query: str,
        elapsed_ms: float,
    ) -> None:
        self._results = results
        for algorithm, hits in results.items():
            tree = self._trees[algorithm]
            for index, hit in enumerate(hits):
                tree.insert("", tk.END, iid=str(index), values=_row_values(hit))

        summary = " | ".join(f"{algo.upper()}: {len(hits)}" for algo, hits in results.items())
        self._status_var.set(f'Consulta "{query}" concluída em {elapsed_ms:.0f} ms. {summary}.')
        self._set_busy(False)

    def _selected_field_weights(self) -> dict[str, float]:
        return {
            field: weight
            for field, weight in DEFAULT_FIELD_WEIGHTS.items()
            if self._field_vars[field].get()
        }

    def _show_selected_hit(self, algorithm: SearchAlgorithm) -> None:
        selection = self._trees[algorithm].selection()
        if not selection:
            return
        index = int(selection[0])
        hits = self._results.get(algorithm, [])
        if not 0 <= index < len(hits):
            return
        self._show_hit_detail(hits[index])

    def _show_hit_detail(self, hit: SearchHit) -> None:
        win = tk.Toplevel(self)
        win.title(f"{hit.track_name} — {hit.algorithm.upper()}")
        win.geometry("920x760")
        win.minsize(760, 520)

        text = tk.Text(win, wrap=tk.WORD, font=("Consolas", 10), padx=12, pady=12)
        sb = ttk.Scrollbar(win, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)

        lines = [
            f"Algoritmo : {hit.algorithm.upper()}",
            f"Rank      : #{hit.rank}",
            f"Score     : {hit.score:.6f}",
            "",
            f"Música    : {hit.track_name}",
            f"Artista   : {hit.primary_artist_name or hit.artist_names or '—'}",
            f"Artistas  : {hit.artist_names or '—'}",
            f"Gêneros   : {hit.artist_genres or '—'}",
            f"Macro     : {hit.macro_genre or '—'}",
            f"Álbum     : {hit.album_name or '—'}",
            f"Lançamento: {hit.release_date or hit.release_year or '—'}",
            f"Popularid.: {hit.track_popularity}/100",
            f"Duração   : {_fmt_duration(hit.duration_ms)}",
            f"Explícita : {'sim' if hit.explicit else 'não'}",
            f"Fonte letra: {hit.lyrics_source or '—'}",
            f"URL letra : {hit.lyrics_source_url or '—'}",
            "",
            "Contribuições por campo:",
        ]

        for contribution in hit.field_scores:
            lines.append(
                f"  - {contribution.label:<12} raw={contribution.raw_score:.6f}  "
                f"norm={contribution.normalized_score:.6f}  "
                f"peso={contribution.weight:.2f}  "
                f"final={contribution.weighted_score:.6f}"
            )

        lines.extend(["", "Letra:", hit.lyrics or "(sem letra)"])

        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)

    def _clear_tables(self) -> None:
        for tree in self._trees.values():
            tree.delete(*tree.get_children())
        self._results = {"bm25": [], "tfidf": []}

    def _show_error(self, message: str) -> None:
        self._status_var.set(f"Erro: {message}")
        self._set_busy(False)
        messagebox.showerror("Erro", message)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self._search_button.configure(state=state)
        self._rebuild_button.configure(state=state)
        self._entry.configure(state=state)
        self._top_spin.configure(state=state)


def main() -> None:
    app = SparseSearchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
