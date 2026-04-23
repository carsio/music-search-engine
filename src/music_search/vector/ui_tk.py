"""Interface Tkinter para a busca vetorial — ferramenta opcional de debug local.

Não é a UI oficial do projeto (essa será a web FastAPI). Serve só para inspecionar
os resultados da busca semântica interativamente sem precisar subir servidor.

Uso:

    uv run python -m music_search.vector.ui_tk
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Literal

from music_search.vector import VectorSearch

_WINDOW_TITLE = "Spotify Search — Busca Semântica"
_WINDOW_SIZE = "1200x680"
_DEFAULT_TOP_K = 20

_Anchor = Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"]

_COLUMNS: list[tuple[str, str, int, _Anchor]] = [
    ("rank", "#", 45, "center"),
    ("score", "Score", 70, "center"),
    ("track_name", "Música", 260, "w"),
    ("artist_names", "Artistas", 200, "w"),
    ("album_name", "Álbum", 200, "w"),
    ("release_date", "Lançamento", 90, "center"),
    ("artist_genres", "Gêneros", 180, "w"),
    ("track_popularity", "Pop.", 50, "center"),
    ("duration", "Duração", 65, "center"),
    ("explicit", "Exp.", 45, "center"),
]


def _fmt_duration(ms: int) -> str:
    mins = ms // 60_000
    secs = (ms % 60_000) // 1_000
    return f"{mins}:{secs:02d}"


def _row_values(
    r: dict[str, Any],
) -> tuple[Any, str, str, str, str, str, str, Any, str, str]:
    return (
        r["rank"],
        f"{r['score']:.4f}",
        r["track_name"],
        r["artist_names"],
        r["album_name"],
        r["release_date"][:4] if r.get("release_date") else "—",
        r["artist_genres"] or "—",
        r["track_popularity"],
        _fmt_duration(r["duration_ms"]),
        "sim" if r["explicit"] else "não",
    )


class SpotifySearchApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(_WINDOW_TITLE)
        self.geometry(_WINDOW_SIZE)
        self.minsize(800, 500)

        self._search_engine = VectorSearch()
        self._results: list[dict[str, Any]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        self._build_search_bar()
        self._build_results_table()
        self._build_status_bar()

    def _build_search_bar(self) -> None:
        frame = ttk.Frame(self, padding=(12, 10, 12, 6))
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="Pesquisar:").pack(side=tk.LEFT)

        self._query_var = tk.StringVar()
        self._entry = ttk.Entry(frame, textvariable=self._query_var, font=("Segoe UI", 11))
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self._entry.bind("<Return>", lambda _e: self._on_search())
        self._entry.focus()

        ttk.Label(frame, text="Top:").pack(side=tk.LEFT)
        self._top_k_var = tk.IntVar(value=_DEFAULT_TOP_K)
        spin = ttk.Spinbox(frame, from_=1, to=100, textvariable=self._top_k_var, width=5)
        spin.pack(side=tk.LEFT, padx=(4, 8))

        self._btn_search = ttk.Button(frame, text="Buscar", command=self._on_search)
        self._btn_search.pack(side=tk.LEFT)

    def _build_results_table(self) -> None:
        frame = ttk.Frame(self, padding=(12, 0, 12, 0))
        frame.pack(fill=tk.BOTH, expand=True)

        cols = [c[0] for c in _COLUMNS]
        self._tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")

        for col_id, heading, width, anchor in _COLUMNS:
            self._tree.heading(col_id, text=heading, command=lambda c=col_id: self._sort_column(c))
            self._tree.column(
                col_id,
                width=width,
                minwidth=30,
                anchor=anchor,
                stretch=(col_id == "track_name"),
            )

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._tree.yview)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._tree.bind("<Double-1>", self._on_row_double_click)
        self._tree.tag_configure("odd", background="#f9f9f9")
        self._tree.tag_configure("even", background="#ffffff")

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Pronto.")
        bar = ttk.Label(
            self,
            textvariable=self._status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(8, 2),
        )
        bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _on_search(self) -> None:
        query = self._query_var.get().strip()
        if not query:
            messagebox.showwarning("Aviso", "Digite um texto para pesquisar.")
            return

        top_k = self._top_k_var.get()
        self._set_searching(True)
        self._status_var.set(f'Buscando: "{query}" …')
        self._clear_table()

        thread = threading.Thread(target=self._search_thread, args=(query, top_k), daemon=True)
        thread.start()

    def _search_thread(self, query: str, top_k: int) -> None:
        try:
            results = self._search_engine.search(query, top_k=top_k)
            self.after(0, self._populate_table, results, query)
        except Exception as exc:
            self.after(0, self._show_error, str(exc))

    def _populate_table(self, results: list[dict[str, Any]], query: str) -> None:
        self._clear_table()
        for i, r in enumerate(results):
            tag = "odd" if i % 2 else "even"
            self._tree.insert("", tk.END, iid=str(i), values=_row_values(r), tags=(tag,))

        self._results = results

        count = len(results)
        s = "s" if count != 1 else ""
        self._status_var.set(f'{count} música{s} encontrada{s} para "{query}".')
        self._set_searching(False)

    def _show_error(self, message: str) -> None:
        self._status_var.set(f"Erro: {message}")
        self._set_searching(False)
        messagebox.showerror("Erro na busca", message)

    def _clear_table(self) -> None:
        self._tree.delete(*self._tree.get_children())
        self._results = []

    def _set_searching(self, searching: bool) -> None:
        state = tk.DISABLED if searching else tk.NORMAL
        self._btn_search.config(state=state)
        self._entry.config(state=state)

    def _sort_column(self, col: str) -> None:
        items = [(self._tree.set(iid, col), iid) for iid in self._tree.get_children()]
        try:
            items.sort(key=lambda t: float(t[0]))
        except ValueError:
            items.sort(key=lambda t: t[0].lower())
        for index, (_, iid) in enumerate(items):
            self._tree.move(iid, "", index)
            tag = "odd" if index % 2 else "even"
            self._tree.item(iid, tags=(tag,))

    def _on_row_double_click(self, _event: object) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self._results):
            return
        self._show_detail(self._results[idx])

    def _show_detail(self, r: dict[str, Any]) -> None:
        win = tk.Toplevel(self)
        win.title(f"{r['track_name']} — detalhe")
        win.geometry("520x460")
        win.resizable(True, True)

        text = tk.Text(win, wrap=tk.WORD, font=("Consolas", 10), padx=10, pady=10)
        sb = ttk.Scrollbar(win, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)

        lines = [
            f"Rank    : #{r['rank']}",
            f"Score   : {r['score']:.6f}",
            "",
            f"Música  : {r['track_name']}",
            f"Artistas: {r['artist_names']}",
            f"Álbum   : {r['album_name']}  ({r.get('release_date', '?')[:4]})",
            f"Tipo    : {r['album_type'] or '—'}",
            f"Gêneros : {r['artist_genres'] or '—'}",
            f"Label   : {r['label'] or '—'}",
            f"Pop. faixa : {r['track_popularity']}/100",
            f"Pop. álbum : {r['album_popularity']}/100",
            f"Duração : {_fmt_duration(r['duration_ms'])}",
            f"Explícito: {'sim' if r['explicit'] else 'não'}",
        ]

        data = r.get("data_completa") or {}
        if data:
            lines += ["", "── Dados completos ──"]
            for k, v in data.items():
                lines.append(f"  {k}: {v}")

        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)


def main() -> None:
    app = SpotifySearchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
