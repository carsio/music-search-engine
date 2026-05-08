"""Interface Tkinter para orquestrar enrichment e pos-processamento completo.

Uso:
    uv run python -m music_search.enrichment.ui_tk

Fluxo coberto pela UI:
1. baixar documentos relacionados
2. normalizar com LLM
3. music_search.scripts.export_entities
4. music_search.scripts.build_dataset --skip-lyrics
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

from music_search._async_http.cache import KeyValueCache
from music_search.enrichment.pipeline import DEFAULT_CACHE_PATH, DEFAULT_CACHE_TABLE

_WINDOW_TITLE = "Enrichment - fluxo completo"
_WINDOW_SIZE = "1180x860"
_STATUS_REFRESH_INTERVAL_MS = 900
_ARTIFACT_REFRESH_INTERVAL_MS = 4500
_CACHE_DETAILS_LIMIT = 1500
_DEFAULT_OUTPUT_DIR = Path("data/derived/final")

_KIND_ORDER = ("artists", "albums", "genres", "composers")
_KIND_TO_SINGULAR = {
    "artists": "artist",
    "albums": "album",
    "genres": "genre",
    "composers": "composer",
}
_KIND_LABEL = {
    "artists": "Artistas",
    "albums": "Albuns",
    "genres": "Generos",
    "composers": "Compositores",
}
_DEFAULT_LIMIT = {
    "artists": "500",
    "albums": "500",
    "genres": "",
    "composers": "500",
}
_UNKNOWN = "-"

_ARTIFACT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("artists", "Parquet artists", "br_artists.parquet", "parquet"),
    ("albums", "Parquet albums", "br_albums.parquet", "parquet"),
    ("genres", "Parquet genres", "br_genres.parquet", "parquet"),
    ("composers", "Parquet composers", "br_composers.parquet", "parquet"),
    ("manifest", "Manifest", "br_dataset_manifest.json", "json"),
)


@dataclass(frozen=True)
class _Stage:
    label: str
    argv: list[str]


class EnrichmentApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(_WINDOW_TITLE)
        self.geometry(_WINDOW_SIZE)
        self.minsize(980, 680)

        self._project_root = Path(__file__).resolve().parents[3]
        self._cache_path = DEFAULT_CACHE_PATH
        self._output_dir = _DEFAULT_OUTPUT_DIR
        self._manifest_path = self._output_dir / "br_dataset_manifest.json"

        self._event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._proc_lock = threading.Lock()
        self._current_proc: subprocess.Popen[str] | None = None

        self._enabled_vars: dict[str, tk.BooleanVar] = {}
        self._limit_vars: dict[str, tk.StringVar] = {}
        self._stage_index_map: dict[int, str] = {}
        self._target_by_kind: dict[str, int | None] = {}
        self._total_by_kind: dict[str, int | None] = {}
        self._retry_errors_by_kind: dict[str, bool] = {}
        self._phase_selection_by_kind: dict[str, set[str]] = {}
        self._seed_total_thread: threading.Thread | None = None
        self._last_status_error: str | None = None
        self._last_artifact_error: str | None = None

        self._build_ui()
        self.after(100, self._poll_events)
        self.after(250, self._refresh_status)
        self.after(300, self._refresh_artifacts)
        self._refresh_seed_totals_async()

    def _build_ui(self) -> None:
        self._build_header()
        self._build_controls()
        self._build_progress()
        self._build_monitor_tabs()
        self._build_log()

    def _build_header(self) -> None:
        frame = ttk.Frame(self, padding=(12, 10, 12, 6))
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="Cache:").grid(row=0, column=0, sticky="w")
        self._cache_var = tk.StringVar(value=str(self._cache_path.resolve()))
        cache_entry = ttk.Entry(frame, textvariable=self._cache_var, width=88)
        cache_entry.grid(row=0, column=1, sticky="we", padx=(6, 6))
        cache_entry.configure(state="readonly")

        ttk.Label(frame, text="Output:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._output_var = tk.StringVar(value=str(self._output_dir.resolve()))
        output_entry = ttk.Entry(frame, textvariable=self._output_var, width=88)
        output_entry.grid(row=1, column=1, sticky="we", padx=(6, 6), pady=(6, 0))
        output_entry.configure(state="readonly")

        actions = ttk.Frame(frame)
        actions.grid(row=0, column=2, rowspan=2, sticky="ns")
        ttk.Button(actions, text="Abrir output", command=self._open_output_folder).pack(fill=tk.X)
        ttk.Button(actions, text="Atualizar artefatos", command=self._refresh_artifacts_once).pack(
            fill=tk.X, pady=(6, 0)
        )
        frame.columnconfigure(1, weight=1)

    def _build_controls(self) -> None:
        frame = ttk.LabelFrame(self, text="Pipeline", padding=(12, 8))
        frame.pack(fill=tk.X, padx=12, pady=(0, 8))

        ttk.Label(frame, text="Concurrency:").grid(row=0, column=0, sticky="w")
        self._concurrency_var = tk.IntVar(value=4)
        ttk.Spinbox(
            frame, from_=1, to=32, increment=1, width=5, textvariable=self._concurrency_var
        ).grid(row=0, column=1, padx=(6, 16), sticky="w")

        self._retry_errors_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Re-tentar erros", variable=self._retry_errors_var).grid(
            row=0, column=2, sticky="w", padx=(0, 16)
        )
        ttk.Label(frame, text="Limit vazio = sem limite").grid(row=0, column=3, sticky="w")

        ttk.Label(frame, text="Fases:").grid(row=0, column=4, sticky="w", padx=(16, 6))
        self._fetch_docs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="1. Baixar documentos", variable=self._fetch_docs_var).grid(
            row=0, column=5, sticky="w", padx=(0, 12)
        )
        self._normalize_llm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="2. Normalizar com LLM",
            variable=self._normalize_llm_var,
        ).grid(row=0, column=6, sticky="w")

        table = ttk.Frame(frame)
        table.grid(row=1, column=0, columnspan=8, sticky="we", pady=(8, 0))
        ttk.Label(table, text="Rodar").grid(row=0, column=0, sticky="w")
        ttk.Label(table, text="Entidade").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(table, text="Limit").grid(row=0, column=2, sticky="w", padx=(8, 0))

        for i, kind in enumerate(_KIND_ORDER, start=1):
            enabled_var = tk.BooleanVar(value=True)
            limit_var = tk.StringVar(value=_DEFAULT_LIMIT[kind])
            self._enabled_vars[kind] = enabled_var
            self._limit_vars[kind] = limit_var

            ttk.Checkbutton(table, variable=enabled_var).grid(row=i, column=0, sticky="w")
            ttk.Label(table, text=_KIND_LABEL[kind]).grid(row=i, column=1, sticky="w", padx=(8, 0))
            ttk.Entry(table, textvariable=limit_var, width=9).grid(
                row=i, column=2, sticky="w", padx=(8, 0)
            )

        self._export_var = tk.BooleanVar(value=False)
        self._build_dataset_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Rodar music_search.scripts.export_entities (apos normalizacao)",
            variable=self._export_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            frame,
            text="Rodar music_search.scripts.build_dataset --skip-lyrics (apos normalizacao)",
            variable=self._build_dataset_var,
        ).grid(row=3, column=0, columnspan=4, sticky="w")

        self._start_btn = ttk.Button(frame, text=">> Rodar pipeline", command=self._start_run)
        self._start_btn.grid(row=4, column=4, sticky="e", padx=(0, 6), pady=(8, 0))

        self._post_btn = ttk.Button(
            frame,
            text=">> Apenas export+build",
            command=self._start_postprocess,
        )
        self._post_btn.grid(row=4, column=5, sticky="e", padx=(0, 6), pady=(8, 0))

        self._stop_btn = ttk.Button(
            frame,
            text="[] Parar",
            command=self._stop_run,
            state="disabled",
        )
        self._stop_btn.grid(row=4, column=6, sticky="e", pady=(8, 0))

        frame.columnconfigure(7, weight=1)

    def _build_progress(self) -> None:
        frame = ttk.Frame(self, padding=(12, 0, 12, 8))
        frame.pack(fill=tk.X)
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(frame, variable=self._progress_var, maximum=1)
        self._progress_bar.pack(fill=tk.X)
        self._progress_label = ttk.Label(frame, text="ocioso")
        self._progress_label.pack(anchor="w", pady=(2, 0))

    def _build_monitor_tabs(self) -> None:
        wrap = ttk.Frame(self, padding=(12, 0, 12, 8))
        wrap.pack(fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(wrap)
        nb.pack(fill=tk.BOTH, expand=True)

        status_tab = ttk.Frame(nb, padding=(8, 6))
        stages_tab = ttk.Frame(nb, padding=(8, 6))
        artifacts_tab = ttk.Frame(nb, padding=(8, 6))
        nb.add(status_tab, text="Cache")
        nb.add(stages_tab, text="Etapas")
        nb.add(artifacts_tab, text="Artefatos")

        self._build_status_table(status_tab)
        self._build_stage_table(stages_tab)
        self._build_artifacts_table(artifacts_tab)

    def _build_status_table(self, parent: ttk.Frame) -> None:
        columns = (
            "kind",
            "total",
            "target",
            "pending",
            "fetched",
            "hit",
            "miss",
            "error",
            "blocked",
        )
        self._status_tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)

        table_wrap = ttk.Frame(parent)
        table_wrap.pack(fill=tk.BOTH, expand=True)
        self._status_tree.heading("kind", text="Entidade")
        self._status_tree.heading("total", text="Total")
        self._status_tree.heading("target", text="Limite")
        self._status_tree.heading("pending", text="Faltam")
        self._status_tree.heading("fetched", text="Bruto")
        self._status_tree.heading("hit", text="Hit")
        self._status_tree.heading("miss", text="Miss")
        self._status_tree.heading("error", text="Error")
        self._status_tree.heading("blocked", text="Blocked")

        self._status_tree.column("kind", width=180, anchor="w")
        self._status_tree.column("total", width=95, anchor="center")
        self._status_tree.column("target", width=95, anchor="center")
        self._status_tree.column("pending", width=95, anchor="center")
        for col in ("fetched", "hit", "miss", "error", "blocked"):
            self._status_tree.column(col, width=88, anchor="center")

        scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self._status_tree.yview)
        self._status_tree.configure(yscrollcommand=scroll.set)
        self._status_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._status_tree.bind("<Double-1>", self._on_status_row_double_click)

        for kind in _KIND_ORDER:
            self._status_tree.insert(
                "",
                "end",
                iid=kind,
                values=(_KIND_LABEL[kind], _UNKNOWN, _UNKNOWN, _UNKNOWN, 0, 0, 0, 0, 0),
            )

    def _build_stage_table(self, parent: ttk.Frame) -> None:
        columns = ("stage", "status", "exit", "started", "finished")
        self._stage_tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)

        self._stage_tree.heading("stage", text="Etapa")
        self._stage_tree.heading("status", text="Status")
        self._stage_tree.heading("exit", text="Exit")
        self._stage_tree.heading("started", text="Inicio")
        self._stage_tree.heading("finished", text="Fim")

        self._stage_tree.column("stage", width=420, anchor="w")
        self._stage_tree.column("status", width=120, anchor="center")
        self._stage_tree.column("exit", width=80, anchor="center")
        self._stage_tree.column("started", width=120, anchor="center")
        self._stage_tree.column("finished", width=120, anchor="center")

        scroll = ttk.Scrollbar(parent, orient="vertical", command=self._stage_tree.yview)
        self._stage_tree.configure(yscrollcommand=scroll.set)
        self._stage_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_artifacts_table(self, parent: ttk.Frame) -> None:
        columns = ("artifact", "status", "rows", "size", "updated", "details")
        self._artifact_tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)

        self._artifact_tree.heading("artifact", text="Artefato")
        self._artifact_tree.heading("status", text="Status")
        self._artifact_tree.heading("rows", text="Registros")
        self._artifact_tree.heading("size", text="Tamanho")
        self._artifact_tree.heading("updated", text="Atualizado")
        self._artifact_tree.heading("details", text="Detalhes")

        self._artifact_tree.column("artifact", width=190, anchor="w")
        self._artifact_tree.column("status", width=95, anchor="center")
        self._artifact_tree.column("rows", width=95, anchor="center")
        self._artifact_tree.column("size", width=95, anchor="center")
        self._artifact_tree.column("updated", width=155, anchor="center")
        self._artifact_tree.column("details", width=390, anchor="w")

        scroll = ttk.Scrollbar(parent, orient="vertical", command=self._artifact_tree.yview)
        self._artifact_tree.configure(yscrollcommand=scroll.set)
        self._artifact_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for key, label, _, _ in _ARTIFACT_SPECS:
            self._artifact_tree.insert(
                "",
                "end",
                iid=key,
                values=(label, "ausente", "-", "-", "-", "-"),
            )

    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self, text="Log", padding=(8, 4))
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._log = scrolledtext.ScrolledText(frame, height=14, wrap=tk.WORD)
        self._log.pack(fill=tk.BOTH, expand=True)
        self._log.configure(state="disabled")

    def _open_output_folder(self) -> None:
        folder = self._output_dir.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                import os

                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Erro", f"Nao consegui abrir a pasta: {exc}")

    def _start_run(self) -> None:
        try:
            stages = self._collect_stages()
        except ValueError as exc:
            messagebox.showerror("Configuracao invalida", str(exc))
            return
        self._start_pipeline(stages)

    def _start_postprocess(self) -> None:
        stages = self._collect_postprocess_stages()
        self._start_pipeline(stages)

    def _start_pipeline(self, stages: list[_Stage]) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Em andamento", "Ja existe uma execucao rodando.")
            return
        if not stages:
            messagebox.showwarning("Nada para executar", "Selecione ao menos uma etapa.")
            return
        if not self._validate_prerequisites(stages):
            return

        (
            self._target_by_kind,
            self._retry_errors_by_kind,
            self._phase_selection_by_kind,
        ) = self._infer_targets(stages)
        self._prepare_stage_rows(stages)

        self._stop_flag.clear()
        self._start_btn.configure(state="disabled")
        self._post_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress_var.set(0.0)
        self._progress_bar.configure(maximum=len(stages))
        self._progress_label.configure(text=f"iniciando... (0/{len(stages)})")
        self._log_line(f">> Iniciando pipeline com {len(stages)} etapa(s).")

        self._worker_thread = threading.Thread(
            target=self._worker_main,
            args=(stages,),
            daemon=True,
        )
        self._worker_thread.start()

    def _infer_targets(
        self,
        stages: list[_Stage],
    ) -> tuple[dict[str, int | None], dict[str, bool], dict[str, set[str]]]:
        target_by_kind: dict[str, int | None] = {}
        retry_by_kind: dict[str, bool] = {}
        phase_by_kind: dict[str, set[str]] = {}
        for stage in stages:
            parsed = self._parse_enrichment_stage(stage.argv)
            if parsed is None:
                continue
            plural_kind, phase, limit, retry_errors = parsed
            singular = _KIND_TO_SINGULAR[plural_kind]
            retry_by_kind[singular] = retry_errors
            phases = phase_by_kind.setdefault(singular, set())
            if phase == "all":
                phases.update({"fetch", "normalize"})
            else:
                phases.add(phase)
            # Nao carregamos seeds/duckdb dentro da UI para evitar conflitos de lifecycle no Tk.
            # Quando ha limit, usamos esse valor como meta; sem limit, fica indeterminado.
            target_by_kind[singular] = limit
        return target_by_kind, retry_by_kind, phase_by_kind

    def _refresh_seed_totals_async(self) -> None:
        if self._seed_total_thread and self._seed_total_thread.is_alive():
            return
        self._seed_total_thread = threading.Thread(target=self._seed_totals_worker, daemon=True)
        self._seed_total_thread.start()

    def _seed_totals_worker(self) -> None:
        totals: dict[str, int | None] = {}
        for plural in _KIND_ORDER:
            singular = _KIND_TO_SINGULAR[plural]
            argv = [sys.executable, "-m", "music_search.enrichment", plural, "--count-only"]
            try:
                proc = subprocess.run(
                    argv,
                    cwd=self._project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception as exc:
                self._event_queue.put(("log", f"(erro ao contar total de {plural}: {exc})"))
                totals[singular] = None
                continue

            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                msg = err[-1] if err else f"exit={proc.returncode}"
                self._event_queue.put(("log", f"(erro ao contar total de {plural}: {msg})"))
                totals[singular] = None
                continue

            output = (proc.stdout or "").strip().splitlines()
            raw = output[-1].strip() if output else ""
            try:
                totals[singular] = int(raw.replace(",", ""))
            except ValueError:
                self._event_queue.put(
                    ("log", f"(saida invalida ao contar total de {plural}: {raw!r})")
                )
                totals[singular] = None

        self._event_queue.put(("seed_totals", totals))

    def _parse_enrichment_stage(
        self,
        argv: list[str],
    ) -> tuple[str, str, int | None, bool] | None:
        if len(argv) < 4 or argv[1:3] != ["-m", "music_search.enrichment"]:
            return None
        plural_kind = argv[3]
        if plural_kind not in _KIND_ORDER:
            return None

        phase = "all"
        limit: int | None = None
        retry_errors = False
        i = 4
        while i < len(argv):
            token = argv[i]
            if token == "--phase" and i + 1 < len(argv):
                candidate = argv[i + 1]
                if candidate in {"all", "fetch", "normalize"}:
                    phase = candidate
                i += 2
                continue
            if token == "--limit" and i + 1 < len(argv):
                try:
                    limit = int(argv[i + 1])
                except ValueError:
                    limit = None
                i += 2
                continue
            if token == "--retry-errors":
                retry_errors = True
            i += 1
        return plural_kind, phase, limit, retry_errors

    def _validate_prerequisites(self, stages: list[_Stage]) -> bool:
        has_normalization = False
        for stage in stages:
            parsed = self._parse_enrichment_stage(stage.argv)
            if parsed is None:
                continue
            _, phase, _, _ = parsed
            if phase in {"all", "normalize"}:
                has_normalization = True
                break
        if not has_normalization:
            return True

        nim_base_url = os.environ.get("NIM_BASE_URL", "").strip()
        if nim_base_url:
            return True

        msg = (
            "NIM_BASE_URL nao esta definido no ambiente atual.\n\n"
            "As etapas de normalizacao dependem de uma API OpenAI-compativel.\n"
            "Configure NIM_BASE_URL (e opcionalmente NIM_API_KEY) antes de iniciar.\n\n"
            "Se voce quiser apenas baixar documentos brutos, desmarque a fase 'Normalizar com LLM'."
        )
        self._log_line("Prerequisito ausente: NIM_BASE_URL.")
        messagebox.showerror("Prerequisito ausente", msg)
        return False

    def _stop_run(self) -> None:
        self._stop_flag.set()
        self._stop_btn.configure(state="disabled")
        self._progress_label.configure(text="parando apos etapa atual...")
        self._log_line("[] Parada pedida.")

        with self._proc_lock:
            proc = self._current_proc
        if proc and proc.poll() is None:
            proc.terminate()

    def _collect_stages(self) -> list[_Stage]:
        stages: list[_Stage] = []
        concurrency = max(1, int(self._concurrency_var.get()))
        retry_errors = bool(self._retry_errors_var.get())
        run_fetch = bool(self._fetch_docs_var.get())
        run_normalize = bool(self._normalize_llm_var.get())
        selected: list[tuple[str, int | None]] = []

        for kind in _KIND_ORDER:
            if not self._enabled_vars[kind].get():
                continue
            limit = self._parse_limit(self._limit_vars[kind].get(), kind)
            selected.append((kind, limit))

        if run_fetch:
            for kind, limit in selected:
                stages.append(
                    self._build_enrichment_stage(
                        kind=kind,
                        limit=limit,
                        concurrency=concurrency,
                        retry_errors=retry_errors,
                        phase="fetch",
                    )
                )

        if run_normalize:
            for kind, limit in selected:
                stages.append(
                    self._build_enrichment_stage(
                        kind=kind,
                        limit=limit,
                        concurrency=concurrency,
                        retry_errors=retry_errors,
                        phase="normalize",
                    )
                )

        if self._export_var.get():
            stages.append(self._stage_export())
        if self._build_dataset_var.get():
            stages.append(self._stage_build_dataset())
        return stages

    def _build_enrichment_stage(
        self,
        *,
        kind: str,
        limit: int | None,
        concurrency: int,
        retry_errors: bool,
        phase: str,
    ) -> _Stage:
        argv = [
            sys.executable,
            "-m",
            "music_search.enrichment",
            kind,
            "--concurrency",
            str(concurrency),
            "--phase",
            phase,
        ]
        if limit is not None:
            argv.extend(["--limit", str(limit)])
        if retry_errors:
            argv.append("--retry-errors")

        label = f"coleta {kind}" if phase == "fetch" else f"normalizacao {kind}"
        return _Stage(label=label, argv=argv)

    def _collect_postprocess_stages(self) -> list[_Stage]:
        return [self._stage_export(), self._stage_build_dataset()]

    def _stage_export(self) -> _Stage:
        return _Stage(
            label="export entities",
            argv=[sys.executable, "-m", "music_search.scripts.export_entities"],
        )

    def _stage_build_dataset(self) -> _Stage:
        return _Stage(
            label="build dataset (--skip-lyrics)",
            argv=[
                sys.executable,
                "-m",
                "music_search.scripts.build_dataset",
                "--skip-lyrics",
            ],
        )

    @staticmethod
    def _parse_limit(raw: str, kind: str) -> int | None:
        text = raw.strip()
        if text == "":
            return None
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"Limit invalido para {kind}: {text!r}") from exc
        if value <= 0:
            raise ValueError(f"Limit deve ser > 0 para {kind}.")
        return value

    def _prepare_stage_rows(self, stages: list[_Stage]) -> None:
        self._stage_tree.delete(*self._stage_tree.get_children())
        self._stage_index_map.clear()
        for idx, stage in enumerate(stages, start=1):
            iid = f"stage-{idx}"
            self._stage_index_map[idx] = iid
            self._stage_tree.insert(
                "",
                "end",
                iid=iid,
                values=(f"{idx}. {stage.label}", "pending", "-", "-", "-"),
            )

    def _mark_stage_status(
        self,
        *,
        index: int,
        status: str,
        exit_code: str = "-",
        started: str | None = None,
        finished: str | None = None,
    ) -> None:
        iid = self._stage_index_map.get(index)
        if not iid:
            return
        current = list(self._stage_tree.item(iid, "values"))
        if len(current) != 5:
            return
        current[1] = status
        current[2] = exit_code
        if started is not None:
            current[3] = started
        if finished is not None:
            current[4] = finished
        self._stage_tree.item(iid, values=tuple(current))

    def _worker_main(self, stages: list[_Stage]) -> None:
        total = len(stages)
        ok = True
        for idx, stage in enumerate(stages, start=1):
            if self._stop_flag.is_set():
                ok = False
                break

            self._event_queue.put(
                (
                    "stage_start",
                    {"index": idx, "total": total, "stage": stage.label},
                )
            )
            rc = self._run_command(stage.argv)
            if self._stop_flag.is_set():
                ok = False
                break
            if rc != 0:
                ok = False
                self._event_queue.put(
                    ("failed", {"index": idx, "stage": stage.label, "returncode": rc})
                )
                break
            self._event_queue.put(
                (
                    "stage_done",
                    {"index": idx, "total": total, "stage": stage.label},
                )
            )

        self._event_queue.put(("done", {"ok": ok, "stopped": self._stop_flag.is_set()}))

    def _run_command(self, argv: list[str]) -> int:
        self._event_queue.put(("log", f"$ {self._fmt_argv(argv)}"))

        proc = subprocess.Popen(
            argv,
            cwd=self._project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with self._proc_lock:
            self._current_proc = proc

        try:
            if proc.stdout is not None:
                for raw in proc.stdout:
                    if self._stop_flag.is_set() and proc.poll() is None:
                        proc.terminate()
                    line = raw.rstrip()
                    if line:
                        self._event_queue.put(("log", line))
            return proc.wait()
        finally:
            with self._proc_lock:
                self._current_proc = None

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self._event_queue.get_nowait()
                if kind == "log":
                    self._log_line(str(payload))
                elif kind == "stage_start":
                    idx, total = payload["index"], payload["total"]
                    self._progress_var.set(max(0, idx - 1))
                    self._progress_label.configure(
                        text=f"rodando {payload['stage']} ({idx}/{total})..."
                    )
                    self._mark_stage_status(
                        index=idx,
                        status="running",
                        started=self._now_hms(),
                    )
                elif kind == "stage_done":
                    idx, total = payload["index"], payload["total"]
                    self._progress_var.set(idx)
                    self._progress_label.configure(
                        text=f"concluida {payload['stage']} ({idx}/{total})"
                    )
                    self._mark_stage_status(
                        index=idx,
                        status="done",
                        exit_code="0",
                        finished=self._now_hms(),
                    )
                    self._refresh_artifacts_once()
                elif kind == "failed":
                    idx = int(payload["index"])
                    rc = int(payload["returncode"])
                    self._mark_stage_status(
                        index=idx,
                        status="failed",
                        exit_code=str(rc),
                        finished=self._now_hms(),
                    )
                    self._log_line(f"X Etapa falhou: {payload['stage']} (exit={rc}).")
                    messagebox.showerror(
                        "Falha no pipeline",
                        f"Etapa '{payload['stage']}' falhou com exit {rc}.",
                    )
                elif kind == "done":
                    self._on_done(ok=bool(payload["ok"]), stopped=bool(payload["stopped"]))
                elif kind == "seed_totals":
                    self._total_by_kind = {
                        str(k): (v if isinstance(v, int) else None)
                        for k, v in dict(payload).items()
                    }
                    self._refresh_status()
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _refresh_status(self) -> None:
        stats_by_kind: dict[str, dict[str, int]] = {}
        try:
            cache = KeyValueCache(self._cache_path, table=DEFAULT_CACHE_TABLE)
            stats_by_kind = cache.stats_by_kind()
            cache.close()
            self._last_status_error = None
        except Exception as exc:
            msg = str(exc)
            if msg != self._last_status_error:
                self._log_line(f"(erro ao ler cache: {msg})")
                self._last_status_error = msg

        for plural in _KIND_ORDER:
            singular = _KIND_TO_SINGULAR[plural]
            by_status = stats_by_kind.get(singular, {})
            fetched = by_status.get("fetched", 0)
            hit = by_status.get("hit", 0)
            miss = by_status.get("miss", 0)
            error = by_status.get("error", 0)
            blocked = by_status.get("blocked", 0)
            total = self._total_by_kind.get(singular)
            target = self._target_by_kind.get(singular)
            retry_errors = self._retry_errors_by_kind.get(singular, False)
            phases = self._phase_selection_by_kind.get(singular, set())
            if target is None:
                effective_target = total
            elif total is None:
                effective_target = target
            else:
                effective_target = min(target, total)
            if effective_target is None:
                pending: int | None = None
            else:
                completed = hit + miss + blocked + (0 if retry_errors else error)
                if "normalize" not in phases and "fetch" in phases:
                    completed += fetched
                pending = max(effective_target - completed, 0)

            self._status_tree.item(
                plural,
                values=(
                    _KIND_LABEL[plural],
                    _UNKNOWN if total is None else f"{total:,}",
                    "sem limite" if target is None else f"{target:,}",
                    _UNKNOWN if pending is None else f"{pending:,}",
                    fetched,
                    hit,
                    miss,
                    error,
                    blocked,
                ),
            )

        self.after(_STATUS_REFRESH_INTERVAL_MS, self._refresh_status)

    def _on_status_row_double_click(self, event: tk.Event[tk.Misc]) -> None:
        iid = self._status_tree.identify_row(event.y)
        if not iid:
            selected = self._status_tree.selection()
            iid = selected[0] if selected else ""
        if iid not in _KIND_ORDER:
            return
        singular = _KIND_TO_SINGULAR[iid]
        self._open_cache_details(kind=singular, label=_KIND_LABEL[iid])

    def _open_cache_details(self, *, kind: str, label: str) -> None:
        try:
            cache = KeyValueCache(self._cache_path, table=DEFAULT_CACHE_TABLE)
            rows = cache.list_by_kind(kind, limit=_CACHE_DETAILS_LIMIT)
            cache.close()
        except Exception as exc:
            messagebox.showerror("Erro ao abrir cache", f"Nao consegui carregar detalhes: {exc}")
            return

        win = tk.Toplevel(self)
        win.title(f"Cache detalhes - {label} ({len(rows):,} registros)")
        win.geometry("1200x680")
        win.minsize(960, 540)

        header = ttk.Frame(win, padding=(10, 8, 10, 4))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text=(
                f"Entidade: {label} ({kind}) | Mostrando ate {_CACHE_DETAILS_LIMIT:,} registros "
                "mais recentes"
            ),
        ).pack(anchor="w")

        body = ttk.Panedwindow(win, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        table_wrap = ttk.Frame(body)
        body.add(table_wrap, weight=3)
        columns = ("query", "status", "source", "attempts", "fetched", "error")
        tree = ttk.Treeview(table_wrap, columns=columns, show="headings", height=18)
        tree.heading("query", text="Query")
        tree.heading("status", text="Status")
        tree.heading("source", text="Source")
        tree.heading("attempts", text="Attempts")
        tree.heading("fetched", text="Atualizado")
        tree.heading("error", text="Erro (resumo)")
        tree.column("query", width=250, anchor="w")
        tree.column("status", width=90, anchor="center")
        tree.column("source", width=130, anchor="center")
        tree.column("attempts", width=80, anchor="center")
        tree.column("fetched", width=160, anchor="center")
        tree.column("error", width=440, anchor="w")
        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(table_wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_wrap.rowconfigure(0, weight=1)
        table_wrap.columnconfigure(0, weight=1)

        details_wrap = ttk.Frame(body)
        body.add(details_wrap, weight=2)
        ttk.Label(details_wrap, text="Detalhes do registro selecionado").pack(anchor="w")
        details = scrolledtext.ScrolledText(details_wrap, wrap=tk.WORD, height=10)
        details.pack(fill=tk.BOTH, expand=True)
        details.configure(state="disabled")

        rows_by_iid: dict[str, dict[str, Any]] = {}
        for idx, row in enumerate(rows, start=1):
            key = str(row.get("key") or "")
            query = key.split(":", 1)[1] if ":" in key else key
            error = str(row.get("error") or "")
            if len(error) > 180:
                error = error[:177] + "..."
            fetched = self._format_cache_timestamp(row.get("fetched_at"))
            iid = f"row-{idx}"
            rows_by_iid[iid] = row
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    query,
                    row.get("status") or "-",
                    row.get("source") or "-",
                    row.get("attempts") or "-",
                    fetched,
                    error or "-",
                ),
            )

        def show_selected_details(_event: tk.Event[tk.Misc] | None = None) -> None:
            selected = tree.selection()
            if not selected:
                return
            row = rows_by_iid.get(selected[0])
            if row is None:
                return
            payload = row.get("payload")
            if payload is None:
                payload_text = "(sem payload JSON)"
            else:
                payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
            text = (
                f"key: {row.get('key')}\n"
                f"kind: {row.get('kind')}\n"
                f"status: {row.get('status')}\n"
                f"source: {row.get('source')}\n"
                f"source_url: {row.get('source_url')}\n"
                f"attempts: {row.get('attempts')}\n"
                f"fetched_at: {self._format_cache_timestamp(row.get('fetched_at'))}\n"
                f"error: {row.get('error') or '-'}\n\n"
                f"payload:\n{payload_text}"
            )
            details.configure(state="normal")
            details.delete("1.0", "end")
            details.insert("1.0", text)
            details.configure(state="disabled")

        tree.bind("<<TreeviewSelect>>", show_selected_details)
        tree.bind("<Double-1>", show_selected_details)
        if rows:
            first = tree.get_children()[0]
            tree.selection_set(first)
            tree.focus(first)
            show_selected_details()

    def _refresh_artifacts(self) -> None:
        self._refresh_artifacts_once()
        self.after(_ARTIFACT_REFRESH_INTERVAL_MS, self._refresh_artifacts)

    def _refresh_artifacts_once(self) -> None:
        try:
            for key, label, filename, kind in _ARTIFACT_SPECS:
                path = self._output_dir / filename
                if not path.exists():
                    self._artifact_tree.item(
                        key,
                        values=(label, "ausente", "-", "-", "-", path.as_posix()),
                    )
                    continue

                mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                size = self._format_size(path.stat().st_size)
                rows = "-"
                details = path.as_posix()
                status = "ok"

                if kind != "parquet":
                    details = self._manifest_details(path)

                self._artifact_tree.item(
                    key,
                    values=(label, status, rows, size, mtime, details),
                )
            self._last_artifact_error = None
        except Exception as exc:
            msg = str(exc)
            if msg != self._last_artifact_error:
                self._log_line(f"(erro ao atualizar artefatos: {msg})")
                self._last_artifact_error = msg

    def _manifest_details(self, path: Path) -> str:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return path.as_posix()
        version = str(data.get("version") or "?")
        generated = str(data.get("generated_at") or "?")
        counts = data.get("counts") or {}
        artists = counts.get("artists", 0)
        albums = counts.get("albums", 0)
        genres = counts.get("genres", 0)
        composers = counts.get("composers", 0)
        return (
            f"v{version} | generated_at={generated} | "
            f"a={artists} alb={albums} g={genres} c={composers}"
        )

    @staticmethod
    def _format_cache_timestamp(value: object) -> str:
        if value in (None, ""):
            return "-"
        if not isinstance(value, (int, float, str)):
            return str(value)
        try:
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError, OSError):
            return str(value)

    def _on_done(self, *, ok: bool, stopped: bool) -> None:
        self._start_btn.configure(state="normal")
        self._post_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._stop_flag.clear()

        if stopped:
            self._progress_label.configure(text="interrompido pelo usuario")
            self._mark_pending_stages_cancelled()
            self._log_line("Pipeline interrompido.")
            return

        if ok:
            self._progress_label.configure(text="finalizado")
            self._log_line("OK Pipeline concluido.")
        else:
            self._progress_label.configure(text="falhou")
            self._mark_pending_stages_cancelled()
        self._refresh_artifacts_once()

    def _mark_pending_stages_cancelled(self) -> None:
        for iid in self._stage_tree.get_children():
            values = list(self._stage_tree.item(iid, "values"))
            if len(values) != 5:
                continue
            if values[1] == "pending":
                values[1] = "cancelled"
                self._stage_tree.item(iid, values=tuple(values))

    def _log_line(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    @staticmethod
    def _fmt_argv(argv: list[str]) -> str:
        parts: list[str] = []
        for p in argv:
            if " " in p:
                parts.append(f'"{p}"')
            else:
                parts.append(p)
        return " ".join(parts)

    @staticmethod
    def _now_hms() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        units = ("B", "KB", "MB", "GB")
        value = float(size_bytes)
        unit = units[0]
        for unit in units:
            if value < 1024.0 or unit == units[-1]:
                break
            value /= 1024.0
        if unit == "B":
            return f"{int(value)} {unit}"
        return f"{value:.1f} {unit}"


def main() -> None:
    app = EnrichmentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
