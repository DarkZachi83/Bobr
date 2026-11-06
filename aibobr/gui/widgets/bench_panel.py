"""Zakładka „Benchmark” (etap 13c, makieta autora) – zastępuje okna „Modele na dysku…”
i „Benchmark”.

Lewa kolumna: Modele na dysku – szukanie po kliknięciu (zgoda raz), lista zapamiętana między
uruchomieniami, werdykt AI Bóbra dla każdego modelu, pole ☐ do zaznaczenia modeli z Ollamy
(benchmark umie tylko je – llama-server i LM Studio w 12d). Dwuklik / „Sprawdź” – model
w zakładce Sprawdź.
Prawa: konfiguracja (konteksty, plan, ostrzeżenia, Start / Zatrzymaj), pod nią przesuwany
podział: tabela wyników z kolumną Status i dziennik na żywo (kolory wpisów, postęp, czas do
końca, kopiowanie, zapis do pliku, eksport CSV). Modele mierzymy PO KOLEI – nigdy kilka naraz.
"""
from __future__ import annotations

import datetime
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from aibobr.bench.runner import estimate_seconds
from aibobr.gui.theme import COLORS, VERDICT_COLORS
from aibobr.i18n import fmt_context, fmt_gb, t
from aibobr.bench.quality import MODES, REPEATS
from aibobr.scan.model import FoundModel, Kind, ScanReport

CONTEXTS = (1024, 8192, 32768)
DEFAULT_ON = {1024, 8192}                  # 32k na życzenie – na słabszym sprzęcie trwa długo
COLUMNS = ("model", "context", "predicted", "pp", "ttft", "tg", "gpu", "status")
WIDTHS = {"model": 150, "context": 62, "predicted": 95, "pp": 88, "ttft": 70, "tg": 112, "gpu": 70,
          "status": 95}
SUITES = ("perf", "pl_jezyk_v1", "en_format_v1")    # etap 12c: wszystkie domyślnie zaznaczone
QCOLS = ("model", "spelling", "inflection", "numerals", "aspect", "knowledge", "json_strict", "json",
         "code_strict", "code", "agree")
QSECONDS = 1.5          # zadanie jakości – pomiar autora: gemma2:9b w karcie, 15 zadań w ok. 15 s
QRELOAD = 4.0           # tryb kontrolowany: zwolnienie + wczytanie modelu przed każdą próbą (model w karcie)
ORDER = {Kind.TEXT: 0, Kind.IMAGE: 1, Kind.ADDON: 2, Kind.OTHER: 3}
LEFT_WIDTH = 400
ON, OFF = "☑", "☐"
LOG_COLORS = {"START": "#66c9fa", "INFO": "#9bacc0", "RUN": "#f5c65f", "RESULT": "#90e0ef",
              "OK": "#53d6a0", "WARN": "#f1a65b", "ERROR": "#ff8282", "END": "#53d6a0"}


def _num(x: Optional[float], digits: int = 1) -> str:
    return "–" if x is None else f"{x:.{digits}f}".replace(".", t("bench.decimal"))


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def benchable(m: FoundModel) -> bool:
    """Benchmark umie uruchomić tylko modele tekstowe z Ollamy."""
    return m.kind == Kind.TEXT and (m.target or "").startswith("ollama:")


class BenchmarkPanel(ttk.Frame):
    def __init__(self, master, controller, worker, current: Callable, on_check: Callable[[str], None],
                 on_status: Callable[[str], None] = lambda text: None):
        super().__init__(master, padding=(0, 8, 0, 0))
        self.c, self.worker, self.current, self.on_check, self.on_status = \
            controller, worker, current, on_check, on_status
        self.confirm = lambda title, text: messagebox.askokcancel(title, text, parent=self)
        self.report = ScanReport()
        self.rows: dict[str, FoundModel] = {}
        self.checked: set[str] = set()             # cele (target) zaznaczone do testu
        self.verdicts: dict[str, str] = {}         # target → green / yellow / red
        self.verdicts_done = False
        self.predicted: dict = {}                  # (target, kontekst) → poziom
        self.scan_stop = threading.Event()
        self.stop = threading.Event()
        self.seen = 0
        self.scanning = False
        self.running = False
        self.results: list = []
        self.log_lines: list[str] = []
        self.quality: list = []
        self.qresults: dict[str, list] = {}           # model → odpowiedzi (podgląd po teście)
        self.started = 0.0
        self.estimated = 0.0
        self.steps_total = 0
        self.steps_done = 0
        self.job_rows: dict = {}                   # (model, kontekst) → wiersz tabeli
        self.columnconfigure(0, weight=0, minsize=LEFT_WIDTH)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.left = ttk.Frame(self, style="Panel.TFrame", padding=10)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.right = ttk.Frame(self)
        self.right.grid(row=0, column=1, sticky="nsew")
        self._build_models()
        self._build_bench()
        report, when = self.c.load_found()
        if report is not None and report.models:
            self.report = report
            self._fill()
            self.scan_status.configure(text=t("bench.scanned_at", date=when or "?",
                                              count=len(report.models)))
        elif not self.c.scan_consent:
            self._show_consent()
        self._update_plan()

    # ================================================================== modele na dysku
    def _build_models(self) -> None:
        f = self.left
        ttk.Label(f, text=t("bench.models_title"), style="Title.TLabel").pack(anchor="w")
        ttk.Label(f, text=t("bench.models_hint"), style="Muted.TLabel", wraplength=LEFT_WIDTH - 30,
                  justify="left").pack(anchor="w", pady=(2, 6))
        self.consent = ttk.Frame(f, style="Inner.TFrame")
        buttons = ttk.Frame(f, style="Inner.TFrame")
        buttons.pack(fill="x", pady=(0, 4))
        self.scan_button = ttk.Button(buttons, text=t("bench.scan"), command=self.search)
        self.scan_button.pack(side="left")
        self.folder_button = ttk.Button(buttons, text=t("gui.scan_folder"), command=self.pick_folder)
        self.folder_button.pack(side="left", padx=6)
        self.scan_stop_button = ttk.Button(buttons, text=t("gui.scan_stop"), command=self.scan_stop.set)
        self.scan_status = ttk.Label(f, text="", style="Muted.TLabel", wraplength=LEFT_WIDTH - 30,
                                     justify="left")
        self.scan_status.pack(anchor="w", pady=(0, 4))
        box = ttk.Frame(f, style="Inner.TFrame")
        box.pack(fill="both", expand=True)
        cols = ("sel", "name", "app", "size", "verdict")
        self.tree = ttk.Treeview(box, columns=cols, show="headings", selectmode="browse")
        for col, width, anchor, stretch in (("sel", 30, "center", False), ("name", 160, "w", True),
                                            ("app", 72, "w", False), ("size", 72, "e", False),
                                            ("verdict", 78, "w", False)):
            self.tree.heading(col, text="" if col == "sel" else
                              t("gui.scan_col.name") if col == "name" else t(f"bench.found.{col}"))
            self.tree.column(col, width=width, minwidth=30, anchor=anchor, stretch=stretch)
        for level, color in VERDICT_COLORS.items():
            self.tree.tag_configure(level.value, foreground=color)
        self.tree.tag_configure("off", foreground=COLORS["muted"])
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<ButtonRelease-1>", self._click)
        self.tree.bind("<space>", lambda e: self._toggle(self.tree.focus()))
        self.tree.bind("<Double-Button-1>", lambda e: self._check())
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._selected())
        self.path = ttk.Label(f, text="", style="Muted.TLabel", wraplength=LEFT_WIDTH - 30, justify="left")
        self.path.pack(anchor="w", pady=(4, 0))
        actions = ttk.Frame(f, style="Inner.TFrame")
        actions.pack(fill="x", pady=(6, 0))
        self.check_button = ttk.Button(actions, text=t("bench.check"), command=self._check)
        self.check_button.pack(side="left")
        self.check_button.state(["disabled"])
        ttk.Button(actions, text=t("bench.check_none"), command=self._uncheck_all).pack(side="right")

    def _show_consent(self) -> None:
        for child in self.consent.winfo_children():
            child.destroy()
        ttk.Label(self.consent, text=t("gui.scan_consent"), wraplength=LEFT_WIDTH - 30, justify="left",
                  style="Panel.TLabel").pack(anchor="w", pady=(0, 6))
        self.consent.pack(fill="x", before=self.scan_status, pady=(0, 6))

    def search(self, folder: Optional[Path] = None) -> None:
        if self.scanning:
            return
        if not self.c.scan_consent:              # kliknięcie „Szukaj modeli” = zgoda (tekst nad przyciskiem)
            self.c.set_scan_consent(True)
            self.consent.pack_forget()
        self.scanning, self.seen, self.folder = True, 0, folder
        self.scan_stop.clear()
        self.scan_button.state(["disabled"])
        self.folder_button.state(["disabled"])
        self.scan_stop_button.pack(side="left")

        def progress(n: int) -> None:            # wątek w tle – tylko przypisanie liczby
            self.seen = n
        self.worker.run("find_models", lambda: self.c.find_models(folder, self.scan_stop, progress),
                        self._found, self._scan_failed)
        self._tick()

    def _tick(self) -> None:
        if self.scanning and self.winfo_exists():
            self.scan_status.configure(text=t("gui.scan_searching", count=self.seen))
            self.after(200, self._tick)

    def _scan_finished(self) -> None:
        self.scanning = False
        if self.winfo_exists():
            self.scan_button.state(["!disabled"])
            self.folder_button.state(["!disabled"])
            self.scan_stop_button.pack_forget()

    def _found(self, report: ScanReport) -> None:
        self._scan_finished()
        if not self.winfo_exists():
            return
        if self.folder:
            self.report.merge(report)
        else:
            self.report = report
        self._fill()
        if not report.models:
            text = t("gui.scan_none_folder") if self.folder else t("gui.scan_none")
        else:
            text = t("bench.found", count=len(self.report.models), size=fmt_gb(self.report.total_size))
        if report.cancelled:
            text += " " + t("gui.scan_cancelled")
        if report.denied:
            text += "\n" + t("gui.scan_denied", folders=", ".join(str(d) for d in report.denied[:3]))
        self.scan_status.configure(text=text)

    def _scan_failed(self, error: BaseException) -> None:
        self._scan_finished()
        if self.winfo_exists():
            self.scan_status.configure(text=t("gui.unexpected", error=repr(error)))

    def pick_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title=t("gui.scan_folder_title"), mustexist=True)
        if folder:
            self.search(Path(folder))

    def _fill(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        models = sorted(self.report.with_copies(), key=lambda m: (ORDER[m.kind], m.name.lower()))
        for i, m in enumerate(models):
            iid = str(i)
            self.rows[iid] = m
            self.tree.insert("", "end", iid=iid)
            self._row(iid)
        self.checked &= {m.target for m in models if benchable(m)}
        self.refresh_verdicts()

    def _row(self, iid: str) -> None:
        m = self.rows[iid]
        level = self.verdicts.get(m.target or "")
        mark = (ON if m.target in self.checked else OFF) if benchable(m) else ""
        pending = m.checkable and m.target not in self.verdicts and not self.verdicts_done
        verdict = t(f"bench.found.level.{level}") if level else ("…" if pending else "–")
        name = m.name + (f" · {m.quant}" if m.quant else "")
        self.tree.item(iid, values=(mark, name, m.app, fmt_gb(m.size) if m.size else "", verdict),
                       tags=(level,) if level else (() if m.checkable else ("off",)))

    def refresh_verdicts(self) -> None:
        """Werdykt każdego modelu dla tego komputera – w tle (wczytuje nagłówki plików)."""
        try:
            hw, settings = self.current()
        except ValueError:
            return
        targets = [m.target for m in self.rows.values() if m.checkable and m.kind in (Kind.TEXT, Kind.IMAGE)]

        def work():
            from aibobr.calculator.verdict import evaluate
            out = {}
            for target in targets:
                try:
                    spec = self.c.load_model(target).spec()
                    out[target] = evaluate(spec, hw, settings, self.c.quants).level.value
                except Exception:
                    continue
            return out

        def done(out):
            if not self.winfo_exists():
                return
            self.verdicts.update(out)
            self.verdicts_done = True
            for iid in self.rows:
                self._row(iid)
        if targets:
            self.verdicts_done = False
            self.worker.run("disk_verdicts", work, done, lambda e: None)

    def _click(self, event) -> None:
        iid = self.tree.identify_row(event.y)
        if iid and self.tree.identify_column(event.x) == "#1":      # kolumna ☐ (pierwsza)
            self._toggle(iid)

    def _toggle(self, iid: str) -> None:
        m = self.rows.get(iid)
        if not m:
            return
        if not benchable(m):
            self.path.configure(text=t("bench.only_ollama"))
            return
        if self.running:
            return
        self.checked ^= {m.target}
        self._row(iid)
        self._targets_changed()

    def _uncheck_all(self) -> None:
        if self.running:
            return
        self.checked.clear()
        for iid in self.rows:
            self._row(iid)
        self._targets_changed()

    def select_targets(self, targets: list[str]) -> None:
        """Z zakładki Sprawdź (⏱ Benchmark): zaznacza ten model, dopisuje go do listy, gdy go nie ma."""
        if self.running:
            return
        known = {m.target for m in self.rows.values()}
        for target in targets:
            if target not in known:
                name = target.split(":", 1)[1]
                self.report.models.append(FoundModel(name, Kind.TEXT, "GGUF", 0, Path(name), "Ollama",
                                                     target=target))
        self.checked = set(targets)
        self._fill()
        self._targets_changed()

    def _selected(self) -> Optional[FoundModel]:
        sel = self.tree.selection()
        m = self.rows.get(sel[0]) if sel else None
        self.check_button.state(["!disabled"] if m and m.checkable else ["disabled"])
        if m:
            text = m.place if m.checkable else f"{m.place}\n{t('gui.scan_not_checkable')}"
            self.path.configure(text=text)
        return m

    def _check(self) -> None:
        m = self._selected()
        if m and m.checkable:
            self.on_check(m.target)

    def targets(self) -> list[str]:
        order = [m.target for m in sorted(self.rows.values(), key=lambda m: m.name.lower())]
        return [tg for tg in order if tg in self.checked]

    # ================================================================== benchmark
    def _build_bench(self) -> None:
        top = ttk.Frame(self.right, style="Panel.TFrame", padding=10)
        top.pack(fill="x")
        ttk.Label(top, text=t("bench.title"), style="Title.TLabel").pack(anchor="w")
        what = ttk.Label(top, text=t("bench.what"), style="Muted.TLabel", wraplength=640, justify="left")
        what.pack(anchor="w", fill="x", pady=(2, 6))
        what.bind("<Configure>", lambda e: what.configure(wraplength=max(300, e.width)))
        opts = ttk.Frame(top, style="Inner.TFrame")
        opts.pack(fill="x")
        ttk.Label(opts, text=t("bench.contexts"), style="Panel.TLabel").pack(side="left")
        self.ctx_vars = {}
        for ctx in CONTEXTS:
            var = tk.BooleanVar(value=ctx in DEFAULT_ON)
            ttk.Checkbutton(opts, text=fmt_context(ctx), variable=var, style="Panel.TCheckbutton",
                            command=self._update_plan).pack(side="left", padx=6)
            self.ctx_vars[ctx] = var
        suites = ttk.Frame(top, style="Inner.TFrame")
        suites.pack(fill="x", pady=(4, 0))
        ttk.Label(suites, text=t("bench.suites_label"), style="Panel.TLabel").pack(side="left")
        self.suite_vars = {}
        for sid in SUITES:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(suites, text=t(f"bench.suite.{sid}"), variable=var, style="Panel.TCheckbutton",
                            command=self._update_plan).pack(side="left", padx=6)
            self.suite_vars[sid] = var
        ttk.Label(suites, text="   " + t("bench.repeats"), style="Panel.TLabel").pack(side="left")
        self.repeats = tk.StringVar(value="1")
        rc = ttk.Combobox(suites, textvariable=self.repeats, values=[str(r) for r in REPEATS], width=3,
                          state="readonly")
        rc.pack(side="left", padx=4)
        rc.bind("<<ComboboxSelected>>", lambda e: self._update_plan())
        self.repeats_box = rc
        # Tryb jakości (pamięć promptu): standardowy – model zwolniony raz przed pakietem, 1 próba;
        # kontrolowany – zwolniony przed każdą próbą, 1/3/5 prób, próby z pamięcią promptu odrzucone.
        modes = ttk.Frame(top, style="Inner.TFrame")
        modes.pack(fill="x", pady=(4, 0))
        ttk.Label(modes, text=t("bench.mode"), style="Panel.TLabel").pack(side="left")
        self.mode = tk.StringVar(value="standard")
        for m in MODES:
            ttk.Radiobutton(modes, text=t(f"bench.mode.{m}"), value=m, variable=self.mode,
                            style="Panel.TRadiobutton", command=self._mode_changed).pack(side="left", padx=6)
        self.mode_note = ttk.Label(top, text="", style="Muted.TLabel", wraplength=640, justify="left")
        self.mode_note.pack(anchor="w", fill="x")
        self._mode_changed(update=False)
        self.plan = ttk.Label(top, text="", style="Panel.TLabel", wraplength=640, justify="left")
        self.plan.pack(anchor="w", pady=(8, 0), fill="x")
        self.warning = ttk.Label(top, text="", style="Note.TLabel", wraplength=640, justify="left")
        self.warning.pack(anchor="w", fill="x")
        for label in (self.plan, self.warning):
            label.bind("<Configure>", lambda e, w=label: w.configure(wraplength=max(300, e.width)))
        buttons = ttk.Frame(top, style="Inner.TFrame")
        buttons.pack(fill="x", pady=(8, 0))
        self.start_button = ttk.Button(buttons, text="▶ " + t("bench.start"), style="Accent.TButton",
                                       command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="■ " + t("bench.stop"), command=self.stop.set)
        self.stop_button.pack(side="left", padx=6)
        self.stop_button.state(["disabled"])
        self.summary = ttk.Label(buttons, text="", style="Panel.TLabel", font=("TkDefaultFont", 10, "bold"))
        self.summary.pack(side="left", padx=10)

        split = ttk.Panedwindow(self.right, orient="vertical")
        split.pack(fill="both", expand=True, pady=(8, 0))
        self.results_tabs = ttk.Notebook(split)
        split.add(self.results_tabs, weight=2)
        res = ttk.Frame(self.results_tabs, style="Panel.TFrame", padding=8)
        self.results_tabs.add(res, text=t("bench.res.perf"))
        qres = ttk.Frame(self.results_tabs, style="Panel.TFrame", padding=8)
        self.results_tabs.add(qres, text=t("bench.res.quality"))
        self.qtable = ttk.Treeview(qres, columns=QCOLS, show="headings", height=6)
        for col in QCOLS:
            self.qtable.heading(col, text=t(f"bench.qcol.{col}"))
            self.qtable.column(col, width=150 if col == "model" else 78, minwidth=50, anchor="w",
                               stretch=col == "model")
        self.qtable.pack(fill="both", expand=True)
        self.qtable.bind("<Double-Button-1>", lambda e: self.show_answers())
        qhint = ttk.Label(qres, text=t("bench.qhint"), style="Muted.TLabel", wraplength=500, justify="left")
        qhint.pack(anchor="w", fill="x", pady=(4, 0))
        qhint.bind("<Configure>", lambda e: qhint.configure(wraplength=max(200, e.width)))
        self.qrows: dict[str, str] = {}                       # model → wiersz tabeli Jakość
        self.table = ttk.Treeview(res, columns=COLUMNS, show="headings", height=6)
        for col in COLUMNS:
            self.table.heading(col, text=t(f"bench.col.{col}"))
            self.table.column(col, width=WIDTHS[col], minwidth=50, anchor="w", stretch=col == "model")
        for level, color in VERDICT_COLORS.items():
            self.table.tag_configure(level.value, foreground=color)
        self.table.tag_configure("skip", foreground=COLORS["muted"])
        scroll = ttk.Scrollbar(res, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        log = ttk.Frame(split, style="Panel.TFrame", padding=8)
        split.add(log, weight=3)
        self.preview_window = None             # okno „Podgląd” – tworzone przy pierwszym Starcie
        head = ttk.Frame(log, style="Inner.TFrame")
        head.pack(fill="x")
        ttk.Label(head, text=t("bench.log.title"), style="Title.TLabel").pack(side="left")
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(head, text=t("bench.log.autoscroll"), variable=self.autoscroll,
                        style="Panel.TCheckbutton").pack(side="right")
        tools = ttk.Frame(log, style="Inner.TFrame")             # osobny rząd – dziennik bywa wąski
        tools.pack(fill="x", pady=(4, 0))
        ttk.Button(tools, text=t("bench.log.copy"), command=self.copy_log).pack(side="left")
        ttk.Button(tools, text=t("bench.log.save"), command=self.save_log).pack(side="left", padx=6)
        ttk.Button(tools, text=t("bench.log.csv"), command=self.export_csv).pack(side="left")
        self.restore_button = ttk.Button(tools, text=t("bench.preview.restore"), command=self.show_preview)
        self.restore_button.pack(side="right")
        ttk.Label(tools, text=t("bench.preview.title") + ":", style="Panel.TLabel").pack(side="right", padx=6)
        self.restore_button.state(["disabled"])
        line = ttk.Frame(log, style="Inner.TFrame")
        line.pack(fill="x", pady=(6, 2))
        self.percent = ttk.Label(line, text="", style="Panel.TLabel", font=("TkDefaultFont", 10, "bold"))
        self.percent.pack(side="right")
        self.stage = ttk.Label(line, text="", style="Note.TLabel", wraplength=300, justify="left")
        self.stage.pack(side="left", fill="x", expand=True)
        self.stage.bind("<Configure>", lambda e: self.stage.configure(wraplength=max(150, e.width)))
        self.bar = ttk.Progressbar(log, mode="determinate", maximum=100)
        self.bar.pack(fill="x")
        self.times = ttk.Label(log, text="", style="Muted.TLabel")
        self.times.pack(anchor="w", pady=(2, 4))
        box = ttk.Frame(log, style="Inner.TFrame")
        box.pack(fill="both", expand=True)
        self.console = tk.Text(box, wrap="word", font="TkFixedFont", height=10, background="#11171d",
                               foreground="#cfd8e3", insertbackground="#ffffff", relief="flat",
                               borderwidth=0, padx=10, pady=8, state="disabled",
                               selectbackground="#354e66", selectforeground="#ffffff")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        self.console.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for tag, color in LOG_COLORS.items():
            self.console.tag_configure(tag, foreground=color)
        self.console.tag_configure("time", foreground="#7c90a4")
        self.bind("<Configure>", lambda e: None)

    def contexts(self) -> list[int]:
        return [c for c in CONTEXTS if self.ctx_vars[c].get()]

    def _targets_changed(self) -> None:
        """Nowe zaznaczenie – przewidywanie AI Bóbra dla nowych modeli (w tle), potem plan."""
        missing = [tg for tg in self.checked if not any((tg, c) in self.predicted for c in CONTEXTS)]
        self._update_plan()
        if not missing:
            return
        try:
            hw, settings = self.current()
        except ValueError:
            return
        self.stage.configure(text=t("bench.predicting"))

        def done(pred):
            self.predicted.update(pred)
            if not self.winfo_exists():
                return
            self.stage.configure(text="")
            self._update_plan()
        self.worker.run("bench_predict", lambda: self.c.bench_predict(missing, CONTEXTS, hw, settings),
                        done, lambda e: done({}))

    def _measured(self, model: str, ctx: int):
        db = getattr(self.c, "bench_db", None)
        if db is None:
            return None
        try:
            hw, _ = self.current()
            rows = db.latest(model, self.c.hardware_text(hw))
        except Exception:
            return None
        row = next((r for r in rows if r["context"] == ctx), None) or (rows[-1] if rows else None)
        return (row["pp_tps"], row["tg_tps"]) if row else None

    @property
    def controlled(self) -> bool:
        return self.mode.get() == "controlled"

    @property
    def repeat_count(self) -> int:
        return int(self.repeats.get()) if self.controlled else 1

    def _mode_changed(self, update: bool = True) -> None:
        if self.controlled:
            self.repeats_box.configure(state="readonly")
        else:
            self.repeats.set("1")                    # bez zwalniania kolejne próby trafiłyby w pamięć promptu
            self.repeats_box.configure(state="disabled")
        self.mode_note.configure(text=t(f"bench.mode_note.{self.mode.get()}"))
        if update:
            self._update_plan()

    def _update_plan(self) -> None:
        if self.running:
            return
        targets = self.targets()
        self.table.delete(*self.table.get_children())
        self.job_rows.clear()
        perf = self.suite_vars["perf"].get()
        quality = [sid for sid in SUITES[1:] if self.suite_vars[sid].get()]
        jobs = [(tg, c) for tg in targets for c in (self.contexts() if perf else ())]
        seconds, skipped = 0.0, 0
        self.qrows.clear()
        self.qtable.delete(*self.qtable.get_children())
        qtasks = 0
        if quality:
            ntasks = sum(len(self.c.quality_suite(sid).get("tests", ())) for sid in quality)
            for tg in targets:
                name = tg[len("ollama:"):] if tg.startswith("ollama:") else tg
                red = self.predicted.get((tg, min(CONTEXTS))) == "red"
                self.qrows[name] = self.qtable.insert("", "end", values=(name,) + (
                    (t("bench.status.skip"),) if red else (t("bench.status.wait"),)) + ("",) * (len(QCOLS) - 2))
                if not red:
                    qtasks += ntasks
                    per_try = QSECONDS + (QRELOAD if self.controlled else 0)
                    seconds += 12 + QRELOAD + ntasks * per_try * self.repeat_count
        for target, ctx in jobs:
            level = self.predicted.get((target, ctx))
            name = target[len("ollama:"):] if target.startswith("ollama:") else target
            if level == "red":
                skipped += 1
                values = (name, fmt_context(ctx), t("bench.pred.red"), "", "", "", "", t("bench.status.skip"))
                self.job_rows[(name, ctx)] = self.table.insert("", "end", values=values, tags=("skip",))
                continue
            seconds += estimate_seconds(ctx, level, measured=self._measured(name, ctx))
            values = (name, fmt_context(ctx), t(f"bench.pred.{level}") if level else "?", "", "", "", "",
                      t("bench.status.wait"))
            self.job_rows[(name, ctx)] = self.table.insert("", "end", values=values,
                                                           tags=(level,) if level else ())
        self.estimated = seconds
        minutes = max(1, round(seconds / 60))
        if not targets:
            self.plan.configure(text=t("bench.none_checked"))
        else:
            text = t("bench.plan", jobs=len(jobs) - skipped, minutes=minutes)
            if quality:
                text = t("bench.plan_q", jobs=len(jobs) - skipped, tasks=qtasks, minutes=minutes)
            if skipped:
                text += " " + t("bench.plan_skipped", n=skipped)
            self.plan.configure(text=text)
        warnings = []
        if len(targets) > 1:
            warnings.append(t("bench.warn_many", minutes=minutes))
        from aibobr.app.controller import power_source
        if power_source() == "battery":
            warnings.append(t("bench.warn_battery"))
        self.warning.configure(text="\n".join("⚠ " + w for w in warnings))
        self.steps_total = (len(jobs) - skipped) * 7 + qtasks
        self.start_button.state(["!disabled"] if len(jobs) > skipped or qtasks else ["disabled"])

    # ------------------------------------------------------------------ dziennik
    def log(self, kind: str, text: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{stamp}] {kind:<7} {text}")
        self.console.configure(state="normal")
        self.console.insert("end", f"[{stamp}] ", ("time",))
        self.console.insert("end", f"{kind:<7} ", (kind,))
        self.console.insert("end", text + "\n")
        self.console.configure(state="disabled")
        if self.autoscroll.get():
            self.console.see("end")

    def copy_log(self) -> None:
        self.clipboard_clear()
        self.clipboard_append("\n".join(self.log_lines) + "\n")
        self.on_status(t("bench.log.copied", n=len(self.log_lines)))

    def save_log(self) -> None:
        name = f"bobr_benchmark_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.log"
        path = filedialog.asksaveasfilename(parent=self, title=t("bench.log.save"), initialfile=name,
                                            defaultextension=".log",
                                            filetypes=[("Log", "*.log"), ("Text", "*.txt")])
        if path:
            Path(path).write_text("\n".join(self.log_lines) + "\n", encoding="utf-8")
            self.on_status(t("bench.log.saved", path=path))

    def export_csv(self) -> None:
        """Wydajność i jakość z bieżącego przebiegu – do trzech plików (aibobr.bench.export)."""
        from aibobr.bench import export
        perf = [r for r in self.results if r.tg_tps]
        quality = self.quality or [r for rs in self.qresults.values() for r in rs]
        if not perf and not quality:
            messagebox.showinfo(t("bench.log.csv"), t("bench.csv_none"), parent=self)
            return
        name = f"bobr_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        path = filedialog.asksaveasfilename(parent=self, title=t("bench.log.csv"), initialfile=name,
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        written = export.export_all(path, perf, quality)
        self.on_status(t("bench.log.exported", files=", ".join(p.name for p in written)))
        self.log("INFO", t("bench.log.exported", files=", ".join(str(p) for p in written)))

    # ------------------------------------------------------------------ pomiar
    def start(self) -> None:
        contexts, targets = self.contexts(), self.targets()
        suites = tuple(sid for sid in SUITES if self.suite_vars[sid].get())
        repeats, mode = self.repeat_count, self.mode.get()   # odczyt tu – zmienne Tk nie z wątku w tle
        if not targets or self.running or not suites or (suites == ("perf",) and not contexts):
            return
        if len(targets) > 1 and not self.confirm(t("bench.title"), t("bench.confirm_many")):
            return
        try:
            hw, _ = self.current()
        except ValueError:
            messagebox.showinfo(t("bench.title"), t("stats.need_hw"), parent=self)
            return
        self.hw = hw
        self.running = True
        self.results = []
        self.quality = []
        self.qresults.clear()                  # eksport i podgląd – tylko ten przebieg
        self.steps_done = 0
        self.started = time.monotonic()
        self.stop.clear()
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.summary.configure(text="")
        self.log_lines.clear()
        self.show_preview()                    # podgląd jako osobne okno (uwaga autora)
        self.preview_window.clear()
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        names = ", ".join(tg[len("ollama:"):] for tg in targets)
        if "perf" in suites:
            self.log("START", t("bench.log.models", names=names,
                                contexts=", ".join(fmt_context(c) for c in contexts)))
        else:                                   # sama jakość – konteksty z pola nie są mierzone
            self.log("START", t("bench.log.models_q", names=names))
        if any(sid != "perf" for sid in suites):
            self.log("INFO", t(f"bench.log.mode.{mode}", runs=repeats))
        self._clock()
        self.worker.run_with_progress(
            "bench",
            lambda report: self.c.run_benchmark(targets, contexts, self.predicted, hw, report, self.stop,
                                                db=self.c.bench_db, suites=suites, repeats=repeats,
                                                mode=mode),
            self._progress, self._done, self._failed)

    def _clock(self) -> None:
        if not self.running or not self.winfo_exists():
            return
        elapsed = time.monotonic() - self.started
        share = self.steps_done / self.steps_total if self.steps_total else 0
        left = elapsed * (1 - share) / share if share > 0.05 else max(0.0, self.estimated - elapsed)
        self.times.configure(text=t("bench.time_line", elapsed=_clock(elapsed), left=_clock(left)))
        self.after(500, self._clock)

    def _status(self, model: str, ctx: int, text: str) -> None:
        iid = self.job_rows.get((model, ctx))
        if iid:
            values = list(self.table.item(iid, "values"))
            values[-1] = text
            self.table.item(iid, values=values)
            self.table.see(iid)

    def _step_text(self, step: str) -> str:
        if step == "warmup":
            return t("bench.step.warmup")
        kind, n = step.split()
        return t(f"bench.step.{kind}", n=n)

    def _progress(self, p: dict) -> None:
        event = p.get("event", "step")
        ctx = fmt_context(p["context"]) if "context" in p else ""
        if event == "start":
            perf = p.get("perf", p["jobs"])
            parts = [t("bench.log.start_perf", jobs=perf)] if perf else []
            if p.get("q_suites"):
                parts.append(t("bench.log.start_quality", suites=p["q_suites"], tasks=p["q_tasks"],
                               runs=p.get("q_runs", 1)))
            self.log("INFO", t("bench.log.start", version=p.get("version") or "?", what=" · ".join(parts)))
            if p.get("busy_mb") is not None:
                self.log("INFO", t("bench.log.busy", mb=p["busy_mb"]))
        elif event == "step":
            step = self._step_text(p["step"])
            self.steps_done += 1
            share = min(1.0, self.steps_done / self.steps_total) if self.steps_total else 0
            self.bar.configure(value=share * 100)
            self.percent.configure(text=f"{share * 100:.0f}%")
            self.stage.configure(text=t("bench.progress", n=p["n"], total=p["total"], model=p["model"],
                                        context=ctx, step=step))
            self._status(p["model"], p["context"], step)
            self.log("RUN", f"{p['model']} | {ctx} | {step}")
        elif event == "measure":
            step = self._step_text(p["step"])
            if p["step"].startswith("prompt"):
                text = t("bench.log.pp", pp=_num(p["pp"], 0), ttft=_num(p["ttft"], 0))
            else:
                text = t("bench.log.tg", tg=_num(p["tg"]))
            self.log("RESULT", f"{p['model']} | {ctx} | {step}: {text}")
        elif event == "result":
            r = p["result"]
            gpu = "–" if r.gpu_share is None else f"{r.gpu_share * 100:.0f}%"
            iid = self.job_rows.get((r.model, r.context))
            if iid:
                self.table.item(iid, values=(r.model, fmt_context(r.context),
                                             t(f"bench.pred.{r.predicted}") if r.predicted else "?",
                                             _num(r.pp_tps, 0), _num(r.ttft_ms, 0), _num(r.tg_tps), gpu,
                                             t("bench.status.done")))
            text = t("bench.log.result", pp=_num(r.pp_tps, 0), ttft=_num(r.ttft_ms, 0),
                     tg=_num(r.tg_tps), gpu=gpu)
            if r.power_draw_w:
                text += " · " + t("bench.log.power", w=f"{r.power_draw_w:.0f}")
            self.log("OK", f"{r.model} | {fmt_context(r.context)} | {text}")
        elif event == "skip":
            self._status(p["model"], p["context"], t("bench.status.skip"))
            self.log("WARN", t("bench.log.skip", model=p["model"], context=ctx))
        elif event == "stopped":
            self._status(p["model"], p["context"], t("bench.status.stopped"))
            self.log("WARN", t("bench.log.stopped"))
        elif event == "p_task":
            self._preview_start(f"{p['model']} | {ctx} | {self._step_text(p['step'])}", p["prompt"])
        elif event == "chunk":
            self._preview_add(p["text"])
        elif event == "q_task":
            self.steps_done += 1
            share = min(1.0, self.steps_done / self.steps_total) if self.steps_total else 0
            self.bar.configure(value=share * 100)
            self.percent.configure(text=f"{share * 100:.0f}%")
            suite = t(f"bench.suite.{p['suite']}")
            label = f"{p['model']} | {suite} | {p['n']}/{p['total']} {p['title']}"
            self.stage.configure(text=label)
            self._qstatus(p["model"], f"{suite} {p['n']}/{p['total']}")
            self._preview_start(label, p["prompt"])
            self.log("RUN", label)
        elif event == "q_result":
            r = p["result"]
            mark = "✓" if r.passed else "✗"
            reason = "" if r.passed else " – " + reason_text(r.reason)
            if r.runs > 1:
                reason += " · " + t("bench.log.agree", same=r.identical, runs=r.runs, passes=r.passes)
            if r.invalid:
                reason += " · " + t("bench.log.invalid", n=r.invalid, total=r.invalid + r.runs)
            if r.cache_warn:
                reason += " · ⚠ " + t("bench.log.cache_warn", cached=r.cached, tokens=r.prompt_tokens)
            self._preview_verdict(f"{mark} {t('bench.preview.verdict')}{reason}", r.passed,
                                  {"model": r.model, "suite": r.suite, "test": r.test_id, "passed": int(r.passed),
                                   "reason": reason.lstrip(" –"), "prompt": r.prompt, "answer": r.answer})
            self.log("OK" if r.passed else "WARN", f"{r.model} | {mark} {r.title}{reason}")
        elif event == "q_done":
            self._quality_row(p["model"], p["results"])
            ok = sum(r.passed for r in p["results"])
            self.log("OK", t("bench.log.q_done", model=p["model"], suite=t(f"bench.suite.{p['suite']}"),
                             ok=ok, total=len(p["results"])))
        elif event == "q_skip":
            self._qstatus(p["model"], t("bench.status.skip"))
            self.log("WARN", t("bench.log.skip", model=p["model"], context=t(f"bench.suite.{p['suite']}")))
        elif event == "q_release_failed":
            self.log("WARN", t("bench.log.release_failed", model=p["model"]))
        elif event == "unload":
            self.log("INFO", t("bench.log.unload", model=p["model"]))

    def _done(self, results: list) -> None:
        self.running = False
        from aibobr.bench.quality import QualityResult
        self.quality = [r for r in results if isinstance(r, QualityResult) and not r.skipped]
        results = [r for r in results if not isinstance(r, QualityResult)]
        self.results = results
        if not self.winfo_exists():
            return
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        done = [r for r in results if r.tg_tps]
        elapsed = time.monotonic() - self.started
        took = t("bench.took", min=int(elapsed // 60), sec=int(elapsed % 60),
                 est=max(1, round(self.estimated / 60)))
        stopped = any(r.stopped for r in results)
        self.stage.configure(text=t("bench.stopped_note") if stopped else t("bench.saved"))
        self.times.configure(text=took)
        if not stopped:
            self.bar.configure(value=100)
            self.percent.configure(text="100%")
        self._power_warning(done)
        if len({r.model for r in done}) >= 2:
            best = max(done, key=lambda r: r.tg_tps)
            self.summary.configure(text=t("bench.fastest", model=best.model, tps=_num(best.tg_tps),
                                          context=fmt_context(best.context)))
        self.log("END", took)

    def _power_warning(self, done: list) -> None:
        from aibobr.bench.power import capped_during_test
        gpu = next((g for g in self.hw.gpus if g.selected and g.power_capped), None)
        peaks = [r.power_draw_w for r in done if r.power_draw_w]
        if not peaks or not capped_during_test(gpu, max(peaks)):
            return
        import sys
        key = "bench.power_capped_linux" if sys.platform.startswith("linux") else "bench.power_capped_other"
        text = t(key, draw=f"{max(peaks):.0f}", limit=f"{gpu.power_limit_w:.0f}", max=f"{gpu.power_max_w:.0f}")
        self.warning.configure(text="⚠ " + text)
        self.log("WARN", text)

    def _failed(self, e: BaseException) -> None:
        self.running = False
        if not self.winfo_exists():
            return
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        from aibobr.sources.base import SourceError
        text = t("bench.no_ollama") if isinstance(e, SourceError) else t("gui.unexpected", error=repr(e))
        self.stage.configure(text=text)
        self.log("ERROR", text)

    # ------------------------------------------------------------------ podgląd i jakość (12c)
    def show_preview(self) -> None:
        """„Przywróć” – okno podglądu znowu na wierzchu (w trakcie testu i po nim)."""
        from aibobr.gui.widgets.preview_window import PreviewWindow
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = PreviewWindow(self, self.on_status)
        self.preview_window.show()
        self.restore_button.state(["!disabled"])

    def _preview_start(self, title: str, prompt: str) -> None:
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.start(title, prompt)

    def _preview_add(self, text: str) -> None:
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.add(text)

    def _preview_verdict(self, text: str, ok: bool, row: dict = None) -> None:
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.verdict(text, ok, row)

    def _qstatus(self, model: str, text: str) -> None:
        iid = self.qrows.get(model)
        if iid and model not in self.qresults:          # wyniki pierwszego pakietu zostają widoczne
            self.qtable.item(iid, values=(model, text) + ("",) * (len(QCOLS) - 2))

    def _quality_row(self, model: str, results: list) -> None:
        from aibobr.bench.quality import score
        self.qresults[model] = self.qresults.get(model, []) + list(results)
        sc = score(self.qresults[model])

        def cell(key):
            return f"{sc[key][0]}/{sc[key][1]}" if key in sc else "–"
        values = (model,) + tuple(cell(c) for c in QCOLS[1:])
        iid = self.qrows.get(model)
        if iid:
            self.qtable.item(iid, values=values)
        else:
            self.qrows[model] = self.qtable.insert("", "end", values=values)

    def show_answers(self) -> None:
        """Dwuklik w tabeli Jakość: wszystkie zadania i odpowiedzi modelu z oceną."""
        sel = self.qtable.selection()
        if not sel:
            return
        model = self.qtable.item(sel[0], "values")[0]
        results = self.qresults.get(model)
        if not results:
            return
        win = tk.Toplevel(self)
        win.title(t("bench.answers_title", model=model))
        win.geometry("760x620")
        text = tk.Text(win, wrap="word", font="TkFixedFont", background="#11171d", foreground="#cfd8e3",
                       relief="flat", padx=12, pady=10)
        scroll = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for tag, color in (("head", "#66c9fa"), ("prompt", "#9bacc0"), ("ok", "#53d6a0"), ("bad", "#ff8282")):
            text.tag_configure(tag, foreground=color)
        for r in results:
            mark = "✓" if r.passed else "✗"
            reason = "" if r.passed else " – " + reason_text(r.reason)
            text.insert("end", f"{mark} {r.title}{reason}\n", ("ok" if r.passed else "bad",))
            text.insert("end", r.prompt + "\n", ("prompt",))
            text.insert("end", (r.answer.strip() or "–") + "\n\n")
        text.configure(state="disabled")
        self.answers_window = win


def reason_text(reason: str) -> str:
    """„quality.reason.no_function:fibonacci” → tekst z tłumaczeń z argumentem."""
    if not reason:
        return ""
    key, _, arg = reason.partition(":")
    return t(key, arg=arg) if key.startswith("quality.") else reason
