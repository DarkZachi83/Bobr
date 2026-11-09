"""Zakładka „Benchmark” (etap 13c, makieta autora) – zastępuje okna „Modele na dysku…”
i „Benchmark”.

Lewa kolumna: Modele na dysku – szukanie po kliknięciu (zgoda raz), lista zapamiętana między
uruchomieniami, werdykt AI Bóbra dla każdego modelu, pole ☐ do zaznaczenia modeli z Ollamy
(benchmark umie tylko je – llama-server i LM Studio w 12d). Dwuklik / „Sprawdź” – model
w zakładce Sprawdź.
Prawa (0.9, benchmark prędkości): przebiegi 1/3/5, długi kontekst 1k/8k/32k, plan, Start /
Zatrzymaj; pod nią zakładki Wydajność (werdykt z 15 zadań z podstawą, TTFT, prompt z 1k, długi
kontekst z TTFT i oceną informacyjną; dwuklik – szczegóły) i Odpowiedzi (podgląd, bez oceny),
dziennik na żywo i eksport sesji do folderu. Okno Monitor – telemetria i odpowiedź na żywo.
Modele mierzymy PO KOLEI – nigdy kilka naraz.
"""
from __future__ import annotations

import datetime
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from aibobr.gui import speed_format as sf
from aibobr.gui.theme import COLORS, VERDICT_COLORS
from aibobr.i18n import fmt_context, fmt_gb, t
from aibobr.scan.model import FoundModel, Kind, ScanReport

CONTEXTS = (1024, 8192, 32768)
DEFAULT_ON = {1024, 8192}                  # 32k na życzenie – na słabszym sprzęcie trwa długo
RUNS = (1, 3, 5)                           # domyślnie 1 (decyzja autora); 3 i 5 – pomiar rozrzutu
COLUMNS = ("model", "run", "verdict", "gen", "ttft", "pp", "l1024", "l8192", "l32768", "notes")
WIDTHS = {"model": 170, "run": 40, "verdict": 120, "gen": 270, "ttft": 130, "pp": 240, "l1024": 220,
          "l8192": 220, "l32768": 220, "notes": 240}
ACOLS = ("model", "run", "task", "tokens", "tps", "flags")
AWIDTHS = {"model": 120, "run": 40, "task": 160, "tokens": 60, "tps": 70, "flags": 140}
LEVEL_COLORS = {"fast": "#53d6a0", "useful": "#66c9fa", "slow": "#f5c65f", "too_slow": "#ff8282"}
TASKS = 15
COOL_EST, TASK_EST, LONG_EST = 60.0, 7.0, 12.0
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
        self.run_results: list = []                # przebiegi ostatniej sesji (wynik silnika)
        self.result_rows: dict = {}                # wiersz tabeli → przebieg (szczegóły)
        self.session_id: Optional[str] = None       # do eksportu
        self.current_run: Optional[tuple] = None
        self.ask = None                             # okno stabilizacji z wątku pomiaru
        self.log_lines: list[str] = []
        self.started = 0.0
        self.estimated = 0.0
        self.steps_total = 0
        self.steps_done = 0
        self.job_rows: dict = {}                   # (model, przebieg) → wiersz tabeli
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

    # ================================================================== benchmark prędkości (0.9)
    def _build_bench(self) -> None:
        top = ttk.Frame(self.right, style="Panel.TFrame", padding=10)
        top.pack(fill="x")
        ttk.Label(top, text=t("bench.title"), style="Title.TLabel").pack(anchor="w")
        what = ttk.Label(top, text=t("speed.what"), style="Muted.TLabel", wraplength=640, justify="left")
        what.pack(anchor="w", fill="x", pady=(2, 6))
        what.bind("<Configure>", lambda e: what.configure(wraplength=max(300, e.width)))
        opts = ttk.Frame(top, style="Inner.TFrame")
        opts.pack(fill="x")
        ttk.Label(opts, text=t("speed.runs"), style="Panel.TLabel").pack(side="left")
        self.runs = tk.IntVar(value=1)
        for n in RUNS:
            ttk.Radiobutton(opts, text=str(n), value=n, variable=self.runs, style="Panel.TRadiobutton",
                            command=self._update_plan).pack(side="left", padx=4)
        ttk.Label(opts, text="   " + t("speed.long"), style="Panel.TLabel").pack(side="left")
        self.ctx_vars = {}
        for ctx in CONTEXTS:
            var = tk.BooleanVar(value=ctx in DEFAULT_ON)
            ttk.Checkbutton(opts, text=fmt_context(ctx), variable=var, style="Panel.TCheckbutton",
                            command=self._update_plan).pack(side="left", padx=4)
            self.ctx_vars[ctx] = var
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
        ans = ttk.Frame(self.results_tabs, style="Panel.TFrame", padding=8)
        self.results_tabs.add(ans, text=t("speed.answers"))
        box = ttk.Frame(res, style="Inner.TFrame")
        box.pack(fill="both", expand=True)
        self.table = ttk.Treeview(box, columns=COLUMNS, show="headings", height=6)
        for col in COLUMNS:
            self.table.heading(col, text=fmt_context(int(col[1:])) if col.startswith("l") else t(f"speed.col.{col}"))
            self.table.column(col, width=WIDTHS.get(col, 150), minwidth=50, anchor="w",
                              stretch=col in ("model", "notes"))
        for name, color in LEVEL_COLORS.items():
            self.table.tag_configure(name, foreground=color)
        self.table.tag_configure("skip", foreground=COLORS["muted"])
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.table.yview)
        xscroll = ttk.Scrollbar(res, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        xscroll.pack(fill="x")
        self.table.bind("<Double-Button-1>", lambda e: self.show_details())
        hint = ttk.Label(res, text=t("speed.hint"), style="Muted.TLabel", wraplength=500, justify="left")
        hint.pack(anchor="w", fill="x", pady=(4, 0))
        hint.bind("<Configure>", lambda e: hint.configure(wraplength=max(200, e.width)))

        apane = ttk.Panedwindow(ans, orient="horizontal")
        apane.pack(fill="both", expand=True)
        alist = ttk.Frame(apane, style="Inner.TFrame")
        apane.add(alist, weight=2)
        self.answers = ttk.Treeview(alist, columns=ACOLS, show="headings", height=6, selectmode="browse")
        for col in ACOLS:
            self.answers.heading(col, text=t(f"speed.acol.{col}"))
            self.answers.column(col, width=AWIDTHS[col], minwidth=40, anchor="w", stretch=col == "task")
        ascroll = ttk.Scrollbar(alist, orient="vertical", command=self.answers.yview)
        self.answers.configure(yscrollcommand=ascroll.set)
        self.answers.pack(side="left", fill="both", expand=True)
        ascroll.pack(side="right", fill="y")
        self.answers.bind("<<TreeviewSelect>>", lambda e: self._show_answer())
        self.answer_text = tk.Text(apane, wrap="word", font="TkFixedFont", height=8, background="#11171d",
                                   foreground="#cfd8e3", relief="flat", padx=10, pady=8, state="disabled")
        self.answer_text.tag_configure("head", foreground="#66c9fa")
        self.answer_text.tag_configure("prompt", foreground="#9bacc0")
        apane.add(self.answer_text, weight=3)
        self.answer_rows: dict[str, tuple] = {}

        log = ttk.Frame(split, style="Panel.TFrame", padding=8)
        split.add(log, weight=3)
        self.monitor = None                    # okno „Monitor” – tworzone przy pierwszym Starcie
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
        ttk.Button(tools, text=t("speed.export"), command=self.export_session).pack(side="left")
        self.restore_button = ttk.Button(tools, text=t("bench.preview.restore"), command=self.show_monitor)
        self.restore_button.pack(side="right")
        ttk.Label(tools, text=t("speed.monitor.title") + ":", style="Panel.TLabel").pack(side="right", padx=6)
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

    @staticmethod
    def _model_name(target: str) -> str:
        return target[len("ollama:"):] if target.startswith("ollama:") else target

    def _plan_jobs(self) -> tuple[list[str], dict, list[str]]:
        """(modele do testu, model → pomijane długości, pominięte modele) wg przewidywania AI Bóbra:
        „nie zmieści się” przy 1k – model pomijamy; przy dłuższym kontekście – tylko ten pomiar."""
        run, skip, skipped = [], {}, []
        for tg in self.targets():
            if self.predicted.get((tg, min(CONTEXTS))) == "red":
                skipped.append(tg)
                continue
            run.append(tg)
            red = [c for c in self.contexts() if self.predicted.get((tg, c)) == "red"]
            if red:
                skip[tg] = red
        return run, skip, skipped

    def _update_plan(self) -> None:
        if self.running:
            return
        targets = self.targets()
        run, skip, skipped = self._plan_jobs()
        runs = int(self.runs.get())
        self.table.delete(*self.table.get_children())
        self.job_rows.clear()
        seconds = 0.0
        for tg in targets:
            name = self._model_name(tg)
            if tg in skipped:
                self.job_rows[(name, 1)] = self.table.insert(
                    "", "end", values=(name, "–", t("bench.status.skip")) + ("",) * (len(COLUMNS) - 3), tags=("skip",))
                continue
            ctxs = [c for c in self.contexts() if c not in skip.get(tg, ())]
            seconds += runs * estimate(ctxs)
            for n in range(1, runs + 1):
                values = [name, str(n), t("bench.status.wait")] + [""] * (len(COLUMNS) - 3)
                for c in skip.get(tg, ()):
                    values[COLUMNS.index(f"l{c}")] = t("bench.status.skip")
                self.job_rows[(name, n)] = self.table.insert("", "end", values=values)
        self.estimated = seconds
        minutes = max(1, round(seconds / 60))
        if not targets:
            self.plan.configure(text=t("bench.none_checked"))
        else:
            ctx = ", ".join(fmt_context(c) for c in self.contexts()) or "—"
            text = t("speed.plan", models=len(run), runs=runs, contexts=ctx, minutes=minutes)
            if skipped:
                text += " " + t("speed.plan_skipped", n=len(skipped))
            self.plan.configure(text=text)
        warnings = []
        if len(run) > 1:
            warnings.append(t("bench.warn_many", minutes=minutes))
        from aibobr.app.controller import power_source
        if power_source() == "battery":
            warnings.append(t("bench.warn_battery"))
        self.warning.configure(text="\n".join("⚠ " + w for w in warnings))
        self.steps_total = sum(runs * (2 + TASKS + len([c for c in self.contexts() if c not in skip.get(tg, ())]))
                               for tg in run)
        self.start_button.state(["!disabled"] if run else ["disabled"])

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

    def export_session(self) -> None:
        """Ostatnia sesja do folderu: werdykt, zadania, długi kontekst, telemetria, manifest.json."""
        from aibobr.bench import export
        db = getattr(self.c, "bench_db", None)
        if not self.session_id or db is None or db.session(self.session_id) is None:
            messagebox.showinfo(t("speed.export"), t("bench.csv_none"), parent=self)
            return
        parent = filedialog.askdirectory(parent=self, title=t("speed.export"), mustexist=True)
        if not parent:
            return
        folder = export.export_session(db, self.session_id, parent)
        self.on_status(t("speed.exported", path=str(folder)))
        self.log("INFO", t("speed.exported", path=str(folder)))

    # ------------------------------------------------------------------ pomiar
    def start(self) -> None:
        contexts, (run, skip, skipped) = self.contexts(), self._plan_jobs()
        runs = int(self.runs.get())             # odczyt tu – zmienne Tk nie z wątku w tle
        if not run or self.running:
            return
        if len(run) > 1 and not self.confirm(t("bench.title"), t("bench.confirm_many")):
            return
        try:
            hw, _ = self.current()
        except ValueError:
            messagebox.showinfo(t("bench.title"), t("stats.need_hw"), parent=self)
            return
        from aibobr.gui.dialogs.cooling import ThreadsafeAsk
        self.hw = hw
        self.running = True
        self.run_results = []
        self.session_id = None
        self.steps_done = 0
        self.started = time.monotonic()
        self.stop.clear()
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.summary.configure(text="")
        self.log_lines.clear()
        self.answers.delete(*self.answers.get_children())
        self.answer_rows.clear()
        self.show_monitor()                    # osobne okno (decyzja autora 4a)
        self.monitor.clear()
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        for tg in skipped:
            self.log("WARN", t("speed.log.skip", model=self._model_name(tg)))
        self.ask = ThreadsafeAsk(self.winfo_toplevel(), stop=self.stop)
        self._clock()
        self.worker.run_with_progress(
            "bench",
            lambda report: self.c.run_speed_benchmark(run, hw, report, self.stop, runs=runs, contexts=contexts,
                                                      on_timeout=self.ask, skip_contexts=skip),
            self._progress, self._done, self._failed)

    def _clock(self) -> None:
        if not self.running or not self.winfo_exists():
            return
        elapsed = time.monotonic() - self.started
        share = self.steps_done / self.steps_total if self.steps_total else 0
        left = elapsed * (1 - share) / share if share > 0.05 else max(0.0, self.estimated - elapsed)
        self.times.configure(text=t("bench.time_line", elapsed=_clock(elapsed), left=_clock(left)))
        self.after(500, self._clock)

    def _step(self) -> None:
        self.steps_done += 1
        share = min(1.0, self.steps_done / self.steps_total) if self.steps_total else 0
        self.bar.configure(value=share * 100)
        self.percent.configure(text=f"{share * 100:.0f}%")

    def _status(self, model: str, run: int, text: str) -> None:
        iid = self.job_rows.get((model, run))
        if iid:
            values = list(self.table.item(iid, "values"))
            values[COLUMNS.index("verdict")] = text
            self.table.item(iid, values=values)
            self.table.see(iid)

    def _phase(self, text: str) -> None:
        self.stage.configure(text=text)
        if self.monitor is not None and self.monitor.winfo_exists():
            self.monitor.set_phase(text)

    def _progress(self, p: dict) -> None:
        event = p.get("event")
        m = self.monitor if self.monitor is not None and self.monitor.winfo_exists() else None
        if event == "telemetry":
            if m:
                m.sample(p["sample"])
        elif event == "s_start":
            self.session_id = p["session_id"]
            self.log("START", t("speed.log.start", version=p.get("version") or "?", models=", ".join(p["models"]),
                                runs=p["runs"], tasks=p["tasks"],
                                contexts=", ".join(fmt_context(c) for c in p["contexts"]) or "—"))
            self.log("INFO", t("speed.disclaimer"))
        elif event == "s_release":
            self.current_run = (p["model"], p["run"])
            self._status(p["model"], p["run"], t("speed.status.release"))
            self.log("INFO", t("speed.log.release", model=p["model"], run=p["run"],
                               released=t("export.yes") if p["released"] else t("export.no"),
                               recovered=t("speed.unknown") if p["recovered"] is None else
                               t("export.yes") if p["recovered"] else t("export.no")))
        elif event == "s_cooling":
            if p.get("key") == "bench.cooling.start":
                text = t("bench.cooling.start", tolerance=sf.num(p["tolerance"]), window=sf.num(p["window"], 0))
                self.log("INFO", text)
                self._phase(t("speed.phase.cooling"))
                if self.current_run:
                    self._status(*self.current_run, t("speed.status.cooling"))
            else:
                self.log("WARN", t(p["key"]))
        elif event == "s_cooled":
            c = p["cooling"]
            self._step()
            self.log("WARN" if c.warning else "OK", t("speed.log.cooled", status=t(f"speed.stab.{c.status}"),
                                                      seconds=sf.num(c.seconds, 0), cpu=sf.num(c.cpu_temp_c, 0),
                                                      gpu=sf.num(c.gpu_temp_c, 0)))
            self._phase(t("speed.phase.baseline"))
            self._status(p["model"], p["run"], t("speed.status.baseline"))
        elif event == "s_baseline":
            self._step()
            mean = p["baseline"]["mean"]
            self.log("INFO", t("speed.log.baseline", cpu=sf.num(mean.get("cpu_pct")),
                               gpu=sf.num(mean.get("gpu_util_pct")), vram=sf.num(mean.get("vram_used_mb"), 0)))
        elif event == "s_task":
            self._step()
            label = t("speed.phase.task", model=p["model"], n=p["n"], total=p["total"], task=p["task"])
            self._phase(label)
            if self.current_run:
                self._status(*self.current_run, f"{p['n']}/{p['total']}")
            if m:
                m.start(label, p["prompt"])
        elif event == "chunk":
            if m:
                m.add(p["text"])
        elif event == "s_task_done":
            r = p["result"]
            state = (t(f"speed.error.{r.error.split(':')[0]}") if r.error else sf.flags(r)) or "ok"
            self.log("RESULT" if not r.error else "WARN",
                     t("speed.log.task", n=r.task_no, total=p["total"], task=r.task_id, tps=sf.num(sf.task_tps(r)),
                       tokens=r.gen_tokens or 0, ttft=sf.ms(r.ttft_wall_ms), state=state))
        elif event == "s_long":
            self._step()
            label = t("speed.phase.long", model=p["model"], context=fmt_context(p["context"]))
            self._phase(label)
            if m:
                m.start(label, t("speed.long_prompt"))
        elif event == "s_long_retry":
            text = t("speed.log.long_retry", context=fmt_context(p["context"]), tokens=sf.num(p["tokens"], 0),
                     low=sf.num(p["low"], 0), high=sf.num(p["high"], 0))
            self.log("INFO", f"{p['model']} | {text}")
            if m:
                m.start(t("speed.phase.long", model=p["model"], context=fmt_context(p["context"]))
                        + " · " + t("speed.long_retry_short"), t("speed.long_prompt"))
        elif event == "s_long_done":
            lc = p["result"]
            self.log("RESULT" if lc.get("valid") else "WARN",
                     f"{p['model']} | {fmt_context(lc['num_ctx'])}: {sf.long_cell(lc_run(lc), lc['num_ctx'], self._th())}"
                     + (" · " + t("speed.log.long_input", tokens=sf.num(lc.get("prompt_tokens"), 0),
                                  ctx=sf.num(lc["num_ctx"], 0)) if lc.get("prompt_tokens") else ""))
        elif event == "s_run_done":
            self._run_done(p["result"])
        elif event == "unload":
            self.log("INFO", t("bench.log.unload", model=p["model"]))
        elif event == "stopped":
            if self.current_run:
                self._status(*self.current_run, t("bench.status.stopped"))
            self.log("WARN", t("bench.log.stopped"))
        elif event == "s_error":
            self.log("ERROR", p.get("error", ""))

    def _th(self) -> dict:
        from aibobr.bench import speed
        return speed.thresholds_of(self.c.settings)

    def _run_done(self, r: dict) -> None:
        self.run_results.append(r)
        th = self._th()
        values = {"model": r["model"], "run": str(r["run_no"]), "verdict": sf.verdict(r), "gen": sf.generation(r),
                  "ttft": sf.ttft(r), "pp": sf.prompt_main(r), "notes": sf.warnings(r) or "—"}
        for c in CONTEXTS:
            values[f"l{c}"] = sf.long_cell(r, c, th)
        iid = self.job_rows.get((r["model"], r["run_no"]))
        row = [values[c] for c in COLUMNS]
        tags = (r.get("verdict"),) if r.get("verdict") else ()
        if iid:
            old = self.table.item(iid, "values")
            for c in CONTEXTS:                                 # „pominięty” zostaje
                i = COLUMNS.index(f"l{c}")
                if old[i] == t("bench.status.skip"):
                    row[i] = old[i]
            self.table.item(iid, values=row, tags=tags)
        else:
            iid = self.table.insert("", "end", values=row, tags=tags)
            self.job_rows[(r["model"], r["run_no"])] = iid
        self.result_rows[iid] = r
        self.log("OK" if r.get("status") == "finished" else "WARN",
                 t("speed.log.verdict", model=r["model"], run=r["run_no"], verdict=sf.verdict(r),
                   gen=sf.generation(r), ttft=sf.ttft(r)))
        if sf.warnings(r):
            self.log("WARN", sf.warnings(r))
        for task in r.get("tasks") or ():
            aid = self.answers.insert("", "end", values=(r["model"], r["run_no"], task.task_id, task.gen_tokens or 0,
                                                         sf.num(sf.task_tps(task)),
                                                         (t(f"speed.error.{task.error.split(':')[0]}")
                                                          if task.error else sf.flags(task))))
            self.answer_rows[aid] = (r["model"], task)

    def _show_answer(self) -> None:
        sel = self.answers.selection()
        if not sel or sel[0] not in self.answer_rows:
            return
        model, task = self.answer_rows[sel[0]]
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", "end")
        self.answer_text.insert("end", f"{model} · {task.task_id}\n", ("head",))
        self.answer_text.insert("end", t("bench.preview.prompt") + "\n", ("head",))
        self.answer_text.insert("end", task.prompt_text + "\n\n", ("prompt",))
        self.answer_text.insert("end", t("bench.preview.answer") + "\n", ("head",))
        self.answer_text.insert("end", (task.response_text or "").strip() or "–")
        if task.error:
            self.answer_text.insert("end", "\n\n⚠ " + task.error)
        self.answer_text.configure(state="disabled")

    def show_details(self) -> None:
        sel = self.table.selection()
        r = self.result_rows.get(sel[0]) if sel else None
        if r is None:
            return
        win = tk.Toplevel(self)
        win.title(t("speed.details_title", model=r["model"], run=r["run_no"]))
        win.geometry("820x640")
        text = tk.Text(win, wrap="none", font="TkFixedFont", background="#11171d", foreground="#cfd8e3",
                       relief="flat", padx=12, pady=10)
        scroll = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        text.insert("end", sf.details(r, self._th()))
        text.configure(state="disabled")
        self.details_window = win

    def _done(self, result: dict) -> None:
        self.running = False
        self._close_ask()
        if not self.winfo_exists():
            return
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        elapsed = time.monotonic() - self.started
        took = t("bench.took", min=int(elapsed // 60), sec=int(elapsed % 60),
                 est=max(1, round(self.estimated / 60)))
        stopped = result.get("status") == "stopped"
        self.stage.configure(text=t("bench.stopped_note") if stopped else t("bench.saved"))
        if self.monitor is not None and self.monitor.winfo_exists():
            self.monitor.set_phase(t("speed.phase.done"))
        self.times.configure(text=took)
        if not stopped:
            self.bar.configure(value=100)
            self.percent.configure(text="100%")
        self._summary()
        self.log("END", took)

    def _summary(self) -> None:
        from aibobr.bench import speed
        models = []
        for r in self.run_results:
            if r["model"] not in models:
                models.append(r["model"])
        parts = []
        for model in models:
            runs = [r for r in self.run_results if r["model"] == model]
            rng = speed.spread(runs)
            if rng and len(runs) > 1:
                parts.append(t("speed.spread", model=model, runs=len(runs), low=sf.num(rng[0]), high=sf.num(rng[1])))
        if parts:
            self.summary.configure(text=" · ".join(parts))
            for p in parts:
                self.log("INFO", p + " – " + t("speed.spread_note"))

    def _close_ask(self) -> None:
        ask = getattr(self, "ask", None)
        if ask is not None:
            ask.close()
            self.ask = None

    def _failed(self, e: BaseException) -> None:
        self.running = False
        self._close_ask()
        if not self.winfo_exists():
            return
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        from aibobr.sources.base import SourceError
        text = t("bench.no_ollama") if isinstance(e, SourceError) else t("gui.unexpected", error=repr(e))
        self.stage.configure(text=text)
        self.log("ERROR", text)

    # ------------------------------------------------------------------ Monitor
    def show_monitor(self) -> None:
        """„Przywróć” – okno Monitor znowu na wierzchu (w trakcie testu i po nim)."""
        from aibobr.gui.widgets.monitor_window import MonitorWindow
        if self.monitor is None or not self.monitor.winfo_exists():
            self.monitor = MonitorWindow(self, self.on_status)
        self.monitor.show()
        self.restore_button.state(["!disabled"])


def lc_run(lc: dict) -> dict:
    """Jeden pomiar długiego kontekstu jako „przebieg” – do wspólnego formatowania komórki."""
    return {"long_context": [lc]}


def estimate(contexts: list[int]) -> float:
    """Szacunek jednego przebiegu jednego modelu w s: chłodzenie (zależy od komputera, przyjmujemy
    ok. 1 min) + linia bazowa + 15 zadań (zwolnienie, wczytanie, odpowiedź) + długi kontekst."""
    return COOL_EST + 10 + TASKS * TASK_EST + sum(LONG_EST + c / 1000 * 2.5 for c in contexts)
