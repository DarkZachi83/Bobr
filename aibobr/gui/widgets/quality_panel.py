"""Zakładka „Jakość” (moduł jakości 0.10, decyzje autora 9–10.11.2026): do czego model się nadaje.

Lewa strona – ta sama lista modeli co w Benchmarku (dziedziczona). Prawa: wybór testów
(humanistyczny PL, ścisły PL, ścisły EN), 1 albo 3 próby (przy 3 – komunikat o czasie), prognoza
czasu i zegar, stan piaskownicy, podgląd na żywo (przy angielskim pytaniu polskie tłumaczenie –
model go nie widzi). Wyniki: osobno humanistyczny i ścisły, ułamki w kategoriach i poziomach,
pokrycie, niezgodny format, etykieta „w tych testach…”; zadania z odpowiedziami i oceną ręczną;
historia sesji z eksportem i usuwaniem. Bez wspólnej oceny i bez wpływu na werdykt prędkości.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from aibobr.gui.widgets.bench_panel import LOG_COLORS, BenchmarkPanel, _clock
from aibobr.i18n import current_language, has_key, t
from aibobr.quality import score

TESTS = (("humanistyczny_v1", "pl"), ("scisly_v1", "pl"), ("scisly_v1", "en"))
MARK = {True: "✓", False: "✗", None: "–"}
RCOLS = ("model", "test", "score", "coverage", "format", "levels")
RWIDTHS = {"model": 170, "test": 170, "score": 130, "coverage": 230, "format": 120, "levels": 260}
TCOLS = ("model", "test", "task", "level", "category", "result", "reason", "frag")
TWIDTHS = {"model": 110, "test": 115, "task": 45, "level": 65, "category": 125, "result": 30, "reason": 230,
           "frag": 80}
HCOLS = ("date", "models", "tests", "attempts", "status")
HWIDTHS = {"date": 130, "models": 260, "tests": 260, "attempts": 60, "status": 100}


CUSTOM_NAMES: dict[str, str] = {}      # id własnego testu → nazwa (uzupełnia panel; usunięty test – samo id)


def test_name(suite: str, lang: str) -> str:
    key = f"q.test.{suite}.{lang}"
    if has_key(key):
        return t(key)
    return f"{CUSTOM_NAMES.get(suite, suite)} ({lang.upper()})"


def frac(correct: int, evaluated: int) -> str:
    return f"{correct}/{evaluated}" if evaluated else "–"


def pct(correct: int, evaluated: int) -> str:
    return f" ({100 * correct / evaluated:.0f}%)" if evaluated else ""


class QualityPanel(BenchmarkPanel):
    # ------------------------------------------------------------------ budowa (prawa strona)
    def _build_bench(self) -> None:
        self.q_results: list[dict] = []
        self.q_session: Optional[str] = None
        self.task_rows: dict[str, tuple] = {}
        self.result_by_row: dict[str, dict] = {}
        self.hist_rows: dict[str, str] = {}
        self.ask = None
        top = ttk.Frame(self.right, style="Panel.TFrame", padding=10)
        top.pack(fill="x")
        ttk.Label(top, text=t("q.title"), style="Title.TLabel").pack(anchor="w")
        what = ttk.Label(top, text=t("q.what"), style="Muted.TLabel", wraplength=640, justify="left")
        what.pack(anchor="w", fill="x", pady=(2, 6))
        what.bind("<Configure>", lambda e: what.configure(wraplength=max(300, e.width)))
        self.opts = ttk.Frame(top, style="Inner.TFrame")
        self.opts.pack(fill="x")
        self.test_vars = {}
        self.suite_data: dict[str, dict] = {}
        self._build_tests()
        line = ttk.Frame(top, style="Inner.TFrame")
        line.pack(fill="x", pady=(6, 0))
        ttk.Label(line, text=t("q.attempts"), style="Panel.TLabel").pack(side="left")
        self.attempts = tk.IntVar(value=1)
        ttk.Radiobutton(line, text="1", value=1, variable=self.attempts, style="Panel.TRadiobutton",
                        command=self._update_plan).pack(side="left", padx=4)
        ttk.Radiobutton(line, text="3", value=3, variable=self.attempts, style="Panel.TRadiobutton",
                        command=self._three).pack(side="left", padx=4)
        ttk.Button(line, text=t("q.ed.open"), command=self.open_editor).pack(side="right")
        self.plan = ttk.Label(top, text="", style="Panel.TLabel", wraplength=640, justify="left")
        self.plan.pack(anchor="w", pady=(8, 0), fill="x")
        self.sandbox_line = ttk.Label(top, text="", style="Note.TLabel", wraplength=640, justify="left")
        self.sandbox_line.pack(anchor="w", fill="x")
        for label in (self.plan, self.sandbox_line):
            label.bind("<Configure>", lambda e, w=label: w.configure(wraplength=max(300, e.width)))
        buttons = ttk.Frame(top, style="Inner.TFrame")
        buttons.pack(fill="x", pady=(8, 0))
        self.start_button = ttk.Button(buttons, text="▶ " + t("bench.start"), style="Accent.TButton",
                                       command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="■ " + t("bench.stop"), command=self.stop.set)
        self.stop_button.pack(side="left", padx=6)
        self.stop_button.state(["disabled"])
        self.summary = ttk.Label(buttons, text="", style="Panel.TLabel", font=("TkDefaultFont", 10, "bold"),
                                 wraplength=500, justify="left")
        self.summary.pack(side="left", padx=10)
        prog = ttk.Frame(top, style="Inner.TFrame")
        prog.pack(fill="x", pady=(8, 0))
        self.percent = ttk.Label(prog, text="", style="Panel.TLabel", font=("TkDefaultFont", 10, "bold"))
        self.percent.pack(side="right")
        self.stage = ttk.Label(prog, text="", style="Note.TLabel", wraplength=300, justify="left")
        self.stage.pack(side="left", fill="x", expand=True)
        self.stage.bind("<Configure>", lambda e: self.stage.configure(wraplength=max(150, e.width)))
        self.bar = ttk.Progressbar(top, mode="determinate", maximum=100)
        self.bar.pack(fill="x")
        self.times = ttk.Label(top, text="", style="Muted.TLabel")
        self.times.pack(anchor="w", pady=(2, 0))

        self.results_tabs = ttk.Notebook(self.right)
        self.results_tabs.pack(fill="both", expand=True, pady=(8, 0))
        self._build_results()
        self._build_tasks()
        self._build_live()
        self._build_history()
        self.results_tabs.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed())
        self.log_lines: list[str] = []

    def _tree(self, parent, cols, widths, prefix, height=6, stretch=("model",)):
        box = ttk.Frame(parent, style="Inner.TFrame")
        box.pack(fill="both", expand=True)
        tree = ttk.Treeview(box, columns=cols, show="headings", height=height, selectmode="browse")
        for col in cols:
            tree.heading(col, text=t(f"{prefix}.{col}"), anchor="w")
            tree.column(col, width=widths[col], minwidth=40, anchor="w", stretch=col in stretch)
        scroll = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return tree

    def _text(self, parent, height=8):
        text = tk.Text(parent, wrap="word", font="TkFixedFont", height=height, background="#11171d",
                       foreground="#cfd8e3", relief="flat", padx=10, pady=8, state="disabled")
        for tag, color in {"head": "#66c9fa", "prompt": "#9bacc0", "tr": "#f5c65f", "ok": "#53d6a0",
                           "bad": "#ff8282", **LOG_COLORS}.items():
            text.tag_configure(tag, foreground=color)
        return text

    def _build_results(self) -> None:
        tab = ttk.Frame(self.results_tabs, style="Panel.TFrame", padding=8)
        self.results_tabs.add(tab, text=t("q.tab.results"))
        self.label_line = ttk.Label(tab, text="", style="Title.TLabel", wraplength=700, justify="left")
        self.label_line.pack(anchor="w", fill="x", pady=(0, 6))
        self.label_line.bind("<Configure>", lambda e: self.label_line.configure(wraplength=max(300, e.width)))
        pane = ttk.Panedwindow(tab, orient="vertical")
        pane.pack(fill="both", expand=True)
        top = ttk.Frame(pane, style="Inner.TFrame")
        pane.add(top, weight=1)
        self.rtable = self._tree(top, RCOLS, RWIDTHS, "q.rcol", height=5)
        self.rtable.bind("<<TreeviewSelect>>", lambda e: self._show_breakdown())
        self.breakdown = self._text(pane, height=8)
        pane.add(self.breakdown, weight=1)

    def _build_tasks(self) -> None:
        tab = ttk.Frame(self.results_tabs, style="Panel.TFrame", padding=8)
        self.results_tabs.add(tab, text=t("q.tab.tasks"))
        pane = ttk.Panedwindow(tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, style="Inner.TFrame")
        pane.add(left, weight=2)
        self.ttable = self._tree(left, TCOLS, TWIDTHS, "q.tcol", height=8, stretch=("model", "reason"))
        for tag, color in (("ok", "#53d6a0"), ("bad", "#ff8282"), ("none", "#9bacc0")):
            self.ttable.tag_configure(tag, foreground=color)
        self.ttable.bind("<<TreeviewSelect>>", lambda e: self._show_task())
        right = ttk.Frame(pane, style="Inner.TFrame")
        pane.add(right, weight=3)
        grade = ttk.Frame(right, style="Inner.TFrame")
        grade.pack(fill="x", pady=(0, 4))
        ttk.Label(grade, text=t("q.manual"), style="Panel.TLabel").pack(side="left")
        self.pass_button = ttk.Button(grade, text="✓ " + t("q.manual.pass"), command=lambda: self.grade(True))
        self.pass_button.pack(side="left", padx=4)
        self.fail_button = ttk.Button(grade, text="✗ " + t("q.manual.fail"), command=lambda: self.grade(False))
        self.fail_button.pack(side="left")
        for b in (self.pass_button, self.fail_button):
            b.state(["disabled"])
        self.task_text = self._text(right, height=14)
        self.task_text.pack(fill="both", expand=True)

    def _build_live(self) -> None:
        tab = ttk.Frame(self.results_tabs, style="Panel.TFrame", padding=8)
        self.results_tabs.add(tab, text=t("q.tab.live"))
        tools = ttk.Frame(tab, style="Inner.TFrame")
        tools.pack(fill="x")
        ttk.Button(tools, text=t("bench.log.copy"), command=self.copy_log).pack(side="left")
        ttk.Button(tools, text=t("bench.log.save"), command=self.save_log).pack(side="left", padx=6)
        self.live = self._text(tab, height=14)
        self.live.pack(fill="both", expand=True, pady=(6, 0))
        self.console = self.live                       # dziennik dziedziczony z Benchmarku pisze tutaj
        self.autoscroll = tk.BooleanVar(value=True)

    def _build_history(self) -> None:
        tab = ttk.Frame(self.results_tabs, style="Panel.TFrame", padding=8)
        self.history_tab = tab
        self.results_tabs.add(tab, text=t("hist.tab"))
        bar = ttk.Frame(tab, style="Inner.TFrame")
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text=t("hist.details"), command=self.history_details).pack(side="left")
        ttk.Button(bar, text=t("speed.export"), command=self.export_history).pack(side="left", padx=6)
        ttk.Button(bar, text=t("hist.delete"), command=self.delete_history).pack(side="left")
        self.htable = self._tree(tab, HCOLS, HWIDTHS, "q.hcol", height=8, stretch=("models", "tests"))
        self.htable.bind("<Double-Button-1>", lambda e: self.history_details())

    # ------------------------------------------------------------------ plan
    def _targets_changed(self) -> None:
        self._update_plan()

    def tests(self) -> list[tuple[str, str]]:
        return [key for key, var in self.test_vars.items() if var.get()]

    def _custom_suites(self) -> list[dict]:
        store = getattr(self.c, "custom_suites", None)
        try:
            return store.valid() if store is not None else []
        except OSError:
            return []

    def _build_tests(self) -> None:
        """Pola wyboru testów: standardowe (domyślnie zaznaczone) i własne – każda wersja językowa osobno
        (np. „Mój test (KO)”, „Mój test (EN)”; domyślnie niezaznaczone). Odświeżane po zapisie w edytorze."""
        from aibobr.quality import suites
        old = {key: var.get() for key, var in self.test_vars.items()}
        for child in self.opts.winfo_children():
            child.destroy()
        self.test_vars = {}
        self.suite_data = {}
        for data in suites.standard():
            self.suite_data[data["suite_info"]["id"]] = data
        for key in TESTS:
            self._test_box(key, old.get(key, True), 0, len(self.test_vars))
        custom = self._custom_suites()
        if custom:
            ttk.Label(self.opts, text=t("q.custom_tests"), style="Muted.TLabel").grid(
                row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        n = 0
        for data in custom:
            info = data["suite_info"]
            name = info["name"].get(current_language()) or next(iter(info["name"].values()))
            CUSTOM_NAMES[info["id"]] = name
            self.suite_data[info["id"]] = data
            for lang in info["languages"]:
                self._test_box((info["id"], lang), old.get((info["id"], lang), False), 2 + n // 3, n % 3)
                n += 1

    def _test_box(self, key, value: bool, row: int, column: int) -> None:
        var = tk.BooleanVar(value=value)
        ttk.Checkbutton(self.opts, text=test_name(*key), variable=var, style="Panel.TCheckbutton",
                        command=self._update_plan).grid(row=row, column=column, sticky="w", padx=(0, 14))
        self.test_vars[key] = var

    def _suites_changed(self) -> None:
        if not self.running:
            self._build_tests()
        self._update_plan()

    def _tasks(self, tests) -> int:
        return sum(len(self.suite_data.get(s, {}).get("tasks") or []) for s, _ in tests)

    def _python_tasks(self, tests) -> int:
        return sum(1 for s, _ in tests for x in self.suite_data.get(s, {}).get("tasks") or []
                   if (x.get("check") or {}).get("type") == "python")

    def _sandbox(self):
        if getattr(self, "_sandbox_obj", None) is None:
            from aibobr.quality.sandbox import Sandbox
            self._sandbox_obj = Sandbox()
        return self._sandbox_obj

    def estimate(self, targets: list[str], tests: list, attempts: int) -> float:
        """Prognoza z ostatniego benchmarku prędkości modelu (wczytanie, tok/s); bez niego – zgrubnie."""
        db = getattr(self.c, "bench_db", None)
        tasks = self._tasks(tests)
        python = self._python_tasks(tests)
        total = 0.0
        for tg in targets:
            last = db.last_speed(self._model_name(tg)) if db is not None else None
            load = (last.get("load_wall_ms") or 3000) / 1000 if last else 3.0
            tps = (last.get("gen_tps") or 20.0) if last else 20.0
            total += score.estimate_seconds(tasks, attempts, load, tps, avg_tokens=60.0, python_tasks=python)
        return total

    def _update_plan(self) -> None:
        if self.running or not hasattr(self, "test_vars"):
            return
        targets, tests, attempts = self.targets(), self.tests(), int(self.attempts.get())
        self.estimated = self.estimate(targets, tests, attempts)
        self.steps_total = len(targets) * self._tasks(tests) * attempts
        if not targets:
            self.plan.configure(text=t("bench.none_checked"))
        else:
            self.plan.configure(text=t("q.plan", models=len(targets), tasks=self._tasks(tests), attempts=attempts,
                                       minutes=max(1, round(self.estimated / 60))))
        self.start_button.state(["!disabled"] if targets and tests else ["disabled"])
        if self._python_tasks(tests):
            st = self._sandbox().status()
            self.sandbox_line.configure(text=t("q.sandbox.ok") if st.ready else
                                        t("q.sandbox.no", reason=t(f"q.sandbox.reason.{st.reason.split(':')[0]}")))
        else:
            self.sandbox_line.configure(text="")

    def open_editor(self) -> None:
        """Edytor testów – osobne okno (tworzenie testów jest opcjonalne); jedno naraz."""
        from aibobr.gui.dialogs.quality_editor import TestEditor
        win = getattr(self, "editor", None)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            return
        self.editor = TestEditor(self, self.c.custom_suites, sandbox=self._sandbox(), busy=lambda: self.running,
                                 on_change=self._suites_changed)

    def _three(self) -> None:
        """Trzy próby – komunikat autora o czasie testu i sugestia, by najpierw zrobić test z jedną próbą."""
        targets, tests = self.targets(), self.tests()
        one = max(1, round(self.estimate(targets, tests, 1) / 60))
        three = max(1, round(self.estimate(targets, tests, 3) / 60))
        if not self.confirm(t("q.three.title"), t("q.three.body", three=three, one=one)):
            self.attempts.set(1)
        self._update_plan()

    # ------------------------------------------------------------------ przebieg
    def start(self) -> None:
        targets, tests, attempts = self.targets(), self.tests(), int(self.attempts.get())
        if not targets or not tests or self.running:
            return
        self.bos_gate(targets, lambda tg, statuses, pairs: self._start(tg, tests, attempts, statuses, pairs))

    def _start(self, targets, tests, attempts, statuses, pairs) -> None:
        self.bos_pairs = pairs                      # poprawiona wersja → oryginał (porównanie „oba”)
        self.steps_total = len(targets) * self._tasks(tests) * attempts
        self.estimated = self.estimate(targets, tests, attempts)
        self.running = True
        self.stop.clear()
        self.started = time.monotonic()
        self.steps_done = 0
        self.q_results = []
        self.q_session = None
        for tree in (self.rtable, self.ttable):
            tree.delete(*tree.get_children())
        self.task_rows.clear()
        self.result_by_row.clear()
        self.__dict__.pop("frag_cache", None)              # świeże znaczniki kruchości przy każdym teście
        self.label_line.configure(text="")
        self.summary.configure(text="")
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.log_lines.clear()
        self._set(self.live, "")
        self.results_tabs.select(2)                    # na żywo
        hw = None
        try:
            hw, _ = self.current()
        except ValueError:
            pass
        self._clock()
        self.worker.run_with_progress(
            "quality",
            lambda report: self._recorded("quality", self.c.run_quality(
                targets, tests, report, self.stop, attempts=attempts, hw=hw, sandbox=self._sandbox()), statuses),
            self._progress, self._done, self._failed)

    def _clock(self) -> None:
        if not self.running or not self.winfo_exists():
            return
        elapsed = time.monotonic() - self.started
        share = self.steps_done / self.steps_total if self.steps_total else 0
        left = elapsed * (1 - share) / share if share > 0.05 else max(0.0, self.estimated - elapsed)
        self.times.configure(text=t("bench.time_line", elapsed=_clock(elapsed), left=_clock(left)))
        self.after(500, self._clock)

    def _set(self, widget, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if text:
            widget.insert("end", text)
        widget.configure(state="disabled")

    def _append(self, text: str, tag: str = "") -> None:
        self.live.configure(state="normal")
        self.live.insert("end", text, (tag,) if tag else ())
        self.live.configure(state="disabled")
        if self.autoscroll.get():
            self.live.see("end")

    def _progress(self, p: dict) -> None:
        event = p.get("event")
        if event == "q_start":
            self.q_session = p["session_id"]
            self.log("START", t("q.log.start", version=p.get("version") or "?", models=", ".join(p["models"]),
                                attempts=p["attempts"], tests=", ".join(p["tests"])))
        elif event == "q_run":
            self.log("INFO", f"{p['model']} | {test_name(p['suite'], p['lang'])}")
        elif event == "q_task":
            self.steps_done += 1
            share = min(1.0, self.steps_done / self.steps_total) if self.steps_total else 0
            self.bar.configure(value=share * 100)
            self.percent.configure(text=f"{share * 100:.0f}%")
            attempt = f" · {t('q.attempt', n=p['attempt'], of=p['attempts'])}" if p["attempts"] > 1 else ""
            self.stage.configure(text=f"{p['model']} | {test_name(p['suite'], p['lang'])} | "
                                      f"{p['n']}/{p['total']} {p['task']}{attempt}")
            self._append(f"\n▶ {p['task']}{attempt}\n", "head")
            self._append(p["prompt"] + "\n", "prompt")
            if p.get("translation"):
                self._append(t("q.translation") + " " + p["translation"] + "\n", "tr")
            self._append(t("bench.preview.answer") + "\n", "head")
        elif event == "chunk":
            self._append(p["text"])
        elif event == "q_task_done":
            r = p["result"]
            self._append(f"\n{MARK[r.correct]} {t('q.reason.' + r.reason)}\n", "ok" if r.correct else "bad")
            self._add_task(p["model"], p["suite"], p["lang"], r)
        elif event == "q_run_done":
            self._add_result(p["result"])
        elif event == "unload":
            self.log("INFO", t("bench.log.unload", model=p["model"]))
        elif event == "stopped":
            self.log("WARN", t("bench.log.stopped"))

    def log(self, kind: str, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {kind:<6} {text}"
        self.log_lines.append(line)
        self._append("\n" + line + "\n", kind)

    def _fragility(self, model: str, suite: str, lang: str, task_id: str) -> str:
        """Znacznik z ostatniej diagnostyki kruchości tego testu (zakładka Diagnostyka); pusty – nie było.
        Niezaliczone zadanie kruche to „mogło wyjść inaczej”, a nie twardy błąd."""
        key = (model, suite, lang)
        cache = self.__dict__.setdefault("frag_cache", {})
        if key not in cache:
            db = self.c.bench_db
            version = (self.suite_data.get(suite) or {}).get("suite_info", {}).get("version")
            try:
                cache[key] = db.last_fragility(model, suite, lang, version) if db is not None else {}
            except Exception:
                cache[key] = {}
        label = cache[key].get(task_id)
        return t(f"d.label.{label}") if label in ("stable", "fragile", "tie") else ""

    @staticmethod
    def _category(category: str) -> str:
        return t(f"q.cat.{category}") if has_key(f"q.cat.{category}") else category

    def _add_task(self, model: str, suite: str, lang: str, r) -> None:
        tag = "ok" if r.correct else "bad" if r.correct is False else "none"
        reason = t(f"q.reason.{r.reason}") + (" · " + t("q.differing") if r.differing else "")
        iid = self.ttable.insert("", "end", values=(model, test_name(suite, lang), r.task_id, t(f"q.level.{r.level}"),
                                                     self._category(r.category), MARK[r.correct], reason,
                                                     self._fragility(model, suite, lang, r.task_id)), tags=(tag,))
        self.task_rows[iid] = (model, suite, lang, r, None, None)

    def _add_result(self, run: dict) -> None:
        self.q_results.append(run)
        s = run["summary"]
        o = s.overall
        cover = t("q.coverage", evaluated=o.evaluated, total=o.total) if o.partial else t("q.coverage_full")
        levels = " · ".join(f"{t('q.level.' + k)} {frac(v.correct, v.evaluated)}"
                            for k, v in sorted(s.by_level.items(), key=lambda kv: ("easy", "medium", "hard").index(kv[0])))
        iid = self.rtable.insert("", "end", values=(run["model"], test_name(run["suite"], run["lang"]),
                                                     frac(o.correct, o.evaluated) + pct(o.correct, o.evaluated),
                                                     cover, str(s.format_noncompliant), levels))
        self.result_by_row[iid] = run
        for tid, (model, suite, lang, r, _, _) in list(self.task_rows.items()):
            if model == run["model"] and suite == run["suite"] and lang == run["lang"] and run.get("id"):
                self.task_rows[tid] = (model, suite, lang, r, run["id"], r.task_no)
        self._label()

    def _label(self) -> None:
        """Etykieta „w tych testach…” – tylko testy standardowe. Własny test w dwóch wersjach: obie obok siebie
        (różnica pokazuje, ile model traci we własnym języku)."""
        lines = []
        for model in dict.fromkeys(r["model"] for r in self.q_results):
            mine = {(r["suite"], r["lang"]): r["summary"].overall for r in self.q_results if r["model"] == model}
            parts = []
            lab = score.label(mine.get(("humanistyczny_v1", "pl")), mine.get(("scisly_v1", "pl")))
            if lab.kind != "none":
                h, s = mine[("humanistyczny_v1", "pl")], mine[("scisly_v1", "pl")]
                text = t(f"q.label.{lab.kind}", h=frac(h.correct, h.evaluated), s=frac(s.correct, s.evaluated))
                if lab.partial:
                    text += " · " + t("q.partial")
                en = mine.get(("scisly_v1", "en"))
                if en and en.evaluated and s.evaluated:
                    text += " · " + t("q.pl_en", pl=frac(s.correct, s.evaluated), en=frac(en.correct, en.evaluated))
                parts.append(text)
            for suite in dict.fromkeys(k[0] for k in mine if k[0] not in ("humanistyczny_v1", "scisly_v1")):
                langs = (self.suite_data.get(suite) or {}).get("suite_info", {}).get("languages") or []
                if len(langs) == 2 and all((suite, l) in mine and mine[(suite, l)].evaluated for l in langs):
                    a, b = mine[(suite, langs[0])], mine[(suite, langs[1])]
                    parts.append(t("q.versions", name=CUSTOM_NAMES.get(suite, suite), main=langs[0].upper(),
                                   a=frac(a.correct, a.evaluated), en=frac(b.correct, b.evaluated)))
            if parts:
                lines.append(f"{model}: " + " · ".join(parts))
        lines += self.bos_comparison()
        self.label_line.configure(text="\n".join(lines))

    def bos_comparison(self) -> list[str]:
        """„Oba – porównanie” (decyzja autora 12.11, pkt 5): oryginał i poprawiona wersja tego samego testu obok
        siebie; każde zadanie, które zmieniło wynik, z opisem z ostatniej diagnostyki kruchości oryginału:
        kruche/remis – „przeważony remis”, stabilne – „rzeczywista różnica wejścia”."""
        out = []
        results = {(r["model"], r["suite"], r["lang"]): r["summary"].overall for r in self.q_results}
        tasks: dict = {}
        for model, suite, lang, r, _, _ in self.task_rows.values():
            tasks.setdefault((model, suite, lang), {})[r.task_id] = r.correct
        for variant, original in (getattr(self, "bos_pairs", None) or {}).items():
            for (model, suite, lang), a in results.items():
                b = results.get((variant, suite, lang))
                if model != original or b is None:
                    continue
                version = (self.suite_data.get(suite) or {}).get("suite_info", {}).get("version")
                db = self.c.bench_db
                frag = db.last_fragility(original, suite, lang, version) if db is not None else {}
                before, after = tasks.get((original, suite, lang), {}), tasks.get((variant, suite, lang), {})
                changes = []
                for task_id, was in before.items():
                    now = after.get(task_id)
                    if was is None or now is None or was == now:
                        continue
                    label = frag.get(task_id)
                    why = (t("d.cmp.flip") if label in ("fragile", "tie") else t("d.cmp.real") if label == "stable"
                           else t("d.cmp.nodiag"))
                    changes.append(f"{task_id} {MARK[was]}→{MARK[now]} ({why})")
                out.append(t("d.cmp.line", model=original, test=test_name(suite, lang), a=frac(a.correct, a.evaluated),
                             b=frac(b.correct, b.evaluated),
                             changes=", ".join(changes) if changes else t("d.cmp.same")))
        return out

    def _show_breakdown(self) -> None:
        sel = self.rtable.selection()
        run = self.result_by_row.get(sel[0]) if sel else None
        if run is None:
            return
        s = run["summary"]
        out = [f"{run['model']} · {test_name(run['suite'], run['lang'])}", "", t("q.by_category")]
        for cat, v in s.by_category.items():
            out.append(f"  {t('q.cat.' + cat):<24} {frac(v.correct, v.evaluated):>6}"
                       + (f"   ({t('q.not_evaluated', n=v.total - v.evaluated)})" if v.partial else ""))
        out += ["", t("q.by_level")]
        for lvl in ("easy", "medium", "hard"):
            v = s.by_level.get(lvl)
            if v:
                out.append(f"  {t('q.level.' + lvl):<24} {frac(v.correct, v.evaluated):>6}")
        out += ["", t("q.format_line", n=s.format_noncompliant), t("q.manual_line", n=s.manual), "", t("q.note")]
        self._set(self.breakdown, "\n".join(out))

    def _show_task(self) -> None:
        sel = self.ttable.selection()
        row = self.task_rows.get(sel[0]) if sel else None
        if row is None:
            return
        model, suite, lang, r, run_id, _ = row
        w = self.task_text
        w.configure(state="normal")
        w.delete("1.0", "end")
        w.insert("end", f"{model} · {test_name(suite, lang)} · {r.task_id} · {t('q.level.' + r.level)} · "
                        f"{t('q.cat.' + r.category)}\n", ("head",))
        w.insert("end", t("bench.preview.prompt") + "\n", ("head",))
        w.insert("end", r.prompt + "\n", ("prompt",))
        if r.translation:
            w.insert("end", t("q.translation") + " " + r.translation + "\n", ("tr",))
        w.insert("end", "\n" + t("q.expected") + " " + r.expected + "\n", ("head",))
        for i, a in enumerate(r.attempts, 1):
            head = t("bench.preview.answer") + (f" ({t('q.attempt', n=i, of=len(r.attempts))})" if len(r.attempts) > 1 else "")
            w.insert("end", "\n" + head + "\n", ("head",))
            w.insert("end", (a.answer or "").strip() + "\n")
            w.insert("end", f"{MARK[a.correct]} {t('q.reason.' + a.reason)}\n", ("ok" if a.correct else "bad",))
        w.configure(state="disabled")
        can = r.correct is None and r.reason in ("not_run", "manual") and run_id is not None and not self.running
        for b in (self.pass_button, self.fail_button):
            b.state(["!disabled"] if can else ["disabled"])

    def grade(self, passed: bool) -> None:
        """Ocena ręczna – tylko zadania nieocenione automatycznie (kod bez piaskownicy, „oceniam sam”)."""
        from aibobr.quality.runner import set_manual
        sel = self.ttable.selection()
        row = self.task_rows.get(sel[0]) if sel else None
        db = getattr(self.c, "bench_db", None)
        if row is None or db is None or row[4] is None:
            return
        model, suite, lang, r, run_id, task_no = row
        task_row = next(x["id"] for x in db.quality_tasks(run_id) if x["task_no"] == task_no)
        values = set_manual(db, run_id, task_row, passed)
        r.correct, r.manual, r.reason = passed, True, "manual"
        self.ttable.item(sel[0], values=(model, test_name(suite, lang), r.task_id, t(f"q.level.{r.level}"),
                                          self._category(r.category), MARK[passed], t("q.reason.manual"),
                                          self._fragility(model, suite, lang, r.task_id)),
                         tags=("ok" if passed else "bad",))
        run = next(x for x in self.q_results if x.get("id") == run_id)
        tasks = [x[3] for x in self.task_rows.values() if x[4] == run_id]
        run["summary"] = score.summarize([vars(x) for x in tasks])
        run.update(values)
        for iid, rr in self.result_by_row.items():
            if rr is run:
                self.rtable.delete(iid)
        self.result_by_row = {k: v for k, v in self.result_by_row.items() if v is not run}
        self.q_results.remove(run)
        self._add_result(run)
        self._show_task()

    def _done(self, result: dict) -> None:
        self.running = False
        if not self.winfo_exists():
            return
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        elapsed = time.monotonic() - self.started
        took = t("bench.took", min=int(elapsed // 60), sec=int(elapsed % 60), est=max(1, round(self.estimated / 60)))
        stopped = result.get("status") == "stopped"
        self.stage.configure(text=t("bench.stopped_note") if stopped else t("q.saved"))
        self.times.configure(text=took)
        if not stopped:
            self.bar.configure(value=100)
            self.percent.configure(text="100%")
        self.log("END", took)
        self.results_tabs.select(0)
        self.refresh_history()

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

    # ------------------------------------------------------------------ historia
    def _tab_changed(self) -> None:
        if self.results_tabs.select() == str(self.history_tab):
            self.refresh_history()

    def refresh_history(self) -> None:
        import json
        db = getattr(self.c, "bench_db", None)
        self.htable.delete(*self.htable.get_children())
        self.hist_rows.clear()
        if db is None:
            return
        for s in db.quality_sessions():
            tests = ", ".join(test_name(*x.split("/")) for x in json.loads(s["tests_json"] or "[]"))
            iid = self.htable.insert("", "end", values=((s["started_at"] or "")[:16].replace("T", " "),
                                                         ", ".join(json.loads(s["models_json"] or "[]")), tests,
                                                         s["attempts"], t(f"q.status.{s['status']}")))
            self.hist_rows[iid] = s["id"]

    def _hist_selected(self) -> Optional[str]:
        sel = self.htable.selection()
        return self.hist_rows.get(sel[0]) if sel else None

    def history_details(self) -> None:
        sid = self._hist_selected()
        db = getattr(self.c, "bench_db", None)
        if sid is None or db is None:
            return
        import json
        lines = []
        for run in db.quality_runs(sid):
            lines.append(f"{run['model']} · {test_name(run['suite'], run['lang'])} (v{run['suite_version']}): "
                         f"{frac(run['correct'] or 0, run['evaluated'] or 0)}{pct(run['correct'] or 0, run['evaluated'] or 0)}"
                         + (f" · {t('q.coverage', evaluated=run['evaluated'], total=run['total'])}"
                            if (run['evaluated'] or 0) < (run['total'] or 0) else ""))
            cats = json.loads(run["by_category_json"] or "{}")
            lines.append("   " + " · ".join(f"{t('q.cat.' + k)} {frac(v['correct'], v['evaluated'])}" for k, v in cats.items()))
            for task in db.quality_tasks(run["id"]):
                mark = MARK[None if task["correct"] is None else bool(task["correct"])]
                lines.append(f"   {task['task_id']:>4} {mark} {t('q.reason.' + (task['reason'] or 'wrong'))}"
                             f" | {(task['answer'] or '').strip()[:70]!r}")
            lines.append("")
        win = tk.Toplevel(self)
        win.title(t("q.history_title"))
        win.geometry("860x600")
        text = self._text(win)
        text.pack(fill="both", expand=True)
        self._set(text, "\n".join(lines) + "\n" + t("q.note"))
        self.details_window = win

    def export_history(self) -> None:
        from aibobr.bench import export
        sid = self._hist_selected() or self.q_session
        db = getattr(self.c, "bench_db", None)
        if sid is None or db is None:
            messagebox.showinfo(t("speed.export"), t("hist.pick"), parent=self)
            return
        parent = filedialog.askdirectory(parent=self, title=t("speed.export"), mustexist=True)
        if parent:
            folder = export.export_quality_session(db, sid, parent)
            self.on_status(t("speed.exported", path=str(folder)))

    def delete_history(self) -> None:
        sid = self._hist_selected()
        db = getattr(self.c, "bench_db", None)
        if sid is None or db is None:
            messagebox.showinfo(t("hist.delete"), t("hist.pick"), parent=self)
            return
        if not self.confirm(t("hist.delete"), t("q.delete_confirm")):
            return
        db.delete_quality_session(sid)
        self.on_status(t("hist.deleted"))
        self.refresh_history()
