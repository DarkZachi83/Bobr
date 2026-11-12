"""Zakładka „Diagnostyka” (0.12, decyzje autora 12.11.2026): co model naprawdę dostał i czy prompt jest kruchy.

Osobna główna zakładka między Jakością a Statystyką (Jakość jest już gęsta), też Ustawienia → Diagnostyka.
Lewa strona – ta sama lista modeli co w Benchmarku (dziedziczona). Prawa – trzy karty:
- Token początku: czy model w Ollamie dostaje token początku (BOS); wynik zapamiętany na wersję pliku,
  „Sprawdź ponownie” mierzy od nowa;
- Kruchość: zadania testu jakości (standardowego albo własnego) z logprobs, osobny przebieg, progi
  ustawiane tutaj i zapamiętywane;
- Historia: przebiegi kruchości z wynikami, usuwanie.
Logika – aibobr/diagnostics, zapis – BenchDb (schemat 3).
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

from aibobr.gui.widgets.bench_panel import LOG_COLORS, BenchmarkPanel
from aibobr.i18n import t

BCOLS = ("model", "status", "expects", "template", "engine", "effective", "checked")
BWIDTHS = {"model": 200, "status": 185, "expects": 75, "template": 95, "engine": 100, "effective": 100,
           "checked": 120}
FCOLS = ("task", "label", "margin", "first", "after")
FWIDTHS = {"task": 55, "label": 90, "margin": 90, "first": 300, "after": 140}
HCOLS = ("date", "model", "test", "result", "thresholds", "bos")
HWIDTHS = {"date": 130, "model": 220, "test": 170, "result": 200, "thresholds": 110, "bos": 110}
LABEL_TAG = {"stable": "ok", "fragile": "warn", "tie": "bad", "unsupported": "none", "not_run": "none"}
BOS_TAG = {"ok": "ok", "not_used": "ok", "missing": "bad", "double": "bad", "unknown": "none"}


def yes_no(value) -> str:
    return "–" if value is None else t("d.yes") if value else t("d.no")


def count_text(value) -> str:
    return "–" if value is None else str(value)


def tok(token) -> str:
    return "‹" + t("d.end") + "›" if token == "" else repr(token)


def pair_text(pair) -> str:
    if not pair:
        return "–"
    first, second = pair[0], pair[1] if len(pair) > 1 else None
    out = f"{tok(first[0])} {first[1]:.3f}"
    return out + (f"  |  {tok(second[0])} {second[1]:.3f}" if second else "")


class DiagnosticsPanel(BenchmarkPanel):
    def _build_bench(self) -> None:
        from aibobr.gui.widgets.quality_panel import test_name
        self.test_name = test_name
        self.bos_rows: dict[str, str] = {}             # model → wiersz tabeli
        self.f_rows: dict[str, dict] = {}              # wiersz → wynik zadania
        self.h_rows: dict[str, str] = {}               # wiersz historii → id sesji
        self.mode = ""
        top = ttk.Frame(self.right, style="Panel.TFrame", padding=12)
        top.pack(fill="x")
        ttk.Label(top, text=t("d.title"), style="Title.TLabel").pack(anchor="w")
        what = ttk.Label(top, text=t("d.what"), style="Muted.TLabel", wraplength=640, justify="left")
        what.pack(anchor="w", fill="x", pady=(2, 6))
        what.bind("<Configure>", lambda e: what.configure(wraplength=max(300, e.width)))
        self.plan = ttk.Label(top, text="", style="Panel.TLabel", wraplength=640, justify="left")
        self.plan.pack(anchor="w", fill="x")
        prog = ttk.Frame(top, style="Inner.TFrame")
        prog.pack(fill="x", pady=(6, 0))
        self.stop_button = ttk.Button(prog, text="■ " + t("bench.stop"), command=self.stop.set)
        self.stop_button.pack(side="right")
        self.stop_button.state(["disabled"])
        self.stage = ttk.Label(prog, text="", style="Note.TLabel", wraplength=500, justify="left")
        self.stage.pack(side="left", fill="x", expand=True)
        self.bar = ttk.Progressbar(top, mode="determinate", maximum=100)
        self.bar.pack(fill="x", pady=(4, 0))

        self.tabs = ttk.Notebook(self.right)
        self.tabs.pack(fill="both", expand=True, pady=(8, 0))
        self._build_bos()
        self._build_fragility()
        self._build_history()
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed())
        self.refresh_bos()

    # ------------------------------------------------------------------ wspólne
    def _tree(self, parent, cols, widths, prefix, height=8, stretch=("model",)):
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
        for tag, color in (("ok", "#53d6a0"), ("warn", "#f5c65f"), ("bad", "#ff8282"), ("none", "#9bacc0")):
            tree.tag_configure(tag, foreground=color)
        return tree

    def _text(self, parent, height=8):
        text = tk.Text(parent, wrap="word", font="TkFixedFont", height=height, width=40, background="#11171d",
                       foreground="#cfd8e3", relief="flat", padx=10, pady=8, state="disabled")
        for tag, color in {"head": "#66c9fa", "ok": "#53d6a0", "warn": "#f5c65f", "bad": "#ff8282",
                           "muted": "#9bacc0", **LOG_COLORS}.items():
            text.tag_configure(tag, foreground=color)
        return text

    def _set(self, widget, parts) -> None:
        """parts: tekst albo lista (tekst, znacznik)."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for text, tag in ([(parts, "")] if isinstance(parts, str) else parts):
            widget.insert("end", text, (tag,) if tag else ())
        widget.configure(state="disabled")

    def _busy_buttons(self, busy: bool) -> None:
        state = ["disabled"] if busy else ["!disabled"]
        for b in (self.bos_button, self.bos_again, self.bos_fix, self.f_button):
            b.state(state)
        self.stop_button.state(["!disabled"] if busy else ["disabled"])
        if not busy:
            self._update_plan()

    # ------------------------------------------------------------------ token początku
    def _build_bos(self) -> None:
        tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=8)
        self.tabs.add(tab, text=t("d.tab.bos"))
        intro = ttk.Label(tab, text=t("d.bos.what"), style="Muted.TLabel", wraplength=700, justify="left")
        intro.pack(anchor="w", fill="x", pady=(0, 6))
        intro.bind("<Configure>", lambda e: intro.configure(wraplength=max(300, e.width)))
        bar = ttk.Frame(tab, style="Inner.TFrame")
        bar.pack(fill="x", pady=(0, 6))
        self.bos_button = ttk.Button(bar, text="▶ " + t("d.bos.check"), style="Accent.TButton",
                                     command=lambda: self.check_bos(False))
        self.bos_button.pack(side="left")
        self.bos_again = ttk.Button(bar, text=t("d.bos.again"), command=lambda: self.check_bos(True))
        self.bos_again.pack(side="left", padx=6)
        self.bos_fix = ttk.Button(bar, text=t("d.bos.fix"), command=self.fix_bos)
        self.bos_fix.pack(side="left")
        pane = ttk.Panedwindow(tab, orient="vertical")
        pane.pack(fill="both", expand=True)
        top = ttk.Frame(pane, style="Inner.TFrame")
        pane.add(top, weight=2)
        self.btable = self._tree(top, BCOLS, BWIDTHS, "d.bcol", height=7)
        self.btable.bind("<<TreeviewSelect>>", lambda e: self._show_bos())
        self.bos_text = self._text(pane, height=7)
        pane.add(self.bos_text, weight=1)
        self.bos_data: dict[str, object] = {}

    def refresh_bos(self) -> None:
        db = self.c.bench_db
        if db is None:
            return
        from aibobr.diagnostics.bos import BosCheck
        for row in db.bos_checks():
            data = {"model": row["model"], "digest": row["digest"]}
            for attr, col in db.BOS_FIELDS.items():
                v = row[col]
                data[attr] = bool(v) if attr in ("expects", "add_bos", "gguf_template_has", "template_has") \
                    and v is not None else v
            self._put_bos(BosCheck.from_dict(data), row["checked_at"])

    def _put_bos(self, check, when: Optional[str]) -> None:
        values = (check.model, t(f"d.bos.{check.status}"), yes_no(check.expects), yes_no(check.template_has),
                  count_text(check.without), count_text(check.effective), (when or "")[:16].replace("T", " "))
        iid = self.bos_rows.get(check.model)
        if iid and self.btable.exists(iid):
            self.btable.item(iid, values=values, tags=(BOS_TAG.get(check.status, "none"),))
        else:
            iid = self.btable.insert("", "end", values=values, tags=(BOS_TAG.get(check.status, "none"),))
            self.bos_rows[check.model] = iid
        self.bos_data[iid] = check

    def _show_bos(self) -> None:
        sel = self.btable.selection()
        check = self.bos_data.get(sel[0]) if sel else None
        if check is None:
            return
        source = {"model_info": t("d.src.ollama"), "file": t("d.src.file"), None: t("d.src.none")}.get(
            check.gguf_template_source, "?")
        tag = BOS_TAG.get(check.status, "none")
        parts = [(f"{check.model}\n", "head"), (t(f"d.bos.{check.status}") + "\n", tag),
                 (t(f"d.bos.meaning.{check.status}") + "\n\n", "")]
        if check.reason and check.reason != "measured":
            parts.append((t("d.reason") + ": " + t(f"d.reason.{check.reason}") + "\n\n", "muted"))
        parts.append((t("d.bos.details", add_bos=yes_no(check.add_bos), gguf=yes_no(check.gguf_template_has),
                        source=source, token=check.bos_text or "–", template=yes_no(check.template_has),
                        without=count_text(check.without), with_text=count_text(check.with_text),
                        effective=count_text(check.effective)), "muted"))
        self._set(self.bos_text, parts)

    def fix_bos(self) -> None:
        """Poprawiona wersja zaznaczonego modelu (brak albo podwójny token) – za zgodą, oryginał nietknięty."""
        from aibobr.diagnostics.bos import variant_name
        from aibobr.sources.factory import OLLAMA_PREFIX
        sel = self.btable.selection()
        check = self.bos_data.get(sel[0]) if sel else None
        if check is None or self.running:
            return
        if not check.warning:
            self.stage.configure(text=t("d.bos.nothing_to_fix"))
            return
        name = variant_name(check.model)
        if not self.confirm(t("d.bos.fix"), t("d.bos.fix_confirm", name=name)):
            return
        self.running, self.mode = True, "fix"
        self.total, self.done_count, self.cached_count = 1, 0, 0
        self._busy_buttons(True)
        self.stage.configure(text=t("d.gate.creating", names=name))

        def done(made):
            vt, vcheck = made
            if vcheck is not None:
                self.stage.configure(text=t("d.bos.fixed", name=self._model_name(vt), status=t(f"d.bos.{vcheck.status}")))
        self.worker.run_with_progress("diag_fix", lambda report: self.c.create_bos_variant(
            OLLAMA_PREFIX + check.model, check, report), self._progress,
            lambda made: (self._done(None), done(made)), self._failed)

    def check_bos(self, force: bool) -> None:
        targets = self.targets()
        if not targets or self.running:
            return
        self.running, self.mode = True, "bos"
        self.stop.clear()
        self.total, self.done_count, self.cached_count = len(targets), 0, 0
        self._busy_buttons(True)
        self.bar.configure(value=0)
        self.tabs.select(0)
        self.worker.run_with_progress(
            "diag_bos", lambda report: self.c.check_bos(targets, report, self.stop, force=force),
            self._progress, self._done, self._failed)

    # ------------------------------------------------------------------ kruchość
    def _build_fragility(self) -> None:
        tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=8)
        self.tabs.add(tab, text=t("d.tab.fragility"))
        intro = ttk.Label(tab, text=t("d.f.what"), style="Muted.TLabel", wraplength=700, justify="left")
        intro.pack(anchor="w", fill="x", pady=(0, 6))
        intro.bind("<Configure>", lambda e: intro.configure(wraplength=max(300, e.width)))
        line = ttk.Frame(tab, style="Inner.TFrame")
        line.pack(fill="x", pady=(0, 6))
        ttk.Label(line, text=t("d.f.test"), style="Panel.TLabel").pack(side="left")
        self.f_test = ttk.Combobox(line, state="readonly", width=34)
        self.f_test.pack(side="left", padx=(6, 14))
        self.f_test.bind("<<ComboboxSelected>>", lambda e: self._update_plan())
        th = self.c.fragility_thresholds()
        ttk.Label(line, text=t("d.f.fragile"), style="Panel.TLabel").pack(side="left")
        self.f_fragile = tk.StringVar(value=f"{th.fragile:g}")
        ttk.Entry(line, textvariable=self.f_fragile, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(line, text=t("d.f.tie"), style="Panel.TLabel").pack(side="left")
        self.f_tie = tk.StringVar(value=f"{th.tie:g}")
        ttk.Entry(line, textvariable=self.f_tie, width=6).pack(side="left", padx=4)
        line = ttk.Frame(tab, style="Inner.TFrame")
        line.pack(fill="x", pady=(0, 6))
        self.f_button = ttk.Button(line, text="▶ " + t("d.f.run"), style="Accent.TButton", command=self.run_fragility)
        self.f_button.pack(side="left")
        self.f_note = ttk.Label(line, text="", style="Note.TLabel")
        self.f_note.pack(side="left", padx=10)
        pane = ttk.Panedwindow(tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, style="Inner.TFrame")
        pane.add(left, weight=3)
        self.ftable = self._tree(left, FCOLS, FWIDTHS, "d.fcol", height=10, stretch=("first",))
        self.ftable.bind("<<TreeviewSelect>>", lambda e: self._show_task())
        self.f_text = self._text(pane, height=10)
        pane.add(self.f_text, weight=2)
        self._fill_tests()

    def _fill_tests(self) -> None:
        from aibobr.quality import suites
        from aibobr.gui.widgets.quality_panel import CUSTOM_NAMES
        from aibobr.i18n import current_language
        keys = []
        for data in suites.standard():
            for lang in data["suite_info"]["languages"]:
                keys.append((data["suite_info"]["id"], lang))
        try:
            custom = self.c.custom_suites.valid()
        except OSError:
            custom = []
        for data in custom:
            info = data["suite_info"]
            CUSTOM_NAMES[info["id"]] = info["name"].get(current_language()) or next(iter(info["name"].values()))
            keys += [(info["id"], lang) for lang in info["languages"]]
        self.test_keys = keys
        current = self.f_test.get()
        names = [self.test_name(*k) for k in keys]
        self.f_test.configure(values=names)
        self.f_test.set(current if current in names else names[0] if names else "")

    def selected_test(self) -> Optional[tuple[str, str]]:
        names = [self.test_name(*k) for k in self.test_keys]
        name = self.f_test.get()
        return self.test_keys[names.index(name)] if name in names else None

    def thresholds(self):
        """Progi z pól; błędne – komunikat i None (nic nie zapisujemy)."""
        try:
            return self.c.set_fragility_thresholds(float(self.f_fragile.get().replace(",", ".")),
                                                   float(self.f_tie.get().replace(",", ".")))
        except ValueError:
            self.f_note.configure(text=t("d.f.bad_thresholds"))
            return None

    def run_fragility(self) -> None:
        targets, test = self.targets(), self.selected_test()
        if not targets or not test or self.running:
            return
        th = self.thresholds()
        if th is None:
            return
        self.f_note.configure(text=t("d.f.saved", fragile=f"{th.fragile:g}", tie=f"{th.tie:g}"))
        self.running, self.mode = True, "fragility"
        self.stop.clear()
        self.ftable.delete(*self.ftable.get_children())
        self.f_rows.clear()
        self._set(self.f_text, "")
        self.total, self.done_count = len(targets) * self._task_count(test), 0
        self._busy_buttons(True)
        self.bar.configure(value=0)
        self.tabs.select(1)

        def work(report):
            out = []
            for target in targets:
                if self.stop.is_set():
                    break
                out.append(self.c.run_fragility(target, test, report, self.stop, thresholds=th))
            return out
        self.worker.run_with_progress("diag_fragility", work, self._progress, self._done, self._failed)

    def _task_count(self, test) -> int:
        from aibobr.quality import suites
        suite_id, lang = test
        try:
            data = suites.load(suite_id) if suite_id in suites.STANDARD else self.c.custom_suites.load(suite_id)
        except Exception:
            return 15
        return sum(1 for x in data["tasks"] if lang in x.get("prompt", {}))

    def _add_task(self, model: str, task: dict) -> None:
        label = task["label"]
        after = (t("d.f.after", fragile=task["after_fragile"], tie=task["after_tie"])
                 if task.get("after_fragile") else "")
        values = (task["task_id"], t(f"d.label.{label}"),
                  "–" if task.get("min_margin") is None else f"{task['min_margin']:.3f}",
                  pair_text(task.get("first_pair")), after)
        iid = self.ftable.insert("", "end", values=values, tags=(LABEL_TAG.get(label, "none"),))
        self.f_rows[iid] = {**task, "model": model}

    def _show_task(self) -> None:
        sel = self.ftable.selection()
        task = self.f_rows.get(sel[0]) if sel else None
        if task is None:
            return
        label = task["label"]
        parts = [(f"{task['model']} · {task['task_id']}\n", "head"),
                 (t(f"d.label.{label}") + "\n", LABEL_TAG.get(label, "none")),
                 (t(f"d.label.meaning.{label}") + "\n\n", "")]
        if task.get("reason"):
            parts.append((t("d.reason") + ": " + task["reason"] + "\n\n", "muted"))
        if task.get("steps"):
            scope = (t("d.f.scope.answer", n=task["answer_steps"], total=task["steps"]) if task.get("scope") == "answer"
                     else t("d.f.scope.whole", total=task["steps"]))
            parts.append((scope + "\n", "muted"))
            parts.append((t("d.f.narrowest", pos=task["min_pos"], margin=f"{task['min_margin']:.3f}",
                            pair=pair_text(task.get("min_pair"))) + "\n", ""))
            if task.get("first_pos") is not None:
                parts.append((t("d.f.first", pos=task["first_pos"], context=task.get("first_context") or "",
                                pair=pair_text(task.get("first_pair")), fragile=task["fragile_steps"],
                                tie=task["tie_steps"]) + "\n", "warn"))
            if task.get("after_fragile"):
                parts.append((t("d.f.after_long", fragile=task["after_fragile"], tie=task["after_tie"]) + "\n",
                              "muted"))
        parts.append(("\n" + t("d.f.answer") + "\n", "head"))
        parts.append((task.get("answer") or "", ""))
        self._set(self.f_text, parts)

    # ------------------------------------------------------------------ postęp
    def _progress(self, e: dict) -> None:
        kind = e.get("event")
        if kind == "bos_start":
            self.stage.configure(text=t("d.bos.running", model=e["model"]))
        elif kind == "bos":
            self.done_count += 1
            self.cached_count += 1 if e.get("cached") else 0
            self._put_bos(e["check"], time.strftime("%Y-%m-%dT%H:%M"))
            self.stage.configure(text=t("d.bos.done_one", model=e["model"], status=t(f"d.bos.{e['check'].status}"))
                                 + (" " + t("d.bos.cached") if e.get("cached") else ""))
        elif kind == "f_task":
            self.stage.configure(text=t("d.f.running", model=e["model"], task=e["task_id"], n=e["n"], total=e["total"]))
        elif kind == "f_task_done":
            self.done_count += 1
            self._add_task(e["model"], e["task"])
            if e["task"]["label"] == "unsupported":
                self.f_note.configure(text=t("d.f.unsupported"))
        elif kind == "f_done":
            c = e["counts"]
            self.stage.configure(text=t("d.f.summary", model=e["model"], stable=c["stable"], fragile=c["fragile"],
                                        tie=c["tie"]))
        if self.total:
            self.bar.configure(value=min(100, 100 * self.done_count / self.total))

    def _done(self, result) -> None:
        self.running = False
        if not self.winfo_exists():
            return
        self._busy_buttons(False)
        self.bar.configure(value=100)
        if self.mode == "bos":
            self.stage.configure(text=t("d.bos.finished", n=len(result or []), cached=self.cached_count))
        self.refresh_history()

    def _failed(self, e: BaseException) -> None:
        self.running = False
        if not self.winfo_exists():
            return
        self._busy_buttons(False)
        from aibobr.sources.base import SourceError
        self.stage.configure(text=t("bench.no_ollama") if isinstance(e, SourceError)
                             else t("gui.unexpected", error=repr(e)))

    # ------------------------------------------------------------------ plan
    def _targets_changed(self) -> None:
        self._update_plan()

    def _update_plan(self) -> None:
        if self.running or not hasattr(self, "f_button"):
            return
        targets = self.targets()
        self.plan.configure(text=t("bench.none_checked") if not targets else t("d.plan", models=len(targets)))
        for b in (self.bos_button, self.bos_again, self.f_button):
            b.state(["!disabled"] if targets else ["disabled"])
        self.bos_fix.state(["!disabled"])                  # działa na zaznaczony wiersz tabeli, nie na listę modeli

    def start(self) -> None:                          # przycisk startu benchmarku – tu nieużywany
        pass

    # ------------------------------------------------------------------ historia
    def _build_history(self) -> None:
        tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=8)
        self.tabs.add(tab, text=t("hist.tab"))
        bar = ttk.Frame(tab, style="Inner.TFrame")
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text=t("hist.details"), command=self.history_details).pack(side="left")
        ttk.Button(bar, text=t("hist.delete"), command=self.delete_history).pack(side="left", padx=6)
        self.htable = self._tree(tab, HCOLS, HWIDTHS, "d.hcol", height=10, stretch=("model", "test"))
        self.htable.bind("<Double-Button-1>", lambda e: self.history_details())

    def _tab_changed(self) -> None:
        if self.tabs.index("current") == 2:
            self.refresh_history()
        elif self.tabs.index("current") == 1:
            self._fill_tests()

    def refresh_history(self) -> None:
        db = self.c.bench_db
        self.htable.delete(*self.htable.get_children())
        self.h_rows.clear()
        if db is None:
            return
        for s in db.diag_sessions():
            result = (t("d.h.result", stable=s["stable"], fragile=s["fragile"], tie=s["tie"])
                      if s["status"] == "finished" else t(f"d.h.status.{s['status']}"))
            values = (s["started_at"][:16].replace("T", " "), s["model"], self.test_name(s["suite"], s["lang"])
                      + (f" v{s['suite_version']}" if s["suite_version"] else ""), result,
                      f"{s['fragile_threshold']:g} / {s['tie_threshold']:g}",
                      t(f"d.bos.{s['bos_status']}") if s["bos_status"] else "–")
            iid = self.htable.insert("", "end", values=values)
            self.h_rows[iid] = s["id"]

    def _hist_selected(self) -> Optional[str]:
        sel = self.htable.selection()
        return self.h_rows.get(sel[0]) if sel else None

    def history_details(self) -> None:
        sid = self._hist_selected()
        db = self.c.bench_db
        if not sid or db is None:
            return
        session = next((s for s in db.diag_sessions() if s["id"] == sid), None)
        if session is not None:
            self._fill_tests()
            name = self.test_name(session["suite"], session["lang"])
            if name in self.f_test.cget("values"):
                self.f_test.set(name)
        self.ftable.delete(*self.ftable.get_children())
        self.f_rows.clear()
        for row in db.diag_tasks(sid):
            task = dict(row)
            task["min_pair"], task["first_pair"] = row["min_pair_json"], row["first_pair_json"]
            self._add_task(session["model"] if session else "", task)
        self._set(self.f_text, "")
        self.tabs.select(1)

    def delete_history(self) -> None:
        sid = self._hist_selected()
        if not sid or not self.confirm(t("hist.delete"), t("d.h.delete_confirm")):
            return
        self.c.bench_db.delete_diag_session(sid)
        self.refresh_history()
