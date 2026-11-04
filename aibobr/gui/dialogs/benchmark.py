"""Okno benchmarku – „Testy Zacharskiego”, wydajność (etap 12a, SPEC §5c).

Przed startem AI Bóbr liczy werdykt dla każdej pary model × kontekst: czerwone pomija (nie
zmieszczą się – test zawiesiłby komputer), przy reszcie pokazuje przewidywanie i szacunek czasu.
Modele mierzymy PO KOLEI przez Ollamę – nigdy kilka naraz. W trakcie: postęp i „Zatrzymaj”;
na końcu: tabela wyników zapisanych w benchmarks.sqlite i najszybszy model na tym komputerze.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from aibobr.bench.runner import estimate_seconds
from aibobr.gui.theme import COLORS, VERDICT_COLORS
from aibobr.i18n import fmt_context, t

CONTEXTS = (1024, 8192, 32768)
DEFAULT_ON = {1024, 8192}                  # 32k na życzenie – na słabszym sprzęcie trwa długo
COLUMNS = ("model", "context", "predicted", "pp", "ttft", "tg", "gpu")
WIDTHS = {"model": 180, "context": 80, "predicted": 120, "pp": 110, "ttft": 90, "tg": 150, "gpu": 80}


def _num(x: Optional[float], digits: int = 1) -> str:
    return "–" if x is None else f"{x:.{digits}f}".replace(".", t("bench.decimal"))


class BenchmarkDialog(tk.Toplevel):
    def __init__(self, master, controller, worker, targets: list[str], hw, settings,
                 confirm: Callable[[str, str], bool] = None):
        super().__init__(master)
        self.c, self.worker, self.targets, self.hw, self.settings = controller, worker, targets, hw, settings
        self.confirm = confirm or (lambda title, text: messagebox.askokcancel(title, text, parent=self))
        self.title(t("bench.title"))
        self.transient(master)
        self.minsize(820, 480)
        self.stop = threading.Event()
        self.predicted: dict = {}
        self._minutes = 0
        self.results: list = []
        self.running = False
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        names = ", ".join(t_[len("ollama:"):] if t_.startswith("ollama:") else t_ for t_ in targets)
        ttk.Label(body, text=t("bench.models", n=len(targets), names=names), wraplength=780,
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(body, text=t("bench.what"), foreground=COLORS["muted"], wraplength=780).pack(anchor="w", pady=(2, 6))

        opts = ttk.Frame(body)
        opts.pack(fill="x")
        ttk.Label(opts, text=t("bench.contexts")).pack(side="left")
        self.ctx_vars = {}
        for ctx in CONTEXTS:
            var = tk.BooleanVar(value=ctx in DEFAULT_ON)
            ttk.Checkbutton(opts, text=fmt_context(ctx), variable=var, command=self._update_plan).pack(side="left", padx=6)
            self.ctx_vars[ctx] = var
        ttk.Label(opts, text="   " + t("bench.suites"), foreground=COLORS["muted"]).pack(side="left")

        self.plan = ttk.Label(body, text="", wraplength=780)
        self.plan.pack(anchor="w", pady=(8, 0))
        self.warning = ttk.Label(body, text="", foreground=COLORS["warn"], wraplength=780)
        self.warning.pack(anchor="w")

        frame = ttk.Frame(body)
        frame.pack(fill="both", expand=True, pady=(8, 0))
        self.tree = ttk.Treeview(frame, columns=COLUMNS, show="headings", height=8)
        for col in COLUMNS:
            self.tree.heading(col, text=t(f"bench.col.{col}"))
            self.tree.column(col, width=WIDTHS[col], anchor="w", stretch=col == "model")
        for level, color in VERDICT_COLORS.items():
            self.tree.tag_configure(level.value, foreground=color)
        self.tree.tag_configure("skip", foreground=COLORS["muted"])
        self.tree.pack(fill="both", expand=True)

        self.progress = ttk.Label(body, text="", foreground=COLORS["muted"], wraplength=780)
        self.progress.pack(anchor="w", pady=(6, 0))
        self.summary = ttk.Label(body, text="", wraplength=780, font=("TkDefaultFont", 10, "bold"))
        self.summary.pack(anchor="w")

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text=t("bench.close"), command=self.close).pack(side="right")
        self.stop_button = ttk.Button(buttons, text=t("bench.stop"), command=self.stop.set)
        self.stop_button.pack(side="right", padx=6)
        self.stop_button.state(["disabled"])
        self.start_button = ttk.Button(buttons, text=t("bench.start"), style="Accent.TButton", command=self.start)
        self.start_button.pack(side="right")
        self.start_button.state(["disabled"])
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.progress.configure(text=t("bench.predicting"))
        self.worker.run("bench_predict",
                        lambda: self.c.bench_predict(targets, CONTEXTS, hw, settings),
                        self._predicted, lambda e: self._predicted({}))

    # ------------------------------------------------------------------ plan
    def contexts(self) -> list[int]:
        return [c for c in CONTEXTS if self.ctx_vars[c].get()]

    def _predicted(self, predicted: dict) -> None:
        self.predicted = predicted
        self.progress.configure(text="")
        self.start_button.state(["!disabled"])
        self._update_plan()

    def _update_plan(self) -> None:
        if self.running:
            return
        self.tree.delete(*self.tree.get_children())
        jobs = [(tg, c) for tg in self.targets for c in self.contexts()]
        seconds, skipped = 0.0, 0
        for target, ctx in jobs:
            level = self.predicted.get((target, ctx))
            name = target[len("ollama:"):] if target.startswith("ollama:") else target
            if level == "red":
                skipped += 1
                values = (name, fmt_context(ctx), t("bench.pred.red"), t("bench.skip_red"), "", "", "")
                self.tree.insert("", "end", values=values, tags=("skip",))
                continue
            seconds += estimate_seconds(ctx, level, measured=self._measured(name, ctx))
            values = (name, fmt_context(ctx), t(f"bench.pred.{level}") if level else "?", "", "", "", "")
            self.tree.insert("", "end", values=values, tags=(level,) if level else ())
        minutes = max(1, round(seconds / 60))
        self._minutes = minutes
        text = t("bench.plan", jobs=len(jobs) - skipped, minutes=minutes)
        if skipped:
            text += " " + t("bench.plan_skipped", n=skipped)
        self.plan.configure(text=text)
        warnings = []
        if len(self.targets) > 1:
            warnings.append(t("bench.warn_many", minutes=minutes))
        from aibobr.app.controller import power_source
        if power_source() == "battery":
            warnings.append(t("bench.warn_battery"))
        self.warning.configure(text="\n".join("⚠ " + w for w in warnings))
        self.start_button.state(["!disabled"] if len(jobs) > skipped else ["disabled"])

    def _measured(self, model: str, ctx: int):
        """Prędkości z ostatniego pomiaru tego modelu na tym sprzęcie – lepszy szacunek czasu."""
        db = getattr(self.c, "bench_db", None)
        if db is None:
            return None
        try:
            rows = db.latest(model, self.c.hardware_text(self.hw))
        except Exception:
            return None
        row = next((r for r in rows if r["context"] == ctx), None) or (rows[-1] if rows else None)
        return (row["pp_tps"], row["tg_tps"]) if row else None

    # ------------------------------------------------------------------ pomiar
    def start(self) -> None:
        contexts = self.contexts()
        if not contexts or self.running:
            return
        if len(self.targets) > 1 and not self.confirm(t("bench.title"), t("bench.confirm_many")):
            return
        self.running = True
        self.started = time.monotonic()
        self.estimated_min = self._minutes
        self.stop.clear()
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.summary.configure(text="")
        self.worker.run_with_progress(
            "bench",
            lambda report: self.c.run_benchmark(self.targets, contexts, self.predicted, self.hw,
                                                report, self.stop, db=self.c.bench_db),
            self._progress, self._done, self._failed)

    def _progress(self, p: dict) -> None:
        step = p["step"]
        step = t("bench.step.warmup") if step == "warmup" else \
            t("bench.step.prompt", n=step.split()[1]) if step.startswith("prompt") else \
            t("bench.step.generate", n=step.split()[1])
        self.progress.configure(text=t("bench.progress", n=p["n"], total=p["total"], model=p["model"],
                                       context=fmt_context(p["context"]), step=step))

    def _done(self, results: list) -> None:
        self.running = False
        self.results = results
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        self.tree.delete(*self.tree.get_children())
        for r in results:
            pred = t(f"bench.pred.{r.predicted}") if r.predicted else "?"
            if r.skipped or r.stopped:
                note = t(r.skipped) if r.skipped else t("bench.stopped")
                self.tree.insert("", "end", values=(r.model, fmt_context(r.context), pred, note, "", "", ""),
                                 tags=("skip",))
                continue
            gpu = "–" if r.gpu_share is None else f"{r.gpu_share * 100:.0f}%"
            self.tree.insert("", "end", values=(
                r.model, fmt_context(r.context), pred, _num(r.pp_tps, 0), _num(r.ttft_ms, 0),
                _num(r.tg_tps), gpu), tags=(r.predicted,) if r.predicted else ())
        done = [r for r in results if r.tg_tps]
        self._power_warning(done)
        elapsed = time.monotonic() - getattr(self, "started", time.monotonic())
        took = t("bench.took", min=int(elapsed // 60), sec=int(elapsed % 60),
                 est=getattr(self, "estimated_min", 0))
        self.progress.configure(text=(t("bench.stopped_note") if any(r.stopped for r in results)
                                      else t("bench.saved")) + " " + took)
        if len({r.model for r in done}) >= 2:
            best = max(done, key=lambda r: r.tg_tps)
            self.summary.configure(text=t("bench.fastest", model=best.model, tps=_num(best.tg_tps),
                                          context=fmt_context(best.context)))

    def _power_warning(self, done: list) -> None:
        """Karta przez test „na ścianie” limitu mocy (pobór ≥ 90% limitu, limit ≤ 60% maksimum)."""
        gpu = next((g for g in self.hw.gpus if g.selected and g.power_capped), None)
        peaks = [r.power_draw_w for r in done if r.power_draw_w]
        if gpu is None or not peaks or max(peaks) < 0.9 * gpu.power_limit_w:
            return
        import sys
        key = "bench.power_capped_linux" if sys.platform.startswith("linux") else "bench.power_capped_other"
        text = "⚠ " + t(key, draw=f"{max(peaks):.0f}", limit=f"{gpu.power_limit_w:.0f}", max=f"{gpu.power_max_w:.0f}")
        self.warning.configure(text=text)

    def _failed(self, e: BaseException) -> None:
        self.running = False
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        from aibobr.sources.base import SourceError
        self.progress.configure(text=t("bench.no_ollama") if isinstance(e, SourceError)
                                else t("gui.unexpected", error=repr(e)))

    def close(self) -> None:
        self.stop.set()                    # pomiar w tle kończy się po bieżącym kroku
        self.destroy()
