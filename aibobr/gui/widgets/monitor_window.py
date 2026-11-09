"""Okno „Monitor” benchmarku 0.9 (decyzja autora: „coś w rodzaju naszego wewnętrznego btop”).

Otwiera się przy Starcie; zamknięcie tylko je chowa (test trwa), „Monitor: Przywróć” w dzienniku
pokazuje je znowu. Na górze faza i bieżące wartości, pod nimi cztery wykresy z ostatnich 2 minut
(procesor z paskiem wątków, temperatury, pamięć, karta), powód ograniczenia taktowania karty,
a na dole bieżące zadanie i odpowiedź modelu na żywo (dawne okno „Podgląd”).
Brak odczytu czujnika = przerwa na wykresie i „–”, nigdy 0.
"""
from __future__ import annotations

import datetime
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Optional

from aibobr.i18n import t

HISTORY = 120                       # próbek (1 na sekundę) – 2 minuty
BG, GRID, TEXT, MUTED = "#11171d", "#24303b", "#cfd8e3", "#7c90a4"
TAGS = {"head": "#66c9fa", "prompt": "#9bacc0", "answer": "#e9ecef", "task": "#f5c65f"}


def _fmt(value: Optional[float], digits: int = 0, unit: str = "") -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}".replace(".", t("bench.decimal")) + unit


def heat(pct: Optional[float]) -> str:
    """Kolor obciążenia wątku: szary (brak) → zielony → żółty → czerwony."""
    if pct is None:
        return "#3a4652"
    if pct < 30:
        return "#2f8f6a"
    if pct < 70:
        return "#c9a227"
    return "#d9534f"


class Chart(tk.Canvas):
    """Wykres linii z ostatnich HISTORY próbek. series: (klucz, kolor, etykieta); ymax: stała albo
    funkcja próbki (np. pojemność VRAM)."""

    def __init__(self, master, title: str, series, unit: str, ymax=100.0, ymin: float = 0.0,
                 digits: int = 0, cores: bool = False):
        super().__init__(master, height=120, background=BG, highlightthickness=0)
        self.title, self.series, self.unit = title, series, unit
        self.ymax, self.ymin, self.digits, self.cores = ymax, ymin, digits, cores
        self.data = {key: deque(maxlen=HISTORY) for key, _, _ in series}
        self.last_cores: list = []
        self.top = ymax if not callable(ymax) else None
        self.bind("<Configure>", lambda e: self.draw())

    def push(self, sample: dict) -> None:
        for key, _, _ in self.series:
            self.data[key].append(sample.get(key))
        if callable(self.ymax):
            value = self.ymax(sample)
            if value:
                self.top = value
        if self.cores:
            self.last_cores = sample.get("cpu_cores_json") or []

    def _y(self, value: float, y0: float, y1: float) -> float:
        top = self.top or max([v for d in self.data.values() for v in d if v is not None] + [1.0])
        top = max(top, self.ymin + 1e-9)
        share = (value - self.ymin) / (top - self.ymin)
        return y1 - max(0.0, min(1.0, share)) * (y1 - y0)

    def draw(self) -> None:
        self.delete("all")
        w, h = max(self.winfo_width(), 200), max(self.winfo_height(), 80)
        strip = 10 if self.cores and self.last_cores else 0
        x0, y0, x1, y1 = 4, 20, w - 4, h - 4 - strip - (2 if strip else 0)
        for frac in (0.0, 0.5, 1.0):
            y = y0 + (y1 - y0) * frac
            self.create_line(x0, y, x1, y, fill=GRID)
        legend = [self.title]
        for key, color, label in self.series:
            values = list(self.data[key])
            step = (x1 - x0) / max(1, HISTORY - 1)
            start = HISTORY - len(values)
            run: list[float] = []
            for i, v in enumerate(values):                       # brak odczytu = przerwa w linii
                if v is None:
                    if len(run) >= 4:
                        self.create_line(*run, fill=color, width=2)
                    run = []
                    continue
                run += [x0 + (start + i) * step, self._y(v, y0, y1)]
            if len(run) >= 4:
                self.create_line(*run, fill=color, width=2)
            elif len(run) == 2:
                self.create_oval(run[0] - 1, run[1] - 1, run[0] + 1, run[1] + 1, fill=color, outline=color)
            last = values[-1] if values else None
            legend.append((f"{label} {_fmt(last, self.digits, self.unit)}", color))
        item = self.create_text(x0 + 2, 3, anchor="nw", text=legend[0], fill=TEXT, font=("TkDefaultFont", 9, "bold"))
        for text, color in legend[1:]:                      # kolejne opisy za poprzednim (rzeczywista szerokość)
            x = (self.bbox(item) or (0, 0, x0, 0))[2] + 12
            item = self.create_text(x, 3, anchor="nw", text=text, fill=color, font=("TkDefaultFont", 9))
        if strip:
            n = len(self.last_cores)
            cw = (x1 - x0) / n
            for i, pct in enumerate(self.last_cores):
                self.create_rectangle(x0 + i * cw + 1, h - 4 - strip, x0 + (i + 1) * cw - 1, h - 4,
                                      fill=heat(pct), outline="")


class MonitorWindow(tk.Toplevel):
    def __init__(self, master, on_status: Callable[[str], None] = lambda text: None):
        super().__init__(master)
        self.on_status = on_status
        self.title(t("speed.monitor.window"))
        self.geometry("900x820")
        self.minsize(560, 520)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)             # chowa – test trwa dalej
        self.last: dict = {}
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        self.phase_label = ttk.Label(body, text=t("speed.monitor.idle"), style="Title.TLabel")
        self.phase_label.pack(anchor="w")
        self.values = ttk.Label(body, text="", style="Muted.TLabel")
        self.values.pack(anchor="w", pady=(2, 6))
        grid = ttk.Frame(body)
        grid.pack(fill="x")
        for col in (0, 1):
            grid.columnconfigure(col, weight=1, uniform="c")
        self.charts = [
            Chart(grid, t("speed.monitor.cpu"), [("cpu_pct", "#66c9fa", t("speed.monitor.load"))], "%",
                  cores=True),
            Chart(grid, t("speed.monitor.temp"), [("cpu_temp_c", "#f1a65b", "CPU"), ("gpu_temp_c", "#ff8282", "GPU")],
                  " °C", ymax=100.0, ymin=20.0),
            Chart(grid, t("speed.monitor.memory"), [("ram_used_mb", "#90e0ef", "RAM"),
                                                   ("swap_used_mb", "#f5c65f", "swap")], " MB",
                  ymax=lambda s: (s.get("ram_used_mb") or 0) + (s.get("ram_available_mb") or 0) or None),
            Chart(grid, t("speed.monitor.gpu"), [("gpu_util_pct", "#53d6a0", t("speed.monitor.load")),
                                                ("vram_pct", "#b48ead", "VRAM")], "%"),
        ]
        for i, chart in enumerate(self.charts):
            chart.grid(row=i // 2, column=i % 2, sticky="nsew", padx=3, pady=3)
        self.gpu_line = ttk.Label(body, text="", style="Muted.TLabel")
        self.gpu_line.pack(anchor="w", pady=(4, 0))
        self.throttle = ttk.Label(body, text="", style="Note.TLabel")   # widoczny tylko, gdy karta zwalnia

        self.task = ttk.Label(body, text=t("bench.preview.empty"), style="Title.TLabel", wraplength=860)
        self.task.pack(anchor="w", fill="x", pady=(10, 0))
        self.task.bind("<Configure>", lambda e: self.task.configure(wraplength=max(300, e.width)))
        tools = ttk.Frame(body)
        tools.pack(fill="x", pady=(4, 4))
        ttk.Button(tools, text=t("bench.preview.copy"), command=self.copy).pack(side="left")
        ttk.Button(tools, text=t("bench.preview.save"), command=self.save).pack(side="left", padx=6)
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(tools, text=t("bench.log.autoscroll"), variable=self.follow).pack(side="right")
        box = ttk.Frame(body)
        box.pack(fill="both", expand=True)
        self.text = tk.Text(box, wrap="word", font="TkFixedFont", background=BG, foreground=TEXT, height=8,
                            relief="flat", borderwidth=0, padx=12, pady=10, state="disabled",
                            selectbackground="#354e66", selectforeground="#ffffff")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for tag, color in TAGS.items():
            self.text.tag_configure(tag, foreground=color)

    # ------------------------------------------------------------------ telemetria
    def set_phase(self, text: str) -> None:
        self.phase_label.configure(text=text)

    def sample(self, s: dict) -> None:
        s = dict(s)
        if s.get("vram_used_mb") is not None and s.get("vram_total_mb"):
            s["vram_pct"] = 100 * s["vram_used_mb"] / s["vram_total_mb"]
        self.last = s
        for chart in self.charts:
            chart.push(s)
            chart.draw()
        self.values.configure(text=t(
            "speed.monitor.values", cpu=_fmt(s.get("cpu_pct")), cpu_t=_fmt(s.get("cpu_temp_c"), 0, " °C"),
            cpu_w=_fmt(s.get("cpu_power_w"), 0, " W"), ram=_fmt(s.get("ram_used_mb"), 0, " MB"),
            avail=_fmt(s.get("ram_available_mb"), 0, " MB"), swap=_fmt(s.get("swap_used_mb"), 0, " MB"),
            swap_io=self._swap_io(s)))
        self.gpu_line.configure(text=t(
            "speed.monitor.gpu_values", util=_fmt(s.get("gpu_util_pct"), 0, "%"),
            temp=_fmt(s.get("gpu_temp_c"), 0, " °C"), power=_fmt(s.get("gpu_power_w"), 0, " W"),
            clock=_fmt(s.get("gpu_clock_mhz"), 0, " MHz"), vram=_fmt(s.get("vram_used_mb"), 0),
            total=_fmt(s.get("vram_total_mb"), 0, " MB"), vram_t=_fmt(s.get("vram_temp_c"), 0, " °C")))
        reasons = s.get("gpu_throttle")
        active = [r for r in (reasons or "").split(",") if r and r not in ("none", "gpu_idle")]
        self.throttle.configure(text="⚠ " + t("speed.monitor.throttle", reasons=", ".join(
            t(f"speed.throttle.{r}") for r in active)) if active else "")
        if active and not self.throttle.winfo_manager():
            self.throttle.pack(anchor="w", after=self.gpu_line)
        elif not active and self.throttle.winfo_manager():
            self.throttle.pack_forget()

    @staticmethod
    def _swap_io(s: dict) -> str:
        if s.get("swap_in_bps") is None and s.get("swap_out_bps") is None:
            return t("speed.monitor.swap_na")
        kb = lambda v: _fmt((v or 0) / 1024, 0, " KB/s")            # noqa: E731
        return t("speed.monitor.swap_io", sin=kb(s.get("swap_in_bps")), sout=kb(s.get("swap_out_bps")))

    # ------------------------------------------------------------------ zadanie i odpowiedź
    def _insert(self, text: str, tag: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", text, (tag,))
        self.text.configure(state="disabled")
        if self.follow.get():
            self.text.see("end")

    def clear(self) -> None:
        self.task.configure(text=t("bench.preview.empty"))
        self.set_phase(t("speed.monitor.idle"))
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        for chart in self.charts:
            for d in chart.data.values():
                d.clear()
            chart.draw()

    def start(self, title: str, prompt: str) -> None:
        self.task.configure(text=title)
        if self.text.index("end-1c") != "1.0":
            self._insert("\n" + "─" * 60 + "\n", "head")
        self._insert(f"▶ {title}\n", "task")
        self._insert(t("bench.preview.prompt") + "\n", "head")
        self._insert(prompt + "\n\n", "prompt")
        self._insert(t("bench.preview.answer") + "\n", "head")

    def add(self, text: str) -> None:
        self._insert(text, "answer")

    def content(self) -> str:
        return self.text.get("1.0", "end-1c")

    def show(self) -> None:
        self.deiconify()
        self.lift()

    def copy(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.content())
        self.on_status(t("bench.preview.copied"))

    def save(self) -> None:
        name = f"bobr_monitor_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        path = filedialog.asksaveasfilename(parent=self, title=t("bench.log.save"), initialfile=name,
                                            defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            Path(path).write_text(self.content() + "\n", encoding="utf-8")
            self.on_status(t("bench.log.saved", path=path))
