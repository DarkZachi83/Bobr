"""Mierniki kołowe (etap 13b): jaką część sprzętu zajmie sprawdzany model.

VRAM: część modelu w karcie / VRAM użytkowy, RAM: część w RAM / RAM użytkowy (decyzja autora:
procent zasobów w odniesieniu do analizowanego modelu – zajętość systemu pokazuje już werdykt).
Kolor VRAM = kolor werdyktu. Czerwony werdykt: pełny czerwony pierścień i „brak X GB”.
Rysowane na Canvas – działa tak samo z ttkbootstrap i bez niego.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

from aibobr.domain.enums import FitLevel
from aibobr.domain.models import Verdict
from aibobr.gui.theme import COLORS, VERDICT_COLORS
from aibobr.i18n import fmt_gb, t

SIZE = 150            # średnica pierścienia
THICK = 14
CELL = SIZE + 46      # pierścień + podpis pod spodem


def meter_values(v: Verdict) -> dict:
    """{"vram"|"ram": (procent 0–100 albo None, podpis pod spodem)} – czyste obliczenia."""
    cap = v.capacity
    out = {}
    if v.level == FitLevel.RED:
        short = t("gui.meter.short", size=fmt_gb(v.shortfall))
        out["vram"] = (100.0 if cap.vram_usable else None, short if cap.vram_usable else t("gui.meter.no_gpu"))
        out["ram"] = (100.0, short)
        return out
    if cap.vram_usable:
        out["vram"] = (100.0 * v.in_vram / cap.vram_usable,
                       t("gui.meter.of", used=fmt_gb(v.in_vram), total=fmt_gb(cap.vram_usable)))
    else:
        out["vram"] = (None, t("gui.meter.no_gpu"))
    out["ram"] = (100.0 * v.in_ram / cap.ram_usable if cap.ram_usable else 0.0,
                  t("gui.meter.of", used=fmt_gb(v.in_ram), total=fmt_gb(cap.ram_usable)))
    return out


class Meters(tk.Canvas):
    def __init__(self, master, vertical: bool = True, **kw):
        # Pionowo – jeden pod drugim (uwaga autora: ładniej w wąskiej prawej kolumnie).
        self.vertical = vertical
        height = 2 * CELL + 8 if vertical else CELL
        super().__init__(master, height=height, width=SIZE + 40 if vertical else 2 * SIZE + 60,
                         highlightthickness=0, background=COLORS["panel"], **kw)
        self.verdict: Optional[Verdict] = None
        self.font = tkfont.nametofont("TkDefaultFont")
        self.big = self.font.copy()
        self.big.configure(size=max(16, int(abs(self.font.cget("size")) * 1.8)), weight="bold")
        self.bold = self.font.copy()
        self.bold.configure(weight="bold")
        self.bind("<Configure>", lambda e: self.redraw())

    def set_verdict(self, verdict: Optional[Verdict]) -> None:
        self.verdict = verdict
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), SIZE + 20 if self.vertical else 2 * SIZE + 20)
        gap = (width - 2 * SIZE) / 3
        v = self.verdict
        values = meter_values(v) if v is not None else {"vram": (None, ""), "ram": (None, "")}
        level_color = VERDICT_COLORS[v.level] if v is not None else COLORS["muted"]
        colors = {"vram": level_color,
                  "ram": COLORS["error"] if v is not None and v.level == FitLevel.RED else COLORS["ram"]}
        for i, (key, label) in enumerate((("vram", "VRAM"), ("ram", "RAM"))):
            pct, caption = values[key]
            if self.vertical:
                x0, y0 = (width - SIZE) / 2, 4 + i * (CELL + 8)
            else:
                x0, y0 = gap + i * (SIZE + gap), 4
            box = (x0 + THICK / 2, y0 + THICK / 2, x0 + SIZE - THICK / 2, y0 + SIZE - THICK / 2)
            self.create_oval(*box, outline=COLORS["track"], width=THICK)
            if pct:
                extent = -359.9 * min(pct, 100.0) / 100
                self.create_arc(*box, start=90, extent=extent, style="arc", outline=colors[key], width=THICK)
            cx, cy = x0 + SIZE / 2, y0 + SIZE / 2
            text = "–" if pct is None else f"{pct:.0f}%"
            self.create_text(cx, cy - 8, text=text, font=self.big,
                             fill=colors[key] if pct else COLORS["text"])
            self.create_text(cx, cy + 18, text=label, font=self.bold, fill=COLORS["muted"])
            self.create_text(cx, y0 + SIZE + 18, text=caption, font=self.font, fill=COLORS["muted"],
                             width=width - 12 if self.vertical else SIZE + gap - 6)
