"""Pasek pamięci (SPEC §6): warstwa systemu i warstwa modelu na wspólnej skali."""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

from aibobr.domain.models import Verdict
from aibobr.domain.models import ImageNeed
from aibobr.gui.bar_layout import bar_layout, image_bar_layout
from aibobr.gui.theme import COLORS, LEVEL_COLORS
from aibobr.i18n import fmt_gb, t

LABEL_W = 76        # kolumna z napisami „System” / „Model”
ROW_H = 30
SYS_Y, MODEL_Y = 12, 54
LEGEND_Y = 104
LEGEND_LINE = 20     # druga linia legendy w wąskim oknie
HEIGHT = 140
# Modele obrazów: trzy tory (System, Szczyt, Naraz)
PEAK_Y, ALL_Y = 50, 88
IMG_LEGEND_Y = 136
IMG_HEIGHT = 172


class MemoryBar(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, height=HEIGHT, width=860, highlightthickness=0,
                         background=COLORS["panel"], **kw)
        self.verdict: Optional[Verdict] = None
        self.hint = True
        self.font = tkfont.nametofont("TkDefaultFont")
        self.bold = self.font.copy()
        self.bold.configure(weight="bold")
        self.bind("<Configure>", lambda e: self.redraw())

    def set_verdict(self, verdict: Optional[Verdict], hint: bool = True) -> None:
        """hint=False: brak wyniku z innego powodu niż brak modelu – pusty pasek bez podpowiedzi."""
        self.verdict, self.hint = verdict, hint
        self.redraw()

    # ------------------------------------------------------------------ rysowanie
    def _label_inside(self, seg, y, text, color):
        if self.font.measure(text) + 12 < seg.width:
            self.create_text((seg.x0 + seg.x1) / 2, y + ROW_H / 2, text=text,
                             fill=color, font=self.bold)

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 300)
        if self.verdict is None:
            if not self.hint:
                return
            self.create_text(width / 2, HEIGHT / 2, text=t("gui.bar.empty"),
                             fill=COLORS["muted"], font=self.font)
            return
        v = self.verdict
        left, right = LABEL_W, width - 12
        image = isinstance(v.need, ImageNeed)
        wanted = IMG_HEIGHT if image else HEIGHT
        if int(self.cget("height")) != wanted:
            self.configure(height=wanted)
        if image:
            self._redraw_image(v, left, right)
            return
        lay = bar_layout(v, left, right - left)
        weights_c, kv_c = LEVEL_COLORS[v.level]

        self.create_text(8, SYS_Y + ROW_H / 2, text=t("cli.row.system"), anchor="w", font=self.bold)
        self.create_text(8, MODEL_Y + ROW_H / 2, text=t("cli.row.model"), anchor="w", font=self.bold)
        for y in (SYS_Y, MODEL_Y):                                   # tło torów
            self.create_rectangle(left, y, right, y + ROW_H, fill="#eef1f4", outline="")

        for seg in lay.system:
            if seg.kind == "missing":
                self.create_rectangle(seg.x0, SYS_Y, seg.x1, SYS_Y + ROW_H,
                                      fill=COLORS["missing"], outline=COLORS["missing_line"])
                self._label_inside(seg, SYS_Y, t("gui.bar.missing", size=fmt_gb(seg.value)),
                                   COLORS["error"])
                continue
            self.create_rectangle(seg.x0, SYS_Y, seg.x1, SYS_Y + ROW_H,
                                  fill=COLORS[seg.kind], outline="")
            label = "VRAM" if seg.kind == "vram" else "RAM"
            self._label_inside(seg, SYS_Y, f"{label} {fmt_gb(seg.value)}", "white")

        colors = {"weights": weights_c, "kv": kv_c, "overhead": COLORS["overhead"],
                  "decoder": COLORS["decoder"]}
        for seg in lay.model:
            self.create_rectangle(seg.x0, MODEL_Y, seg.x1, MODEL_Y + ROW_H,
                                  fill=colors[seg.kind], outline="")
        if lay.model:
            first = lay.model[0]
            self._label_inside(first, MODEL_Y, fmt_gb(v.need.total), "white")

        for x in lay.markers:                                         # granice stref
            self.create_line(x, SYS_Y - 6, x, MODEL_Y + ROW_H + 6, dash=(4, 3),
                             fill=COLORS["marker"], width=1)

        need, cap = v.need, v.capacity
        items = [
            (COLORS["vram"], t("gui.bar.vram", size=fmt_gb(cap.vram_usable))),
            (COLORS["ram"], t("gui.bar.ram", size=fmt_gb(cap.ram_usable))),
            (weights_c, t("gui.bar.weights", size=fmt_gb(need.weights_total))),
            (kv_c, t("gui.bar.kv", size=fmt_gb(need.kv_cache))),
            (COLORS["overhead"], t("gui.bar.overhead", size=fmt_gb(need.overhead))),
        ]
        if need.decoder:
            items.append((COLORS["decoder"], t("gui.bar.decoder", size=fmt_gb(need.decoder))))
        x, y = left, LEGEND_Y
        for color, text in items:
            if x + 14 + self.font.measure(text) > right:
                if y > LEGEND_Y:
                    break                                             # bardzo wąskie okno
                x, y = left, y + LEGEND_LINE                          # wąskie okno: druga linia
            self.create_rectangle(x, y - 5, x + 10, y + 5, fill=color, outline="")
            self.create_text(x + 14, y, text=text, anchor="w", font=self.font,
                             fill=COLORS["muted"])
            x += 14 + self.font.measure(text) + 16

    def _system_row(self, segments, y) -> None:
        for seg in segments:
            if seg.kind == "missing":
                self.create_rectangle(seg.x0, y, seg.x1, y + ROW_H, fill=COLORS["missing"],
                                      outline=COLORS["missing_line"])
                self._label_inside(seg, y, t("gui.bar.missing", size=fmt_gb(seg.value)), COLORS["error"])
                continue
            self.create_rectangle(seg.x0, y, seg.x1, y + ROW_H, fill=COLORS[seg.kind], outline="")
            label = "VRAM" if seg.kind == "vram" else "RAM"
            self._label_inside(seg, y, f"{label} {fmt_gb(seg.value)}", "white")

    def _legend(self, items, left, right, y0) -> None:
        x, y = left, y0
        for color, text in items:
            if x + 14 + self.font.measure(text) > right:
                if y > y0:
                    break
                x, y = left, y + LEGEND_LINE
            self.create_rectangle(x, y - 5, x + 10, y + 5, fill=color, outline="")
            self.create_text(x + 14, y, text=text, anchor="w", font=self.font, fill=COLORS["muted"])
            x += 14 + self.font.measure(text) + 16

    def _redraw_image(self, v: Verdict, left: float, right: float) -> None:
        """Modele obrazów (SPEC §5a): System · Szczyt (kolor werdyktu) · Naraz (jaśniej)."""
        lay = image_bar_layout(v, left, right - left)
        main_c, light_c = LEVEL_COLORS[v.level]
        rows = ((SYS_Y, t("cli.row.system")), (PEAK_Y, t("gui.bar.peak")), (ALL_Y, t("gui.bar.all")))
        for y, label in rows:
            self.create_text(8, y + ROW_H / 2, text=label, anchor="w", font=self.bold)
            self.create_rectangle(left, y, right, y + ROW_H, fill="#eef1f4", outline="")
        self._system_row(lay.system, SYS_Y)
        colors = {"main": main_c, "vae": main_c, "encoders": main_c, "compute": light_c,
                  "overhead": COLORS["overhead"]}
        for seg in lay.peak:
            self.create_rectangle(seg.x0, PEAK_Y, seg.x1, PEAK_Y + ROW_H, fill=colors[seg.kind], outline="")
        if lay.peak:
            self._label_inside(lay.peak[0], PEAK_Y, fmt_gb(v.need.peak), "white")
        soft = {"main": "#c9cfd6", "compute": "#dde1e6", "overhead": "#e4e7eb",
                "encoders": "#b9c4d0", "vae": "#cdd6df"}
        for seg in lay.at_once:
            self.create_rectangle(seg.x0, ALL_Y, seg.x1, ALL_Y + ROW_H, fill=soft[seg.kind], outline="")
        if lay.at_once:
            self._label_inside(lay.at_once[0], ALL_Y, fmt_gb(v.need.all_at_once), COLORS["text"])
        for x in lay.markers:
            self.create_line(x, SYS_Y - 6, x, ALL_Y + ROW_H + 6, dash=(4, 3), fill=COLORS["marker"], width=1)
        n, cap = v.need, v.capacity
        comp = {"sample": n.main_compute, "decode": n.vae_compute, "encode": 0}[n.peak_stage]
        items = [(COLORS["vram"], t("gui.bar.vram", size=fmt_gb(cap.vram_usable))),
                 (COLORS["ram"], t("gui.bar.ram", size=fmt_gb(cap.ram_usable))),
                 (main_c, t("gui.bar.main", size=fmt_gb(n.main))),
                 (light_c, t("gui.bar.compute", size=fmt_gb(comp))),
                 (COLORS["overhead"], t("gui.bar.overhead", size=fmt_gb(n.overhead))),
                 (soft["encoders"], t("gui.bar.encoders", size=fmt_gb(n.encoders))),
                 (soft["vae"], t("gui.bar.vae", size=fmt_gb(n.vae)))]
        self._legend(items, left, right, IMG_LEGEND_Y)
