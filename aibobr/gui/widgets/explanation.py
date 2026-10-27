"""Przycisk „Wyjaśnienie” i zwijany panel (SPEC §8) – domyślnie zwinięty."""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont, ttk
from typing import Optional

from aibobr.gui.theme import COLORS
from aibobr.i18n import t
from aibobr.knowledge.advisor import Explanation, render, render_list


class ExplanationPanel(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, style="Inner.TFrame", **kw)
        self.expanded = False
        self.explanation: Optional[Explanation] = None
        self.button = ttk.Button(self, command=self.toggle)
        self.button.pack(anchor="w", padx=9, pady=(6, 8))
        self.text = tk.Text(self, height=14, wrap="word", relief="flat", borderwidth=0,
                            background=COLORS["panel"], foreground=COLORS["text"],
                            padx=12, pady=4, cursor="arrow", font="TkDefaultFont")
        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold")
        self.text.tag_configure("title", font=bold, spacing1=10, spacing3=2)
        self.text.tag_configure("item", lmargin1=14, lmargin2=30, spacing1=1)
        # Szerokość okna zmienia liczbę zawiniętych linii – przeliczamy wysokość po zmianie.
        self.text.bind("<Configure>", lambda e: self._fit_height())
        self._update_button()

    def _update_button(self) -> None:
        arrow = "▾" if self.expanded else "▸"
        self.button.configure(text=f"{arrow} {t('gui.explain')}")
        self.button.state(["!disabled"] if self.explanation else ["disabled"])

    def toggle(self) -> None:
        if not self.expanded and self.explanation is None:
            return
        self.expanded = not self.expanded
        if self.expanded:
            self.text.pack(fill="both", expand=True, padx=2, pady=(0, 8))
        else:
            self.text.pack_forget()
        self._update_button()

    def set_explanation(self, explanation: Optional[Explanation]) -> None:
        self.explanation = explanation
        if explanation is None and self.expanded:
            self.toggle()
        self._fill()
        self._update_button()

    def _fill(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        e = self.explanation
        if e is not None:
            sections = [(render(e.need[0]), e.need[1:]),
                        (t("expl.pc.title"), e.computer),
                        (t(e.actions_title), e.actions)]
            for i, (title, lines) in enumerate(sections):
                self.text.insert("end", title + "\n", "title")
                for text in render_list(lines):
                    self.text.insert("end", f"→ {text}\n", "item")
        self.text.configure(state="disabled")
        self.after_idle(self._fit_height)

    def _fit_height(self) -> None:
        """Wysokość w liniach ekranu (z zawijaniem i odstępami), żeby nic nie ucinać."""
        self.text.update_idletasks()
        pixels = self.text.count("1.0", "end", "ypixels")
        line = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        value = pixels[0] if isinstance(pixels, tuple) else (pixels or 0)
        lines = max(5, min(value // max(line, 1) + 1, 32))
        if int(self.text.cget("height")) != lines:      # bez pętli zdarzeń <Configure>
            self.text.configure(height=lines)
