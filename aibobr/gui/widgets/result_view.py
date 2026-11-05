"""Wynik sprawdzenia (etap 13b): nagłówek, pasek pamięci, werdykt, uwagi, przyciski, wyjaśnienie.

Ten sam widok w zakładce Sprawdź (Twój komputer) i Zaawansowane (sprzęt wpisany ręcznie).
Ramka ma kolor werdyktu (decyzja autora: ramki neutralne, tylko wynik mówi kolorem).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from aibobr.gui.theme import COLORS, VERDICT_COLORS
from aibobr.gui.widgets.explanation import ExplanationPanel
from aibobr.gui.widgets.memory_bar import MemoryBar
from aibobr.knowledge.advisor import render


class ResultView(tk.Frame):
    def __init__(self, master, wrap: Callable[[], int]):
        super().__init__(master, background=COLORS["panel"], highlightthickness=2,
                         highlightbackground=COLORS["border"], highlightcolor=COLORS["border"])
        self.wrap = wrap
        self.header = ttk.Label(self, text="", style="Title.TLabel")
        self.header.pack(anchor="w", padx=10, pady=(8, 0))
        self.bar = MemoryBar(self)
        self.bar.pack(fill="x", padx=6, pady=(4, 0))
        self.verdict = ttk.Label(self, text="", style="Verdict.TLabel", wraplength=700)
        self.verdict.pack(anchor="w", padx=10)
        self.notes = ttk.Frame(self, style="Inner.TFrame")
        self.notes.pack(fill="x", padx=10, pady=(2, 0))
        self.actions = ttk.Frame(self, style="Inner.TFrame")      # przyciski (tylko w Sprawdź)
        self.actions.pack(fill="x", padx=10, pady=(6, 0))
        self.explanation = ExplanationPanel(self)
        self.explanation.pack(fill="x", padx=1, pady=(4, 1))
        self.bind("<Configure>", lambda e: self._rewrap())

    def _border(self, color: str) -> None:
        self.configure(highlightbackground=color, highlightcolor=color)

    def _rewrap(self) -> None:
        width = max(300, self.winfo_width() - 30)
        self.verdict.configure(wraplength=width)
        for child in self.notes.winfo_children():
            if child.winfo_class() == "TLabel":
                child.configure(wraplength=width)

    def clear(self, message: str = "", style: str = "Verdict.TLabel", color: Optional[str] = None,
              hint: bool = True) -> None:
        self._border(COLORS["border"])
        self.bar.set_verdict(None, hint=hint)
        self.header.configure(text="")
        self.verdict.configure(text=message, style=style, foreground=color or COLORS["text"])
        for child in self.notes.winfo_children():
            child.destroy()
        self.explanation.set_explanation(None)

    def show(self, header: str, verdict, expl) -> None:
        color = VERDICT_COLORS[verdict.level]
        self._border(color)
        self.header.configure(text=header)
        self.bar.set_verdict(verdict)
        self.verdict.configure(text=render(expl.summary), style="Verdict.TLabel", foreground=color)
        for child in self.notes.winfo_children():
            child.destroy()
        width = max(300, self.winfo_width() - 30)
        for note in expl.notes:
            ttk.Label(self.notes, text="⚠ " + render(note), style="Note.TLabel",
                      wraplength=width).pack(anchor="w")
        self.explanation.set_explanation(expl)
