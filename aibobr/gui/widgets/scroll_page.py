"""Strona zakładki z przewijaniem w pionie i dwiema kolumnami (etap 13b).

Szerokie okno: lewa kolumna (model, wynik) obok prawej (Twój komputer). Wąskie – np. laptop
1366×768 ze skalowaniem 125% (ok. 1090 px) – prawa kolumna przechodzi pod lewą, a całość
się przewija, żeby nic nie znikało poza ekranem.
"""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

from aibobr.gui.theme import COLORS

BREAKPOINT = 1000          # px szerokości strony: poniżej – jedna kolumna
RIGHT_WIDTH = 380          # stała szerokość prawej kolumny w układzie dwukolumnowym


class ScrollPage(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=COLORS["bg"])
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self._update_region())
        self.canvas.bind("<Configure>", self._resize)
        self.left = ttk.Frame(self.inner)
        self.right = ttk.Frame(self.inner, width=RIGHT_WIDTH)
        self.wide = None
        self._place(True)
        # Kółko myszy tylko, gdy kursor jest nad tą stroną (inne zakładki mają swoje).
        self.bind("<Enter>", lambda e: self._wheel(True))
        self.bind("<Leave>", lambda e: self._wheel(False))

    # ------------------------------------------------------------------ układ
    def _place(self, wide: bool) -> None:
        if wide == self.wide:
            return
        self.wide = wide
        self.left.grid_forget()
        self.right.grid_forget()
        if wide:
            # Obie kolumny do dołu okna – ramki kończą się równo (uwaga autora).
            self.inner.columnconfigure(0, weight=1)
            self.inner.columnconfigure(1, weight=0, minsize=RIGHT_WIDTH)
            self.inner.rowconfigure(0, weight=1)
            self.inner.rowconfigure(1, weight=0)
            self.left.grid(row=0, column=0, sticky="nswe", padx=(0, 10))
            self.right.grid(row=0, column=1, sticky="nswe")
        else:
            self.inner.columnconfigure(0, weight=1)
            self.inner.columnconfigure(1, weight=0, minsize=0)
            self.inner.rowconfigure(0, weight=0)
            self.inner.rowconfigure(1, weight=1)
            self.left.grid(row=0, column=0, sticky="nwe")
            self.right.grid(row=1, column=0, sticky="nswe", pady=(10, 0))

    def _resize(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)
        self._place(event.width >= BREAKPOINT)
        self._update_region()

    def _update_region(self) -> None:
        # Strona co najmniej tak wysoka jak okno – kolumny sięgają dołu; dłuższa – przewijanie.
        height = max(self.canvas.winfo_height(), self.inner.winfo_reqheight())
        if int(self.canvas.itemcget(self._window, "height") or 0) != height:
            self.canvas.itemconfigure(self._window, height=height)
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), height))
        fits = self.inner.winfo_reqheight() <= self.canvas.winfo_height()
        if fits:
            self.canvas.yview_moveto(0)

    @property
    def wrap(self) -> int:
        """Szerokość tekstu w lewej kolumnie (zawijanie werdyktu i uwag)."""
        return max(360, self.left.winfo_width() - 40)

    # ------------------------------------------------------------------ kółko myszy
    def _wheel(self, on: bool) -> None:
        if on:
            if sys.platform.startswith("linux"):
                self.bind_all("<Button-4>", lambda e: self._scroll(-1))
                self.bind_all("<Button-5>", lambda e: self._scroll(1))
            else:
                self.bind_all("<MouseWheel>", lambda e: self._scroll(-1 if e.delta > 0 else 1))
        else:
            for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
                self.unbind_all(seq)

    def _scroll(self, step: int) -> None:
        if self.inner.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(step * 3, "units")
