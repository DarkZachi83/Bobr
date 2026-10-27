"""Okno „O programie” – wizytówka: ilustracja, wersja, autorzy, licencja, link."""
from __future__ import annotations

import tkinter as tk
import webbrowser
from tkinter import ttk

from aibobr import __version__, resources
from aibobr.i18n import t

REPO_URL = "https://github.com/DarkZachi83/Bobr"


class AboutDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title(t("gui.menu.about"))
        self.transient(master)
        self.resizable(False, False)
        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)

        picture = resources.path("assets", "about.png")
        if picture.exists():
            self.image = tk.PhotoImage(file=str(picture))      # referencja, inaczej znika
            ttk.Label(body, image=self.image).pack(pady=(0, 10))
        ttk.Label(body, text=t("gui.app_name"), font=("TkDefaultFont", 16, "bold")).pack()
        ttk.Label(body, text=f"AI Hardware Checker · {__version__}").pack(pady=(0, 8))
        ttk.Label(body, text=t("gui.about_text"), justify="center", wraplength=500).pack()

        buttons = ttk.Frame(body)
        buttons.pack(pady=(12, 0))
        ttk.Button(buttons, text=t("gui.open_page"),
                   command=lambda: webbrowser.open(REPO_URL)).pack(side="left", padx=4)
        ttk.Button(buttons, text=t("gui.close"), command=self.destroy).pack(side="left", padx=4)
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()
