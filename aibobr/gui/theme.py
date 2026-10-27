"""Kolory, czcionki i ustawienia ekranu dla GUI."""
from __future__ import annotations

import sys

from aibobr.domain.enums import FitLevel

COLORS = {
    "bg": "#f6f7f9",
    "panel": "#ffffff",
    "text": "#1f2328",
    "muted": "#5b6470",
    "border": "#d0d7de",
    # pasek systemu
    "vram": "#3fa66b",
    "ram": "#e5ad37",
    "missing": "#f3c2c2",
    "missing_line": "#c94a4a",
    # pasek modelu – kolor werdyktu, KV jaśniejszy, narzut szary
    "overhead": "#8b929b",
    "marker": "#30363d",
    "warn": "#9a6700",
    "error": "#b42318",
}

LEVEL_COLORS = {
    FitLevel.GREEN: ("#1f7a45", "#7cc49a"),     # (wagi, pamięć rozmowy)
    FitLevel.YELLOW: ("#b7791f", "#e8c27a"),
    FitLevel.RED: ("#b42318", "#ec8f88"),
}

VERDICT_COLORS = {FitLevel.GREEN: "#1f7a45", FitLevel.YELLOW: "#9a6700", FitLevel.RED: "#b42318"}


def enable_dpi_awareness() -> None:
    """Windows ze skalowaniem 125–150%: bez tego okno tkintera jest rozmazane (SPEC §11)."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def apply_style(root) -> None:
    from tkinter import ttk
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    root.configure(background=COLORS["bg"])
    style.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("Panel.TFrame", background=COLORS["panel"], relief="solid", borderwidth=1)
    style.configure("Inner.TFrame", background=COLORS["panel"], relief="flat", borderwidth=0)
    style.configure("Panel.TLabel", background=COLORS["panel"])
    style.configure("Panel.TCheckbutton", background=COLORS["panel"])
    style.configure("Panel.TRadiobutton", background=COLORS["panel"])
    style.configure("Muted.TLabel", foreground=COLORS["muted"], background=COLORS["panel"])
    style.configure("Title.TLabel", font=("TkDefaultFont", 11, "bold"), background=COLORS["panel"])
    style.configure("Verdict.TLabel", font=("TkDefaultFont", 12, "bold"), background=COLORS["panel"])
    style.configure("Note.TLabel", foreground=COLORS["warn"], background=COLORS["panel"])
    style.configure("Error.TLabel", foreground=COLORS["error"], background=COLORS["bg"])
    style.configure("Invalid.TEntry", fieldbackground="#fde2e1")
    style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))
