"""Okno „Jaki test wykonać?” przy brakującym albo podwójnym tokenie początku (decyzja autora 12.11.2026, pkt 5).

Decyduje użytkownik; program tylko podpowiada, co da każda opcja:
- jak podaje Ollama – tak model działa na co dzień, wynik może być zaniżony;
- z poprawionym tokenem początku – osobna wersja w Ollamie (np. …-bos), tworzona raz; oryginał nietknięty;
- oba – porównanie: pokazuje, ile zmienia sam token początku; trwa dwa razy dłużej.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from aibobr.i18n import t

CHOICES = ("as_is", "fixed", "both")


class BosChoiceDialog(tk.Toplevel):
    def __init__(self, master, checks: list, variants: dict[str, str]):
        """checks – wyniki z ostrzeżeniem (BosCheck); variants – model → nazwa poprawionej wersji."""
        super().__init__(master)
        self.title(t("d.choice.title"))
        self.resizable(False, False)
        self.transient(master.winfo_toplevel())
        self.result: Optional[str] = None
        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)
        lines = [f"• {c.model}: {t('d.bos.' + c.status)}" for c in checks]
        ttk.Label(body, text=t("d.choice.intro") + "\n" + "\n".join(lines), wraplength=560,
                  justify="left").pack(anchor="w")
        self.var = tk.StringVar(value="as_is")
        names = ", ".join(variants.values())
        for key in CHOICES:
            ttk.Radiobutton(body, text=t(f"d.choice.{key}"), value=key, variable=self.var).pack(anchor="w", pady=(10, 0))
            ttk.Label(body, text=t(f"d.choice.{key}.hint", names=names), wraplength=540, justify="left",
                      foreground="#8b929b").pack(anchor="w", padx=(24, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text=t("d.choice.cancel"), command=self.destroy).pack(side="right")
        ttk.Button(buttons, text=t("d.choice.ok"), style="Accent.TButton", command=self._ok).pack(side="right", padx=6)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _ok(self) -> None:
        self.result = self.var.get()
        self.destroy()

    @classmethod
    def ask(cls, master, checks: list, variants: dict[str, str]) -> Optional[str]:
        dialog = cls(master, checks, variants)
        dialog.grab_set()
        master.wait_window(dialog)
        return dialog.result
