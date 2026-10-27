"""Okno „Wpisz dane modelu ręcznie” (SPEC §4, trzecie źródło).

Poziom podstawowy: parametry, kwantyzacja (wynik oznaczony jako szacunek).
Poziom zaawansowany: warstwy, głowice KV, wymiar głowicy – razem albo wcale.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from aibobr.calculator.quants import QuantTable
from aibobr.gui.widgets.hardware_panel import parse_number
from aibobr.i18n import t
from aibobr.sources.base import LoadedModel, SourceError
from aibobr.sources.manual import ManualInput, load_manual


class ManualModelDialog(tk.Toplevel):
    FIELDS = [("name", "gui.manual.name", False), ("params", "gui.manual.params", False),
              ("layers", "gui.manual.layers", True), ("kv_heads", "gui.manual.kv_heads", True),
              ("head_dim", "gui.manual.head_dim", True), ("max_context", "gui.manual.max_context", True),
              ("active", "gui.manual.active", True)]

    def __init__(self, master, quants: QuantTable):
        super().__init__(master)
        self.title(t("gui.manual.title"))
        self.transient(master)
        self.resizable(False, False)
        self.quants = quants
        self.result: Optional[LoadedModel] = None
        self.vars = {key: tk.StringVar() for key, _, _ in self.FIELDS}
        self.quant = tk.StringVar(value="Q4_K_M")

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        r = 0
        for key, label, optional in self.FIELDS:
            if key == "layers":
                ttk.Label(body, text=t("gui.manual.advanced"), font=("TkDefaultFont", 9, "bold")).grid(
                    row=r, column=0, columnspan=2, sticky="w", pady=(10, 2))
                r += 1
            ttk.Label(body, text=t(label)).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=2)
            ttk.Entry(body, textvariable=self.vars[key], width=24).grid(row=r, column=1, sticky="w")
            r += 1
            if key == "params":
                ttk.Label(body, text=t("gui.quant")).grid(row=r, column=0, sticky="w", pady=2)
                ttk.Combobox(body, textvariable=self.quant, values=quants.names_by_size(),
                             state="readonly", width=12).grid(row=r, column=1, sticky="w")
                r += 1
        self.error = ttk.Label(body, text="", foreground="#b42318", wraplength=360)
        self.error.grid(row=r, column=0, columnspan=2, sticky="w", pady=(8, 0))
        buttons = ttk.Frame(body)
        buttons.grid(row=r + 1, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text=t("gui.cancel"), command=self.destroy).pack(side="right")
        ttk.Button(buttons, text=t("gui.ok"), style="Accent.TButton",
                   command=self._ok).pack(side="right", padx=6)
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()

    def _opt_int(self, key: str) -> Optional[int]:
        text = self.vars[key].get().strip()
        return int(parse_number(text)) if text else None

    def _ok(self) -> None:
        try:
            params = parse_number(self.vars["params"].get())
            active = self.vars["active"].get().strip()
            data = ManualInput(
                params_billion=params, quant=self.quant.get(), name=self.vars["name"].get(),
                n_layers=self._opt_int("layers"), n_kv_heads=self._opt_int("kv_heads"),
                head_dim=self._opt_int("head_dim"), max_context=self._opt_int("max_context"),
                active_params_billion=parse_number(active) if active else None,
                # Podanie aktywnych parametrów = model MoE; dokładna liczba ekspertów
                # nie jest potrzebna do obliczeń, wystarczy oznaczyć model jako MoE.
                n_experts=2 if active else None)
            self.result = load_manual(data, self.quants)
        except ValueError:
            self.error.configure(text=t("gui.manual.numbers"))
            return
        except SourceError as e:
            self.error.configure(text=t(e.code))
            return
        self.destroy()
