"""Wybór modelu zainstalowanego w Ollamie (tryb Zaawansowany)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from aibobr.i18n import fmt_gb, t
from aibobr.sources.base import SourceError


class OllamaModelsDialog(tk.Toplevel):
    def __init__(self, master, worker, fetch: Callable[[], list[dict]]):
        super().__init__(master)
        self.title(t("gui.ollama_title"))
        self.transient(master)
        self.result: Optional[str] = None
        self.names: list[str] = []
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        self.status = ttk.Label(body, text=t("gui.ollama_loading"), wraplength=420)
        self.status.pack(anchor="w", pady=(0, 6))
        self.listbox = tk.Listbox(body, height=12, width=56, activestyle="dotbox", exportselection=False)
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<Double-Button-1>", lambda e: self._ok())
        self.listbox.bind("<Return>", lambda e: self._ok())
        buttons = ttk.Frame(body)
        buttons.pack(anchor="e", pady=(8, 0))
        ttk.Button(buttons, text=t("gui.cancel"), command=self.destroy).pack(side="right")
        self.ok = ttk.Button(buttons, text=t("gui.choose"), style="Accent.TButton", command=self._ok)
        self.ok.pack(side="right", padx=6)
        self.ok.state(["disabled"])
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()
        worker.run("ollama_list", fetch, self._fill, self._failed)

    def _fill(self, models: list[dict]) -> None:
        if not self.winfo_exists():
            return
        if not models:
            self.status.configure(text=t("gui.ollama_empty"))
            return
        self.status.configure(text=t("gui.ollama_hint"))
        for m in sorted(models, key=lambda m: str(m.get("name", ""))):
            name = str(m.get("name", ""))
            quant = (m.get("details") or {}).get("quantization_level") or ""
            self.names.append(name)
            self.listbox.insert("end", f"{name}   ·   {quant}   ·   {fmt_gb(m.get('size'))}")
        self.listbox.selection_set(0)
        self.listbox.focus_set()
        self.ok.state(["!disabled"])

    def _failed(self, error: BaseException) -> None:
        if self.winfo_exists():
            code = error.code if isinstance(error, SourceError) else "ollama.not_running"
            self.status.configure(text=t(code))

    def _ok(self) -> None:
        sel = self.listbox.curselection()
        if sel:
            self.result = self.names[sel[0]]
            self.destroy()
