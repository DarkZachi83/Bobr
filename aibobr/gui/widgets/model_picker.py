"""Panel „Wskaż model” (SPEC §4): link / plik .gguf / wpis ręczny + wybór kwantyzacji."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Optional

from aibobr.i18n import fmt_gb, t
from aibobr.sources.base import LoadedModel


class ModelPicker(ttk.Frame):
    def __init__(self, master, on_check: Callable[[str], None], on_manual: Callable[[], None],
                 on_quant: Callable[[str], None]):
        super().__init__(master, style="Panel.TFrame")
        self.on_check, self.on_quant = on_check, on_quant
        self.text = tk.StringVar()
        self.quant_label = tk.StringVar()
        self._quants: dict[str, str] = {}

        ttk.Label(self, text=t("gui.model"), style="Title.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(8, 2))
        ttk.Label(self, text=t("cli.ask_model_hint"), style="Muted.TLabel").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=10)
        self.entry = ttk.Entry(self, textvariable=self.text)
        self.entry.grid(row=2, column=0, sticky="we", padx=(10, 6), pady=6)
        self.entry.bind("<Return>", lambda e: self._check())
        ttk.Button(self, text=t("gui.browse"), command=self.browse).grid(row=2, column=1, padx=(0, 6))
        self.check_button = ttk.Button(self, text=t("gui.check"), style="Accent.TButton",
                                       command=self._check)
        self.check_button.grid(row=2, column=2, padx=(0, 10))
        self.columnconfigure(0, weight=1)

        bottom = ttk.Frame(self, style="Inner.TFrame")
        bottom.grid(row=3, column=0, columnspan=3, sticky="we", padx=10, pady=(0, 8))
        ttk.Label(bottom, text=t("gui.quant"), style="Muted.TLabel").pack(side="left")
        self.combo = ttk.Combobox(bottom, textvariable=self.quant_label, state="disabled", width=26)
        self.combo.pack(side="left", padx=6)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.on_quant(self._quants[self.quant_label.get()]))
        ttk.Button(bottom, text=t("gui.manual"), command=on_manual).pack(side="right")
        self.status = ttk.Label(self, text="", style="Muted.TLabel", wraplength=760)
        self.status.grid(row=4, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 8))
        self.status.grid_remove()

    def _check(self) -> None:
        if self.text.get().strip():
            self.on_check(self.text.get())

    def browse(self) -> None:
        path = filedialog.askopenfilename(
            title=t("gui.browse_title"),
            filetypes=[(t("gui.gguf_files"), "*.gguf"), (t("gui.all_files"), "*")])
        if path:
            self.text.set(path)
            self.on_check(path)

    def set_busy(self, busy: bool) -> None:
        self.check_button.state(["disabled"] if busy else ["!disabled"])

    def set_status(self, text: str = "", kind: str = "Muted") -> None:
        if text:
            self.status.configure(text=text, style=f"{kind}.TLabel" if kind != "Error" else "Note.TLabel")
            if kind == "Error":
                self.status.configure(foreground="#b42318")
            self.status.grid()
        else:
            self.status.grid_remove()

    def set_model(self, model: Optional[LoadedModel], quant: Optional[str]) -> None:
        self._quants.clear()
        if model is None:
            self.combo.configure(values=[], state="disabled")
            self.quant_label.set("")
            return
        labels = []
        for v in model.variants:
            label = f"{v.quant}  ·  {fmt_gb(v.size)}" if v.size else f"{v.quant}  ·  {t('gui.estimated')}"
            self._quants[label] = v.quant
            labels.append(label)
        self.combo.configure(values=labels, state="readonly" if len(labels) > 1 else "disabled")
        current = next((lbl for lbl, q in self._quants.items() if q == quant), labels[0])
        self.quant_label.set(current)
