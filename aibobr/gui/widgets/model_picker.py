"""Sekcja „Model” (SPEC §4, etap 13b): link / plik .gguf, kwantyzacja, kontekst i KV cache
(RunOptions). Wpis ręczny – menu Plik; modele na dysku – zakładka Benchmark (etap 13c).

Komunikaty (wczytywanie, błędy) idą do paska stanu okna – on_status(tekst, rodzaj).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Optional

from aibobr.i18n import fmt_gb, t
from aibobr.sources.base import LoadedModel


class ModelPicker(ttk.Frame):
    def __init__(self, master, on_check: Callable[[str], None], on_quant: Callable[[str], None],
                 on_status: Optional[Callable[[str, str], None]] = None,
                 options: Optional[Callable[[tk.Widget], tk.Widget]] = None):
        super().__init__(master, style="Panel.TFrame")
        self.on_check, self.on_quant = on_check, on_quant
        self.on_status = on_status or (lambda text, kind: None)
        self.text = tk.StringVar()
        self.quant_label = tk.StringVar()
        self._quants: dict[str, str] = {}
        self.columnconfigure(0, weight=1)

        ttk.Label(self, text=t("gui.model"), style="Title.TLabel").grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        ttk.Label(self, text=t("cli.ask_model_hint"), style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", padx=10)

        line = ttk.Frame(self, style="Inner.TFrame")
        line.grid(row=2, column=0, sticky="we", padx=10, pady=6)
        line.columnconfigure(0, weight=1)
        self.entry = ttk.Entry(line, textvariable=self.text)
        self.entry.grid(row=0, column=0, sticky="we", padx=(0, 6))
        self.entry.bind("<Return>", lambda e: self._check())
        ttk.Button(line, text=t("gui.paste"), command=self.paste).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(line, text=t("gui.clear"), command=self.clear).grid(row=0, column=2, padx=(0, 4))
        ttk.Button(line, text=t("gui.browse"), command=self.browse).grid(row=0, column=3, padx=(0, 6))
        self.check_button = ttk.Button(line, text=t("gui.check"), style="Accent.TButton", command=self._check)
        self.check_button.grid(row=0, column=4)

        bottom = ttk.Frame(self, style="Inner.TFrame")
        bottom.grid(row=3, column=0, sticky="we", padx=10, pady=(0, 10))
        ttk.Label(bottom, text=t("gui.quant"), style="Muted.TLabel").pack(side="left")
        self.combo = ttk.Combobox(bottom, textvariable=self.quant_label, state="disabled", width=22)
        self.combo.pack(side="left", padx=(6, 12))
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.on_quant(self._quants[self.quant_label.get()]))
        self.options = options(bottom) if options else None
        if self.options is not None:
            self.options.pack(side="left")


    def _check(self) -> None:
        if self.text.get().strip():
            self.on_check(self.text.get().strip())

    def paste(self) -> None:
        try:
            text = self.clipboard_get().strip()
        except tk.TclError:
            text = ""
        if text:
            self.text.set(text)
            self.entry.icursor("end")
        else:
            self.on_status(t("gui.clipboard_empty"), "Muted")

    def clear(self) -> None:
        self.text.set("")
        self.entry.focus_set()

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
        self.on_status(text, kind)

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
