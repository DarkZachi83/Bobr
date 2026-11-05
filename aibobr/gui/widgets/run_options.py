"""Najczęściej zmieniane ustawienia w zakładce Sprawdź (etap 13b): kontekst i KV cache,
przy modelach obrazów – rozdzielczość i liczba obrazów naraz.

Wcześniej były tylko w trybie Zaawansowanym; „czy pójdzie przy 32k?” to najczęstsze pytanie,
więc są obok kwantyzacji (decyzja autora: zdublowane – Zaawansowane mają własne).
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import ttk
from typing import Callable

from aibobr.cli.cli import parse_context, parse_resolution
from aibobr.domain.enums import KvCacheType
from aibobr.domain.models import RunSettings
from aibobr.i18n import fmt_context, t

CONTEXTS = ("4k", "8k", "16k", "32k", "64k", "128k")


class RunOptions(ttk.Frame):
    def __init__(self, master, on_change: Callable[[], None]):
        super().__init__(master, style="Inner.TFrame")
        self.on_change = on_change
        self.kind = "text"
        self.context = tk.StringVar(value="8k")
        self.kv = tk.StringVar(value="f16")
        self.resolution = tk.StringVar()
        self.batch = tk.StringVar(value="1")
        self._pending = None
        self.widgets: dict[str, ttk.Widget] = {}
        for var in (self.context, self.kv, self.resolution, self.batch):
            var.trace_add("write", lambda *_: self._changed())
        self._build()
        self.bind("<Destroy>", self._cancel, add="+")

    def _cancel(self, event=None) -> None:
        if self._pending and (event is None or event.widget is self):
            try:
                self.after_cancel(self._pending)
            except Exception:
                pass
            self._pending = None

    def _build(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self.widgets.clear()
        if self.kind == "image":
            ttk.Label(self, text=t("gui.resolution"), style="Muted.TLabel").pack(side="left")
            e = ttk.Entry(self, textvariable=self.resolution, width=11)
            e.pack(side="left", padx=(6, 12))
            self.widgets["resolution"] = e
            ttk.Label(self, text=t("gui.batch"), style="Muted.TLabel").pack(side="left")
            e = ttk.Entry(self, textvariable=self.batch, width=4)
            e.pack(side="left", padx=(6, 0))
            self.widgets["batch"] = e
            return
        ttk.Label(self, text=t("gui.context"), style="Muted.TLabel").pack(side="left")
        c = ttk.Combobox(self, textvariable=self.context, values=CONTEXTS, width=7)
        c.pack(side="left", padx=(6, 12))
        c.bind("<<ComboboxSelected>>", lambda e: self.on_change())
        self.widgets["context"] = c
        ttk.Label(self, text=t("gui.kv_cache"), style="Muted.TLabel").pack(side="left")
        k = ttk.Combobox(self, textvariable=self.kv, values=[k.value for k in KvCacheType],
                         state="readonly", width=6)
        k.pack(side="left", padx=(6, 0))
        self.widgets["kv"] = k

    def _changed(self) -> None:
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(350, self.on_change)

    def set_kind(self, kind: str) -> None:
        if kind != self.kind:
            self.kind = kind
            self._build()

    def set_defaults(self, d: RunSettings) -> None:
        self.context.set(fmt_context(d.context_tokens))
        self.kv.set(d.kv_cache_type.value)

    def apply(self, settings: RunSettings) -> RunSettings:
        """Ustawienia z pól na bazie domyślnych; ValueError(lista pól), gdy coś jest błędne."""
        invalid, out = [], {}
        if self.kind == "image":
            res = None
            if self.resolution.get().strip():
                try:
                    res = parse_resolution(self.resolution.get())
                except ValueError:
                    invalid.append("resolution")
            try:
                batch = max(1, int(self.batch.get().strip() or "1"))
            except ValueError:
                invalid.append("batch")
                batch = 1
            out = {"image_width": res[0] if res else None, "image_height": res[1] if res else None,
                   "image_batch": batch}
        else:
            try:
                out["context_tokens"] = parse_context(self.context.get())
            except ValueError:
                invalid.append("context")
            out["kv_cache_type"] = KvCacheType(self.kv.get())
        for key, widget in self.widgets.items():
            widget.configure(style="Invalid.TEntry" if key in invalid and key != "context"
                             else ("TCombobox" if isinstance(widget, ttk.Combobox) else "TEntry"))
        if invalid:
            raise ValueError(invalid)
        return replace(settings, **out)

    def export_state(self) -> dict:
        return {"context": self.context.get(), "kv": self.kv.get(), "resolution": self.resolution.get(),
                "batch": self.batch.get(), "kind": self.kind}

    def import_state(self, state: dict) -> None:
        self.set_kind(state.get("kind", "text"))
        for key in ("context", "kv", "resolution", "batch"):
            getattr(self, key).set(state.get(key, getattr(self, key).get()))

    def apply_saved(self, item) -> None:
        """Ustawienia zapisanego modelu (Statystyka → Otwórz)."""
        if item.context:
            self.context.set(fmt_context(item.context))
        if item.kv_cache:
            self.kv.set(item.kv_cache)
        custom = not item.native_resolution and item.resolution
        self.resolution.set(f"{item.resolution[0]}x{item.resolution[1]}" if custom else "")
        self.batch.set(str(item.batch or 1))
