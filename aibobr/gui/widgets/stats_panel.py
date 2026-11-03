"""Zakładka „Statystyka” (etap 11a, SPEC §5b): zapisane modele w trzech rozwijanych grupach.

Każda grupa (tekst / obraz / dźwięk) to tabela do 10 modeli: nazwa, kwantyzacja, ustawienia,
data sprawdzenia, werdykt z dnia zapisu i sprzęt. Dwuklik albo „Otwórz” wczytuje model
w zakładce Sprawdź z zapisanymi ustawieniami; „Usuń” zwalnia miejsce na liście.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from aibobr.app.saved import KINDS, MAX_PER_KIND, SavedModel, SavedStore
from aibobr.domain.units import gib
from aibobr.gui.theme import VERDICT_COLORS
from aibobr.i18n import fmt_context, fmt_gb, t

COLUMNS = ("model", "quant", "settings", "date", "level", "need", "hardware")
WIDTHS = {"model": 200, "quant": 100, "settings": 160, "date": 130, "level": 115, "need": 80, "hardware": 240}


def settings_text(item: SavedModel) -> str:
    if item.kind == "image" and item.resolution:
        w, h = item.resolution
        text = f"{w}×{h}"
        if item.batch > 1:
            text += f" · {t('image.batch', n=item.batch)}"
        return text
    parts = [f"{t('cli.context')} {fmt_context(item.context or 0)}"]
    if item.kv_cache and item.kv_cache != "f16":
        parts.append(f"KV {item.kv_cache}")
    return " · ".join(parts)


class Section(ttk.Frame):
    """Nagłówek „▸ Modele językowe (3/10)” – klik rozwija tabelę."""

    def __init__(self, master, kind: str, on_open: Callable, on_remove: Callable):
        super().__init__(master, style="Panel.TFrame")
        self.kind, self.on_open, self.on_remove = kind, on_open, on_remove
        self.items: dict[str, SavedModel] = {}
        self.expanded = True
        self.toggle = ttk.Button(self, style="Toggle.TButton", command=self.flip)
        self.toggle.pack(fill="x", anchor="w")
        self.body = ttk.Frame(self, style="Inner.TFrame")
        self.empty = ttk.Label(self.body, text=t("stats.empty"), style="Muted.TLabel")
        self.tree = ttk.Treeview(self.body, columns=COLUMNS, show="headings", height=MAX_PER_KIND,
                                 selectmode="browse")
        for col in COLUMNS:
            self.tree.heading(col, text=t(f"stats.col.{col}"))
            self.tree.column(col, width=WIDTHS[col], minwidth=60, anchor="w", stretch=col in ("model", "hardware"))
        for level, color in VERDICT_COLORS.items():
            self.tree.tag_configure(level.value, foreground=color)
        self.tree.bind("<Double-1>", lambda e: self._open())
        buttons = ttk.Frame(self.body, style="Inner.TFrame")
        self.open_button = ttk.Button(buttons, text=t("stats.open"), command=self._open)
        self.open_button.pack(side="left")
        self.remove_button = ttk.Button(buttons, text=t("stats.remove"), command=self._remove)
        self.remove_button.pack(side="left", padx=6)
        self.buttons = buttons
        self.body.pack(fill="x", padx=8, pady=(2, 8))

    def flip(self) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self.body.pack(fill="x", padx=8, pady=(2, 8))
        else:
            self.body.pack_forget()
        self._title()

    def _title(self) -> None:
        arrow = "▾" if self.expanded else "▸"
        self.toggle.configure(text=f"{arrow} {t(f'stats.kind.{self.kind}')}  "
                                   f"({len(self.items)}/{MAX_PER_KIND})")

    def fill(self, items: list[SavedModel]) -> None:
        self.tree.delete(*self.tree.get_children())
        self.items = {}
        for item in items:
            iid = self.tree.insert("", "end", values=(
                item.name, item.quant, settings_text(item), item.saved_at,
                t(f"stats.level.{item.level}"), fmt_gb(gib(item.need_gib)), item.hardware),
                tags=(item.level,))
            self.items[iid] = item
        for w in (self.empty, self.tree, self.buttons):
            w.pack_forget()
        if items:
            self.tree.configure(height=max(1, len(items)))
            self.tree.pack(fill="x")
            self.buttons.pack(anchor="w", pady=(6, 0))
        else:
            self.empty.pack(anchor="w", pady=4)
        self._title()

    def selected(self):
        sel = self.tree.selection()
        return self.items.get(sel[0]) if sel else None

    def _open(self) -> None:
        item = self.selected()
        if item:
            self.on_open(item)

    def _remove(self) -> None:
        item = self.selected()
        if item:
            self.on_remove(item)


class StatsPanel(ttk.Frame):
    def __init__(self, master, store: SavedStore, on_open: Callable[[SavedModel], None]):
        super().__init__(master, padding=4)
        self.store, self.on_open = store, on_open
        ttk.Label(self, text=t("stats.title"), style="Title.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(self, text=t("stats.hint", max=MAX_PER_KIND), style="Muted.TLabel", wraplength=820).pack(anchor="w", pady=(0, 8))
        canvas_holder = ttk.Frame(self)
        canvas_holder.pack(fill="both", expand=True)
        self.sections = {k: Section(canvas_holder, k, on_open, self.remove) for k in KINDS}
        for section in self.sections.values():
            section.pack(fill="x", pady=(0, 8))
        self.refresh()

    def refresh(self) -> None:
        for kind, section in self.sections.items():
            section.fill(self.store.of_kind(kind))

    def remove(self, item: SavedModel) -> None:
        self.store.remove(item)
        self.refresh()
