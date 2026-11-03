"""Zakładka „Statystyka” (etap 11, SPEC §5b): zapisane modele w trzech rozwijanych grupach.

Każda grupa (tekst / obraz / dźwięk) to tabela do 10 modeli. Kolumny porównania (etap 11b)
liczą się na bieżąco dla tego komputera: werdykt dziś, klasa szybkości z miejscem w grupie
(★ najszybszy, ▼ najwolniejszy), największy kontekst albo rozdzielczość w całości w karcie,
najlepsza kwantyzacja w karcie, zapas, rozmiar i to, czy model już jest na dysku. Pod
tabelą: szczegóły zaznaczonego modelu (z dnia zapisu) i podsumowanie grupy.
Dwuklik albo „Otwórz” wczytuje model w zakładce Sprawdź; „Usuń” zwalnia miejsce.
"""
from __future__ import annotations

from tkinter import ttk
from typing import Callable, Optional

from aibobr.app.saved import KINDS, MAX_PER_KIND, SavedModel, SavedStore
from aibobr.calculator.compare import Comparison
from aibobr.domain.units import gib
from aibobr.gui.theme import COLORS, VERDICT_COLORS
from aibobr.i18n import fmt_context, fmt_count, fmt_gb, t

COLUMNS = ("model", "quant", "settings", "today", "speed", "maxfit", "bestq", "margin", "size", "disk",
           "downloads")
WIDTHS = {"model": 150, "quant": 75, "settings": 105, "today": 110, "speed": 115, "maxfit": 95,
          "bestq": 80, "margin": 95, "size": 70, "disk": 80, "downloads": 85}
DASH = "–"


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


def speed_text(c: Comparison, ranked: int) -> str:
    label = t(f"stats.speed.{c.speed_class}")
    if c.rank is None:
        return label
    if ranked >= 2 and c.rank == 1:
        return f"★ {label}"
    if ranked >= 2 and c.rank == ranked:
        return f"▼ {label}"
    return f"{c.rank}. {label}"


def maxfit_text(item: SavedModel, c: Comparison) -> str:
    if c.max_fit is None:
        return DASH
    return f"{c.max_fit} px" if item.kind == "image" else fmt_context(c.max_fit)


def hf_text(item: SavedModel) -> str:
    hf = item.hf or {}
    if not hf:
        return ""
    parts = []
    if hf.get("downloads") is not None:
        parts.append(t("stats.hf.downloads", n=fmt_count(hf["downloads"])))
    if hf.get("likes") is not None:
        parts.append(t("stats.hf.likes", n=fmt_count(hf["likes"])))
    if hf.get("updated"):
        parts.append(t("stats.hf.updated", date=hf["updated"]))
    if hf.get("license"):
        text = t("stats.hf.license", license=hf["license"])
        if hf.get("license_link"):
            text += f" ({hf['license_link']})"
        parts.append(text)
    elif hf.get("license_other"):
        parts.append(t("stats.hf.license_other"))
    if hf.get("base_model"):
        parts.append(t("stats.hf.base", base=hf["base_model"]))
    text = "Hugging Face: " + " · ".join(parts) + "."
    if hf.get("noncommercial"):
        text += " ⚠ " + t("stats.hf.noncommercial")
    if hf.get("gated"):
        text += " ⚠ " + t("stats.hf.gated")
    return text


def row_values(item: SavedModel, c: Optional[Comparison], ranked: int) -> tuple:
    base = (item.name, item.quant, settings_text(item))
    downloads = fmt_count((item.hf or {}).get("downloads"))
    if c is None:                                         # porównanie jeszcze się liczy
        return base + ("…",) + ("",) * 6 + (downloads,)
    if c.error:
        return base + (t("stats.error"),) + ("",) * 6 + (downloads,)
    today = t(f"stats.level.{c.level.value}") + (" •" if c.level.value != item.level else "")
    margin = DASH if c.margin is None else f"{fmt_gb(c.margin)} ({c.margin_pct * 100:.0f}%)"
    disk = {None: "", "file": t("stats.disk.file")}.get(c.on_disk, c.on_disk)
    return base + (today, speed_text(c, ranked), maxfit_text(item, c), c.best_quant or DASH,
                   margin, fmt_gb(c.download) if c.download else DASH, disk, downloads)


class Section(ttk.Frame):
    """Nagłówek „▾ Modele językowe (3/10)” – klik zwija i rozwija tabelę."""

    def __init__(self, master, kind: str, on_open: Callable, on_remove: Callable):
        super().__init__(master, style="Panel.TFrame")
        self.kind, self.on_open, self.on_remove = kind, on_open, on_remove
        self.items: dict[str, SavedModel] = {}
        self.comparisons: dict = {}
        self.expanded = True
        self.toggle = ttk.Button(self, style="Toggle.TButton", command=self.flip)
        self.toggle.pack(fill="x", anchor="w")
        self.body = ttk.Frame(self, style="Inner.TFrame")
        self.empty = ttk.Label(self.body, text=t("stats.empty"), style="Muted.TLabel")
        self.tree = ttk.Treeview(self.body, columns=COLUMNS, show="headings", height=MAX_PER_KIND,
                                 selectmode="browse")
        for col in COLUMNS:
            name = f"stats.col.{col}.image" if kind == "image" and col == "maxfit" else f"stats.col.{col}"
            self.tree.heading(col, text=t(name))
            self.tree.column(col, width=WIDTHS[col], minwidth=50, anchor="w", stretch=col == "model")
        for level, color in VERDICT_COLORS.items():
            self.tree.tag_configure(level.value, foreground=color)
        self.tree.tag_configure("error", foreground=COLORS["muted"])
        self.tree.bind("<Double-1>", lambda e: self._open())
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._details())
        self.detail = ttk.Label(self.body, text="", style="Muted.TLabel", wraplength=960)
        self.summary = ttk.Label(self.body, text="", style="Panel.TLabel", wraplength=960)
        buttons = ttk.Frame(self.body, style="Inner.TFrame")
        ttk.Button(buttons, text=t("stats.open"), command=self._open).pack(side="left")
        ttk.Button(buttons, text=t("stats.remove"), command=self._remove).pack(side="left", padx=6)
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

    def fill(self, items: list[SavedModel], comparisons: Optional[dict] = None) -> None:
        self.comparisons = comparisons or {}
        self.tree.delete(*self.tree.get_children())
        self.items = {}
        ranked = sum(1 for i in items if self._cmp(i) and self._cmp(i).rank)
        for item in items:
            c = self._cmp(item)
            tag = "error" if c and c.error else (c.level.value if c and c.level else item.level)
            iid = self.tree.insert("", "end", values=row_values(item, c, ranked), tags=(tag,))
            self.items[iid] = item
        for w in (self.empty, self.tree, self.summary, self.detail, self.buttons):
            w.pack_forget()
        if items:
            self.tree.configure(height=max(1, len(items)))
            self.tree.pack(fill="x")
            self.summary.configure(text=self._summary(items))
            self.summary.pack(anchor="w", pady=(6, 0))
            self.buttons.pack(anchor="w", pady=(6, 0))
            self.detail.configure(text="")
            self.detail.pack(anchor="w", pady=(4, 0))
        else:
            self.empty.pack(anchor="w", pady=4)
        self._title()

    def _cmp(self, item: SavedModel) -> Optional[Comparison]:
        return self.comparisons.get(item.key)

    def _summary(self, items: list[SavedModel]) -> str:
        comps = [(i, self._cmp(i)) for i in items if self._cmp(i) and not self._cmp(i).error]
        if not comps:
            return ""
        count = {lvl: sum(1 for _, c in comps if c.level.value == lvl) for lvl in ("green", "yellow", "red")}
        total = sum(c.download or 0 for _, c in comps)
        parts = [t("stats.sum.levels", **count)]
        if total:
            parts.append(t("stats.sum.size", size=fmt_gb(total)))
        ranked = sorted(((c.rank, i) for i, c in comps if c.rank), key=lambda x: x[0])
        if len(ranked) >= 2:
            fastest = ranked[0][1]
            text = t("stats.sum.fastest", name=fastest.name)
            if fastest.hf and fastest.hf.get("downloads") is not None:
                text += " (" + t("stats.hf.downloads", n=fmt_count(fastest.hf["downloads"])) + ")"
            parts.append(text)
            parts.append(t("stats.sum.slowest", name=ranked[-1][1].name))
        popular = [i for i in items if i.hf and i.hf.get("downloads")]
        if len(popular) >= 2:
            top = max(popular, key=lambda i: i.hf["downloads"])
            parts.append(t("stats.sum.popular", name=top.name))
        return " · ".join(parts)

    def _details(self) -> None:
        item = self.selected()
        if not item:
            return
        text = t("stats.detail", date=item.saved_at, hardware=item.hardware,
                 level=t(f"stats.level.{item.level}"), need=fmt_gb(gib(item.need_gib)))
        c = self._cmp(item)
        if c and c.error:
            text += " " + t(c.error)
        elif c and c.level and c.level.value != item.level:
            text += " " + t("stats.detail_changed")
        hf = hf_text(item)
        if hf:
            text += "\n" + hf
        self.detail.configure(text=text)

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
    def __init__(self, master, store: SavedStore, on_open: Callable[[SavedModel], None],
                 on_compare: Optional[Callable[[], None]] = None,
                 on_hf: Optional[Callable[[], None]] = None):
        super().__init__(master, padding=4)
        self.store, self.on_open, self.on_compare = store, on_open, on_compare
        self.comparisons: dict = {}
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(4, 0))
        ttk.Label(top, text=t("stats.title"), font=("TkDefaultFont", 11, "bold")).pack(side="left")
        if on_compare:
            ttk.Button(top, text=t("stats.recompute"), command=on_compare).pack(side="right")
        if on_hf:
            self.hf_button = ttk.Button(top, text=t("stats.hf.button"), command=on_hf)
            self.hf_button.pack(side="right", padx=6)
        self.status = ttk.Label(self, text="", foreground=COLORS["muted"], wraplength=960)
        self.status.pack(anchor="w")
        ttk.Label(self, text=t("stats.hint", max=MAX_PER_KIND), foreground=COLORS["muted"],
                  wraplength=960).pack(anchor="w", pady=(0, 8))
        holder = ttk.Frame(self)
        holder.pack(fill="both", expand=True)
        self.sections = {k: Section(holder, k, on_open, self.remove) for k in KINDS}
        for section in self.sections.values():
            section.pack(fill="x", pady=(0, 8))
        self.refresh()

    def refresh(self) -> None:
        for kind, section in self.sections.items():
            section.fill(self.store.of_kind(kind), self.comparisons)

    def show(self, comparisons: dict, hardware: str) -> None:
        self.comparisons = comparisons
        self.status.configure(text=t("stats.for", hardware=hardware))
        self.refresh()

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def remove(self, item: SavedModel) -> None:
        self.store.remove(item)
        self.comparisons.pop(item.key, None)
        self.refresh()
