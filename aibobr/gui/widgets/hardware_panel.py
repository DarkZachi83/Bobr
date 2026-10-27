"""Panel „Twój komputer” (SPEC §3, §10).

Automatyczny: podsumowanie skanu, liczy z ustawieniami domyślnymi.
Zaawansowany: te same dane w polach liczbowych (bez suwaków), wpisane ze skanu – żeby
poprawić tylko błędną wartość albo sprawdzić sprzęt, którego jeszcze nie ma – oraz
parametry uruchomienia (kontekst, cache KV, rezerwy, próg zapasu).
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import ttk
from typing import Callable, Optional

from aibobr.cli.cli import parse_context
from aibobr.domain.enums import GpuVendor, KvCacheType
from aibobr.domain.models import GpuInfo, HardwareProfile, RunSettings
from aibobr.domain.units import gib, to_gib
from aibobr.i18n import fmt_context, fmt_gb, fmt_number, t


def parse_number(text: str) -> float:
    """„15,3” albo „15.3” → 15.3; puste/błędne → ValueError."""
    value = float(str(text).strip().replace(",", ".").replace(" ", ""))
    if value < 0:
        raise ValueError(text)
    return value


def _num(value: Optional[float], decimals: int = 1) -> str:
    return "" if value is None else fmt_number(value, decimals).replace(" ", "")


class GpuRow:
    def __init__(self, gpu: Optional[GpuInfo], name: str, vram: str, selected: bool):
        self.gpu = gpu
        self.name = name
        self.shown = vram               # tekst wpisany ze skanu (zaokrąglony do 0,1 GB)
        self.vram = tk.StringVar(value=vram)
        self.selected = tk.BooleanVar(value=selected)
        self.entry: Optional[ttk.Entry] = None


class HardwarePanel(ttk.Frame):
    def __init__(self, master, on_change: Callable[[], None],
                 on_save_defaults: Callable[[dict], None]):
        super().__init__(master, style="Panel.TFrame")
        self.on_change = on_change
        self.on_save_defaults = on_save_defaults
        self.mode = tk.StringVar(value="auto")
        self.scan: Optional[HardwareProfile] = None
        self.platform = "linux"
        self.defaults = RunSettings()
        self.unsupported: Optional[str] = None
        self.rows: list[GpuRow] = []
        self.ram_shown = ""
        self.ram = tk.StringVar()
        self.context = tk.StringVar()
        self.kv = tk.StringVar(value="f16")
        self.vram_reserve = tk.StringVar()
        self.ram_reserve = tk.StringVar()
        self.margin = tk.StringVar()
        self.entries: dict[str, ttk.Entry] = {}
        self._pending = None
        for var in (self.ram, self.context, self.kv, self.vram_reserve, self.ram_reserve, self.margin):
            var.trace_add("write", lambda *_: self._changed())

        top = ttk.Frame(self, style="Inner.TFrame")
        top.pack(fill="x", padx=10, pady=(8, 4))
        ttk.Label(top, text=t("gui.computer"), style="Title.TLabel").pack(side="left")
        for value, key in (("advanced", "gui.mode.advanced"), ("auto", "gui.mode.auto")):
            ttk.Radiobutton(top, text=t(key), value=value, variable=self.mode,
                            style="Panel.TRadiobutton",
                            command=self._mode_changed).pack(side="right", padx=4)
        self.body = ttk.Frame(self, style="Inner.TFrame")
        self.body.pack(fill="x", padx=10, pady=(0, 8))

    # ------------------------------------------------------------------ dane wejściowe
    def set_scan(self, hw: Optional[HardwareProfile], defaults: RunSettings, platform: str,
                 unsupported: Optional[str] = None) -> None:
        self.scan, self.defaults, self.platform, self.unsupported = hw, defaults, platform, unsupported
        self.reset_fields()
        if unsupported:
            self.mode.set("advanced")
        self.render()

    def reset_fields(self) -> None:
        hw, d = self.scan, self.defaults
        self.ram_shown = _num(to_gib(hw.ram_total)) if hw and hw.ram_total else ""
        self.ram.set(self.ram_shown)
        self.rows = [GpuRow(g, g.name, _num(to_gib(g.vram_total)) if g.vram_total else "", g.selected)
                     for g in (hw.gpus if hw else ()) if not g.integrated]
        if not self.rows and (hw is None or self.unsupported):
            self.rows = [GpuRow(None, t("gui.gpu_n", n=1), "", True)]
        self.context.set(str(d.context_tokens))
        self.kv.set(d.kv_cache_type.value)
        self.vram_reserve.set(_num(to_gib(d.vram_reserve)))
        self.ram_reserve.set(_num(to_gib(d.ram_reserve)))
        self.margin.set(_num(d.margin_ratio * 100, 0))

    def export_state(self) -> dict:
        return {"mode": self.mode.get(), "ram": self.ram.get(), "context": self.context.get(),
                "kv": self.kv.get(), "vram_reserve": self.vram_reserve.get(),
                "ram_reserve": self.ram_reserve.get(), "margin": self.margin.get(),
                "rows": [(r.gpu, r.name, r.vram.get(), r.selected.get()) for r in self.rows]}

    def import_state(self, state: dict) -> None:
        for key in ("ram", "context", "kv", "vram_reserve", "ram_reserve", "margin"):
            getattr(self, key).set(state[key])
        self.rows = []
        for gpu, name, vram, selected in state["rows"]:
            row = GpuRow(gpu, name, vram, selected)
            row.shown = _num(to_gib(gpu.vram_total)) if gpu and gpu.vram_total else ""
            self.rows.append(row)
        self.mode.set(state["mode"])
        self.render()

    # ------------------------------------------------------------------ widok
    def _mode_changed(self) -> None:
        self.render()
        self.on_change()

    def _changed(self) -> None:
        if self.mode.get() != "advanced":
            return
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(350, self.on_change)

    def render(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        self.entries.clear()
        if self.mode.get() == "auto":
            self._render_auto()
        else:
            self._render_advanced()

    def _line(self, parent, label: str, value: str, row: int, style="Panel.TLabel") -> None:
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 12))
        ttk.Label(parent, text=value, style=style).grid(row=row, column=1, sticky="w")

    def _render_auto(self) -> None:
        hw = self.scan
        if hw is None:
            ttk.Label(self.body, text=t("gui.scanning") if not self.unsupported else
                      t("detection.unsupported_gui", platform=self.unsupported),
                      style="Muted.TLabel").pack(anchor="w")
            return
        grid = ttk.Frame(self.body, style="Inner.TFrame")
        grid.pack(fill="x")
        self._line(grid, t("gui.cpu"), hw.cpu_name or "?", 0)
        ram = fmt_gb(hw.ram_total)
        if hw.ram_used is not None:
            ram += "  " + t("gui.used_now", size=fmt_gb(hw.ram_used))
        if hw.ram_description:
            ram += f"  ({hw.ram_description})"
        self._line(grid, "RAM", ram, 1)
        if not hw.gpus:
            self._line(grid, t("gui.gpu"), t("gui.no_gpu"), 2)
        for i, g in enumerate(hw.gpus):
            if g.integrated:
                value = f"{g.name}  —  {t('gui.integrated')}"
            else:
                value = f"{g.name}  ·  VRAM {fmt_gb(g.vram_total)}"
            self._line(grid, t("gui.gpu") if i == 0 else "", value, 2 + i)

    def _entry(self, parent, var, key, row, col, width=8):
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=col, sticky="w", pady=2)
        self.entries[key] = e
        return e

    def _render_advanced(self) -> None:
        if self.unsupported:
            ttk.Label(self.body, text=t("detection.unsupported_gui", platform=self.unsupported),
                      style="Note.TLabel", wraplength=640).pack(anchor="w", pady=(0, 6))
        g = ttk.Frame(self.body, style="Inner.TFrame")
        g.pack(fill="x")
        r = 0
        ttk.Label(g, text="RAM", style="Muted.TLabel").grid(row=r, column=0, sticky="w")
        self._entry(g, self.ram, "ram", r, 1)
        ttk.Label(g, text="GB", style="Panel.TLabel").grid(row=r, column=2, sticky="w", padx=4)
        r += 1
        for i, row in enumerate(self.rows):
            ttk.Checkbutton(g, text=row.name, variable=row.selected, style="Panel.TCheckbutton",
                            command=self.on_change).grid(row=r, column=0, sticky="w")
            row.entry = self._entry(g, row.vram, f"gpu{i}", r, 1)
            row.vram.trace_add("write", lambda *_: self._changed())
            ttk.Label(g, text="GB VRAM", style="Panel.TLabel").grid(row=r, column=2, sticky="w", padx=4)
            r += 1
        ttk.Button(g, text=t("gui.add_gpu"), command=self._add_gpu).grid(row=r, column=0, sticky="w", pady=(2, 8))
        r += 1

        params = [
            ("context", self.context, t("gui.context"), t("gui.context_hint")),
            ("vram_reserve", self.vram_reserve, t("gui.vram_reserve"), t("gui.vram_reserve_hint")),
            ("ram_reserve", self.ram_reserve, t("gui.ram_reserve"), "GB"),
            ("margin", self.margin, t("gui.margin"), "%"),
        ]
        for key, var, label, unit in params:
            ttk.Label(g, text=label, style="Muted.TLabel").grid(row=r, column=0, sticky="w")
            self._entry(g, var, key, r, 1)
            ttk.Label(g, text=unit, style="Panel.TLabel").grid(row=r, column=2, sticky="w", padx=4)
            r += 1
            if key == "context":
                ttk.Label(g, text=t("gui.kv_cache"), style="Muted.TLabel").grid(row=r, column=0, sticky="w")
                ttk.Combobox(g, textvariable=self.kv, values=[k.value for k in KvCacheType],
                             state="readonly", width=7).grid(row=r, column=1, sticky="w", pady=2)
                r += 1
        buttons = ttk.Frame(self.body, style="Inner.TFrame")
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text=t("gui.save_defaults"), command=self._save).pack(side="left")
        ttk.Button(buttons, text=t("gui.reset_scan"),
                   command=lambda: (self.reset_fields(), self.render(), self.on_change())).pack(side="left", padx=6)

    def _add_gpu(self) -> None:
        self.rows.append(GpuRow(None, t("gui.gpu_n", n=len(self.rows) + 1), "", True))
        self.render()

    # ------------------------------------------------------------------ wynik
    def _mark(self, invalid: set[str]) -> None:
        for key, entry in self.entries.items():
            entry.configure(style="Invalid.TEntry" if key in invalid else "TEntry")

    def current(self) -> tuple[HardwareProfile, RunSettings]:
        """Sprzęt i ustawienia do obliczeń. ValueError(lista pól), gdy coś jest błędne."""
        if self.mode.get() == "auto":
            if self.scan is None:
                raise ValueError(["scan"])
            return self.scan, self.defaults
        invalid, values = set(), {}
        for key, var in (("ram", self.ram), ("vram_reserve", self.vram_reserve),
                         ("ram_reserve", self.ram_reserve), ("margin", self.margin)):
            try:
                values[key] = parse_number(var.get())
            except ValueError:
                invalid.add(key)
        try:
            values["context"] = parse_context(self.context.get())
        except ValueError:
            invalid.add("context")
        gpus = []
        vendor = next((r.gpu.vendor for r in self.rows if r.gpu), GpuVendor.OTHER)
        for i, row in enumerate(self.rows):
            try:
                # Pole niezmienione = dokładna wartość ze skanu, nie zaokrąglony napis
                # (inaczej samo przełączenie trybu zmieniałoby wynik).
                if row.gpu and row.gpu.vram_total and row.vram.get() == row.shown:
                    vram = row.gpu.vram_total
                else:
                    vram = gib(parse_number(row.vram.get()))
            except ValueError:
                if row.selected.get():
                    invalid.add(f"gpu{i}")
                vram = None
            base = row.gpu or GpuInfo(name=row.name, vendor=vendor, vram_total=None)
            gpus.append(replace(base, vram_total=vram, selected=row.selected.get()))
        self._mark(invalid)
        if invalid:
            raise ValueError(sorted(invalid))

        scan = self.scan or HardwareProfile(platform=self.platform, cpu_name=None, ram_total=None)
        ram = scan.ram_total if scan.ram_total and self.ram.get() == self.ram_shown \
            else gib(values["ram"])
        same_ram = scan.ram_total is not None and abs(scan.ram_total - ram) < gib(0.05)
        integrated = tuple(g for g in scan.gpus if g.integrated)
        hw = replace(scan, ram_total=ram, ram_used=scan.ram_used if same_ram else None,
                     gpus=tuple(gpus) + integrated)
        settings = replace(self.defaults, context_tokens=values["context"],
                           kv_cache_type=KvCacheType(self.kv.get()),
                           vram_reserve=gib(values["vram_reserve"]),
                           ram_reserve=gib(values["ram_reserve"]),
                           margin_ratio=values["margin"] / 100)
        return hw, settings

    def _save(self) -> None:
        try:
            _, s = self.current()
        except ValueError:
            return
        self.on_save_defaults({
            "default_context": s.context_tokens, "kv_cache_type": s.kv_cache_type.value,
            "vram_reserve_gb": round(to_gib(s.vram_reserve), 2),
            "ram_reserve_gb": round(to_gib(s.ram_reserve), 2),
            "margin_ratio": round(s.margin_ratio, 3),
        })

    def show_advanced(self) -> None:
        self.mode.set("advanced")
        self._mode_changed()

    @staticmethod
    def context_label(settings: RunSettings) -> str:
        return fmt_context(settings.context_tokens)
