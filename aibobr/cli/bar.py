"""Pasek pamięci w terminalu (SPEC §6) – wersja znakowa grafiki z GUI.

    System [==============---------------------------         ]
    Model  [############++:                                    ]

System: „=” VRAM (szybki), „-” RAM (wolny), „x” brakująca pamięć.
Model:  „#” wagi, „+” pamięć rozmowy (cache KV), „:” narzut.
Kolor paska modelu = kolor werdyktu. Bez kolorów (plik, NO_COLOR) – same znaki.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from aibobr.domain.enums import FitLevel
from aibobr.domain.models import Verdict
from aibobr.i18n import fmt_gb, t


@dataclass(frozen=True)
class Palette:
    green: str = ""
    yellow: str = ""
    red: str = ""
    bold_yellow: str = ""
    dim: str = ""
    reset: str = ""

    def paint(self, text: str, color: str) -> str:
        return f"{color}{text}{self.reset}" if color and text else text


PLAIN = Palette()
ANSI = Palette(green="\033[32m", yellow="\033[33m", red="\033[31m",
               bold_yellow="\033[1;33m", dim="\033[2m", reset="\033[0m")


def _enable_windows_ansi() -> bool:
    """Windows 10+: włącza obsługę kodów ANSI w konsoli (ENABLE_VIRTUAL_TERMINAL_PROCESSING)."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)              # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def palette_for(stream, force_plain: bool = False) -> Palette:
    if force_plain or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return PLAIN
    if not hasattr(stream, "isatty") or not stream.isatty():
        return PLAIN
    if sys.platform.startswith("win") and not _enable_windows_ansi():
        return PLAIN
    return ANSI


LEVEL_COLOR = {FitLevel.GREEN: "green", FitLevel.YELLOW: "yellow", FitLevel.RED: "red"}


def render_image_bar(v: Verdict, palette: Palette = PLAIN, width: int = 50,
                     bobr: bool = False) -> list[str]:
    """Modele obrazów (SPEC §5a): System · Szczyt (najcięższy etap) · Naraz (wszystkie części)."""
    cap, n = v.capacity, v.need
    vram, total = cap.vram_usable, cap.total
    scale = max(total, n.all_at_once, n.peak) or 1

    def col(x: int) -> int:
        return min(width, round(x * width / scale))

    c_vram, c_total = col(vram), col(total)
    c_miss = max(0, col(max(n.peak, n.weights)) - c_total) if v.level.name == "RED" else 0
    system = (palette.paint("=" * c_vram, palette.green) + palette.paint("-" * (c_total - c_vram), palette.yellow)
              + palette.paint("x" * c_miss, palette.red) + " " * (width - c_total - c_miss))
    color = getattr(palette, LEVEL_COLOR[v.level])
    stage = n.peak_stage
    weight, comp = {"sample": (n.main, n.main_compute), "decode": (n.vae, n.vae_compute),
                    "encode": (n.encoders, 0)}[stage]
    c_w = col(weight)
    c_c = col(weight + comp) - c_w
    c_o = col(n.peak) - c_w - c_c
    mark = "B" if bobr and v.level.name == "GREEN" else "#"
    peak = (palette.paint(mark * c_w, palette.bold_yellow if mark == "B" else color)
            + palette.paint("+" * c_c, color) + palette.paint(":" * c_o, color))
    peak += " " * max(0, width - (c_w + c_c + c_o))
    parts = [("#", n.main), ("+", max(n.main_compute, n.vae_compute)), (":", n.overhead),
             ("~", n.encoders), ("°", n.vae)]
    at_once, used = "", 0
    for ch, size in parts:
        c = col(used + size) - col(used)
        at_once += ch * c
        used += size
    at_once = palette.paint(at_once[:width], palette.dim) + " " * max(0, width - len(at_once))

    labels = [t("cli.row.system"), t("cli.row.peak"), t("cli.row.all")]
    pad = max(len(x) for x in labels)
    legend = t("cli.legend_image", vram=fmt_gb(vram), ram=fmt_gb(cap.ram_usable), main=fmt_gb(n.main),
               comp=fmt_gb(comp), overhead=fmt_gb(n.overhead),
               enc=fmt_gb(n.encoders), vae=fmt_gb(n.vae))
    return [f"  {labels[0]:<{pad}} [{system}]",
            f"  {labels[1]:<{pad}} [{peak}]",
            f"  {labels[2]:<{pad}} [{at_once}]",
            "  " + palette.paint(legend, palette.dim)]


def render_bar(v: Verdict, palette: Palette = PLAIN, width: int = 50,
               bobr: bool = False) -> list[str]:
    from aibobr.domain.models import ImageNeed
    if isinstance(v.need, ImageNeed):
        return render_image_bar(v, palette, width, bobr)
    cap, need = v.capacity, v.need
    vram, total = cap.vram_usable, cap.total
    scale = max(total, need.total) or 1

    def col(x: int) -> int:
        return min(width, round(x * width / scale))

    c_vram, c_total, c_need = col(vram), col(total), col(need.total)
    system = (palette.paint("=" * c_vram, palette.green)
              + palette.paint("-" * (c_total - c_vram), palette.yellow)
              + palette.paint("x" * max(0, c_need - c_total), palette.red)
              + " " * (width - max(c_total, c_need)))

    c_w = col(need.weights_total)
    c_kv = col(need.weights_total + need.kv_cache) - c_w
    c_dec = col(need.decoder) if need.decoder else 0
    c_ov = c_need - c_w - c_kv - c_dec
    color = getattr(palette, LEVEL_COLOR[v.level])
    if bobr:
        weights = palette.paint("B" * c_w, palette.bold_yellow)
    else:
        weights = palette.paint("#" * c_w, color)
    model = (weights + palette.paint("+" * c_kv, color) + palette.paint(":" * c_ov, color)
             + palette.paint("~" * c_dec, color) + " " * (width - c_need))

    label_sys, label_model = t("cli.row.system"), t("cli.row.model")
    pad = max(len(label_sys), len(label_model))
    legend = t("cli.legend", vram=fmt_gb(vram), ram=fmt_gb(cap.ram_usable),
               weights=fmt_gb(need.weights_total), kv=fmt_gb(need.kv_cache),
               overhead=fmt_gb(need.overhead))
    if need.decoder:
        legend += "  " + t("cli.legend_decoder", size=fmt_gb(need.decoder))
    return [
        f"  {label_sys:<{pad}} [{system}]",
        f"  {label_model:<{pad}} [{model}]",
        "  " + palette.paint(legend, palette.dim),
    ]
