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


def render_bar(v: Verdict, palette: Palette = PLAIN, width: int = 50,
               bobr: bool = False) -> list[str]:
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

    c_w = col(need.weights)
    c_kv = col(need.weights + need.kv_cache) - c_w
    c_ov = c_need - c_w - c_kv
    color = getattr(palette, LEVEL_COLOR[v.level])
    if bobr:
        weights = palette.paint("B" * c_w, palette.bold_yellow)
    else:
        weights = palette.paint("#" * c_w, color)
    model = (weights + palette.paint("+" * c_kv, color) + palette.paint(":" * c_ov, color)
             + " " * (width - c_need))

    label_sys, label_model = t("cli.row.system"), t("cli.row.model")
    pad = max(len(label_sys), len(label_model))
    legend = t("cli.legend", vram=fmt_gb(vram), ram=fmt_gb(cap.ram_usable),
               weights=fmt_gb(need.weights), kv=fmt_gb(need.kv_cache),
               overhead=fmt_gb(need.overhead))
    return [
        f"  {label_sys:<{pad}} [{system}]",
        f"  {label_model:<{pad}} [{model}]",
        "  " + palette.paint(legend, palette.dim),
    ]
