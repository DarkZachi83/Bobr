"""Geometria paska pamięci (SPEC §6) – czyste obliczenia, bez tkintera.

Dwie warstwy na wspólnej skali w bajtach:
  system: VRAM (szybki) → RAM (wolny) → „brak pamięci” (tylko gdy model się nie mieści),
  model:  wagi → pamięć rozmowy (KV) → narzut.
Znaczniki: koniec VRAM i koniec całej pamięci (linie przerywane w GUI).
"""
from __future__ import annotations

from dataclasses import dataclass

from aibobr.domain.models import Verdict


@dataclass(frozen=True)
class Segment:
    kind: str          # vram, ram, missing | weights, kv, overhead, decoder
    x0: float
    x1: float
    value: int         # bajty – do podpisu

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass(frozen=True)
class BarLayout:
    system: tuple[Segment, ...]
    model: tuple[Segment, ...]
    markers: tuple[float, ...]      # x końca VRAM (jeśli jest) i końca całej pamięci
    scale: int                      # bajty odpowiadające całej szerokości


def bar_layout(v: Verdict, x0: float, width: float, headroom: float = 1.04) -> BarLayout:
    cap, need = v.capacity, v.need
    vram, ram, total = cap.vram_usable, cap.ram_usable, cap.total
    scale = int(max(total, need.total) * headroom) or 1

    def x(value: int) -> float:
        return x0 + width * value / scale

    system = []
    if vram:
        system.append(Segment("vram", x(0), x(vram), vram))
    if ram:
        system.append(Segment("ram", x(vram), x(total), ram))
    if need.total > total:
        system.append(Segment("missing", x(total), x(need.total), need.total - total))

    model, start = [], 0
    for kind, value in (("weights", need.weights_total), ("kv", need.kv_cache),
                        ("overhead", need.overhead), ("decoder", need.decoder)):
        if value:
            model.append(Segment(kind, x(start), x(start + value), value))
            start += value

    markers = tuple(m for m in ((x(vram) if vram else None), x(total)) if m is not None)
    return BarLayout(tuple(system), tuple(model), markers, scale)


@dataclass(frozen=True)
class ImageBarLayout:
    """Modele obrazów (SPEC §5a): system, szczyt najcięższego etapu i „wszystko naraz”."""
    system: tuple[Segment, ...]
    peak: tuple[Segment, ...]
    at_once: tuple[Segment, ...]
    markers: tuple[float, ...]
    scale: int


def image_bar_layout(v: Verdict, x0: float, width: float, headroom: float = 1.04) -> ImageBarLayout:
    cap, n = v.capacity, v.need
    vram, ram, total = cap.vram_usable, cap.ram_usable, cap.total
    scale = int(max(total, n.all_at_once, n.peak, n.weights) * headroom) or 1

    def x(value: int) -> float:
        return x0 + width * value / scale

    def row(items) -> tuple[Segment, ...]:
        out, start = [], 0
        for kind, value in items:
            if value:
                out.append(Segment(kind, x(start), x(start + value), value))
                start += value
        return tuple(out)

    system = []
    if vram:
        system.append(Segment("vram", x(0), x(vram), vram))
    if ram:
        system.append(Segment("ram", x(vram), x(total), ram))
    if n.weights > total:
        system.append(Segment("missing", x(total), x(n.weights), n.weights - total))

    stage = n.peak_stage
    weight_kind, weight, comp = {"sample": ("main", n.main, n.main_compute),
                                 "decode": ("vae", n.vae, n.vae_compute),
                                 "encode": ("encoders", n.encoders, 0)}[stage]
    peak = row([(weight_kind, weight), ("compute", comp), ("overhead", n.overhead)])
    at_once = row([("main", n.main), ("compute", max(n.main_compute, n.vae_compute)),
                   ("overhead", n.overhead), ("encoders", n.encoders), ("vae", n.vae)])
    markers = tuple(m for m in ((x(vram) if vram else None), x(total)) if m is not None)
    return ImageBarLayout(tuple(system), peak, at_once, markers, scale)
