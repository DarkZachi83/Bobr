"""Werdykt: gdzie kończy się pasek modelu na pasku systemu (SPEC §6, §10)."""
from __future__ import annotations

from collections import defaultdict

from aibobr.calculator.memory import memory_need
from aibobr.calculator.quants import QuantTable
from aibobr.calculator.runtime_support import gpu_warnings
from aibobr.domain.enums import FitLevel
from aibobr.domain.models import (
    Capacity,
    GpuInfo,
    HardwareProfile,
    ModelSpec,
    RunSettings,
    Verdict,
)


class IncompleteProfileError(ValueError):
    """Nie znamy części sprzętu, więc werdyktu nie wydajemy (SPEC §10).

    GUI pokazuje wtedy „?” w brakujących polach i proponuje tryb Zaawansowany."""

    def __init__(self, missing: tuple[str, ...]):
        super().__init__(", ".join(missing))
        self.missing = missing


def _usable(gpu: GpuInfo, settings: RunSettings) -> int:
    return max(0, (gpu.vram_total or 0) - settings.vram_reserve)


def capacity(hw: HardwareProfile, settings: RunSettings) -> tuple[Capacity, tuple[str, ...]]:
    """Odcinki paska systemu i ostrzeżenia (klucze i18n)."""
    missing: list[str] = []
    warnings: list[str] = []
    if hw.ram_total is None:
        missing.append("ram_total")

    # Grafika zintegrowana nie ma własnej pamięci – to zwykły RAM, więc jej nie liczymy.
    discrete = [g for g in hw.gpus if g.selected and not g.integrated]
    for g in discrete:
        if g.vram_total is None:
            missing.append(f"gpu_vram:{g.name}")

    # Runtime łączy karty jednego producenta; mieszanki (NVIDIA + AMD) nie liczymy razem.
    # Bierzemy grupę z największą użytkową pamięcią.
    groups: dict[object, list[GpuInfo]] = defaultdict(list)
    for g in discrete:
        if g.vram_total is not None:
            groups[g.vendor].append(g)
    used: list[GpuInfo] = []
    if groups:
        used = max(groups.values(), key=lambda gs: sum(_usable(g, settings) for g in gs))
        if len(groups) > 1:
            warnings.append("warn.mixed_gpu_vendors")
    for g in used:
        warnings += [w for w in gpu_warnings(g) if w not in warnings]

    cap = Capacity(
        vram_usable=sum(_usable(g, settings) for g in used),
        ram_usable=max(0, (hw.ram_total or 0) - settings.ram_reserve),
        gpus_used=tuple(used),
        cpu_only=not used,
        missing=tuple(missing),
    )
    return cap, tuple(warnings)


def evaluate(
    model: ModelSpec,
    hw: HardwareProfile,
    settings: RunSettings,
    quants: QuantTable,
) -> Verdict:
    """Czysta funkcja: ten sam model, sprzęt i ustawienia → zawsze ten sam werdykt.

    Podpowiedzi „co możesz zrobić” wywołują ją wielokrotnie z wariantami."""
    cap, warnings = capacity(hw, settings)
    if not cap.complete:
        raise IncompleteProfileError(cap.missing)

    need = memory_need(model, settings, quants)
    total = need.total
    vram, ram = cap.vram_usable, cap.ram_usable

    in_vram = min(total, vram)
    in_ram = min(total - in_vram, ram)
    shortfall = max(0, total - vram - ram)

    if vram > 0 and total <= vram:
        level, margin, zone_size = FitLevel.GREEN, vram - total, vram
    elif total <= vram + ram:
        level, margin, zone_size = FitLevel.YELLOW, vram + ram - total, ram
    else:
        level, margin, zone_size = FitLevel.RED, None, 0

    low_margin = margin is not None and margin < max(
        settings.margin_ratio * zone_size, settings.margin_min
    )

    return Verdict(
        level=level,
        need=need,
        capacity=cap,
        in_vram=in_vram,
        in_ram=in_ram,
        shortfall=shortfall,
        margin=margin,
        low_margin=low_margin,
        context_exceeds_model=bool(
            model.max_context and settings.context_tokens > model.max_context
        ),
        warnings=warnings,
    )
