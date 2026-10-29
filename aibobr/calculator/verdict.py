"""Werdykt: gdzie kończy się pasek modelu na pasku systemu (SPEC §6, §10)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from aibobr.calculator.image import image_need, tiled
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
    if model.image is not None:
        return evaluate_image(model, cap, warnings, settings)

    need = memory_need(model, settings, quants, gpu=not cap.cpu_only)
    total = need.total
    vram, ram = cap.vram_usable, cap.ram_usable

    # Tablica słownika (need.embd_in_ram) zawsze zostaje w RAM; reszta wypełnia najpierw VRAM.
    in_vram = min(need.vram_part, vram)
    rest = total - in_vram
    in_ram = min(rest, ram)
    shortfall = max(0, rest - ram)

    if vram > 0 and need.vram_part <= vram and rest <= ram:
        level, margin, zone_size = FitLevel.GREEN, vram - need.vram_part, vram
    elif rest <= ram:
        level, margin, zone_size = FitLevel.YELLOW, ram - rest, ram
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


def evaluate_image(model: ModelSpec, cap: Capacity, warnings: tuple[str, ...],
                   settings: RunSettings) -> Verdict:
    """Modele obrazów (SPEC §5a): części wędrują między RAM a kartą etapami.

    🟢 całość naraz w karcie · 🟢 mieści się etapami · 🟡 część główna dociągana z RAM
    (wolniej) · 🔴 nie mieszczą się nawet karta i RAM razem."""
    need = image_need(model, settings)
    vram, ram = cap.vram_usable, cap.ram_usable
    # Enkoder większy od karty, ale mieszczący się w RAM, liczy procesor – raz na opis.
    if vram and need.stages["encode"] > vram and need.encoders <= ram:
        need = replace(need, encoders_on_cpu=True)
    # Zwykłe dekodowanie VAE się nie mieści – programy same przechodzą na kafelki.
    if vram and need.stages["decode"] > vram:
        need = tiled(model, need)
    peak, all_at_once = need.peak, False
    working = peak - need.main if need.peak_stage == "sample" else peak   # obliczenia bez wag

    if cap.cpu_only:
        fits = need.weights + max(need.main_compute, need.vae_compute) + need.overhead <= ram
        level = FitLevel.YELLOW if fits else FitLevel.RED
        margin = ram - need.total if fits else None
        zone = ram
    elif peak <= vram and need.weights <= vram + ram:
        level, margin, zone = FitLevel.GREEN, vram - peak, vram
        all_at_once = need.all_at_once <= vram
    elif working <= vram and need.weights <= vram + ram:
        level, margin, zone = FitLevel.YELLOW, vram + ram - need.weights - working, ram
    else:
        level, margin, zone = FitLevel.RED, None, 0

    in_vram = min(peak, vram)
    shortfall = 0
    if level == FitLevel.RED:
        shortfall = max(need.weights + working - vram - ram, working - vram, 0) if vram \
            else max(need.total - ram, 0)
    low_margin = margin is not None and margin < max(settings.margin_ratio * zone, settings.margin_min)
    return Verdict(level=level, need=need, capacity=cap, in_vram=in_vram,
                   in_ram=max(0, peak - in_vram), shortfall=shortfall, margin=margin,
                   low_margin=low_margin, context_exceeds_model=False, warnings=warnings,
                   all_at_once=all_at_once)
