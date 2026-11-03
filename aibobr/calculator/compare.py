"""Porównanie zapisanych modeli na tym komputerze (etap 11b, SPEC §5b).

Szybkości nie mierzymy – ustalamy KLASĘ SZYBKOŚCI z werdyktu (całość w karcie › prawie
całość › karta + RAM › sam procesor) i w obrębie klasy porównujemy, ile danych model czyta
na krok: przy modelach językowych to wagi czytane na token (MoE – tylko aktywne), przy
obrazach część główna razem z liczbą pikseli. Wynik jest względny: „szybszy niż”, bez liczb.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from aibobr.calculator.quants import QuantTable
from aibobr.calculator.verdict import evaluate
from aibobr.domain.enums import FitLevel
from aibobr.domain.models import HardwareProfile, ImageNeed, ModelSpec, RunSettings, Verdict

# Klasy szybkości – od najszybszej
IN_VRAM, MOSTLY_VRAM, SPLIT, CPU, WONT_RUN = range(5)
MOSTLY = 0.9                      # jak „prawie całość w karcie” w podpowiedziach
CONTEXT_STEP = 1024
IMAGE_SIDES = tuple(range(512, 2049, 64))


@dataclass(frozen=True)
class Comparison:
    level: Optional[FitLevel] = None
    speed_class: int = WONT_RUN
    cost: float = 0.0                  # bajty czytane na krok (niżej = szybciej w tej samej klasie)
    max_fit: Optional[int] = None      # największy kontekst (tokeny) albo bok obrazu (px) w całości w karcie
    best_quant: Optional[str] = None   # najwyższa jakość, która mieści się w całości w karcie
    margin: Optional[int] = None       # zapas w karcie (bajty), tylko przy zielonym werdykcie
    margin_pct: Optional[float] = None
    download: Optional[int] = None     # rozmiar do pobrania
    on_disk: Optional[str] = None      # gdzie już leży: "file", "Ollama", "LM Studio"…
    rank: Optional[int] = None         # 1 = najszybszy w grupie
    error: Optional[str] = None        # klucz i18n, gdy modelu nie dało się wczytać


def speed_class(v: Verdict) -> int:
    if v.level == FitLevel.RED:
        return WONT_RUN
    if v.capacity.cpu_only:
        return CPU
    if v.level == FitLevel.GREEN:
        return IN_VRAM
    if isinstance(v.need, ImageNeed):
        return SPLIT                   # obrazy: żółty = część główna dociągana z RAM
    return MOSTLY_VRAM if v.vram_fraction >= MOSTLY else SPLIT


def read_cost(spec: ModelSpec, v: Verdict) -> float:
    n = v.need
    if spec.image is not None:
        return n.main * (n.width * n.height / 1e6) * n.batch
    weights = n.weights
    if spec.is_moe and spec.params_active and spec.params_total:
        weights = weights * spec.params_active / spec.params_total
    return float(weights)


def _green(spec: ModelSpec, hw: HardwareProfile, settings: RunSettings, quants: QuantTable) -> bool:
    try:
        return evaluate(spec, hw, settings, quants).level == FitLevel.GREEN
    except Exception:                  # brak danych do obliczenia – traktujemy jak „nie mieści się”
        return False


def max_context(spec: ModelSpec, hw, settings: RunSettings, quants) -> Optional[int]:
    """Największy kontekst (wielokrotność 1024) z całością w karcie; None – nawet 1024 nie."""
    fits = lambda ctx: _green(spec, hw, replace(settings, context_tokens=ctx), quants)   # noqa: E731
    top = spec.max_context or 131072
    if not fits(CONTEXT_STEP):
        return None
    if fits(top):
        return top
    lo, hi = 1, top // CONTEXT_STEP              # w krokach po 1024: lo mieści się, hi nie
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid * CONTEXT_STEP) else (lo, mid)
    return lo * CONTEXT_STEP


def max_side(spec: ModelSpec, hw, settings: RunSettings, quants) -> Optional[int]:
    best = None
    for side in IMAGE_SIDES:
        if _green(spec, hw, replace(settings, image_width=side, image_height=side), quants):
            best = side
        elif best is not None:
            break
    return best


def best_quant(model, hw, settings: RunSettings, quants: QuantTable) -> Optional[str]:
    """Najwięcej bitów na wagę spośród wariantów, który mieści się w całości w karcie."""
    variants = [v.quant for v in model.variants if quants.bits(v.quant) is not None]
    if len(variants) < 2:                      # jeden plik, jedna kwantyzacja – nie ma z czego wybierać
        return None
    for quant in sorted(variants, key=lambda q: quants.bits(q), reverse=True):
        if _green(model.spec(quant), hw, settings, quants):
            return quant
    return None


def compare(model, quant: str, hw: HardwareProfile, settings: RunSettings,
            quants: QuantTable) -> Comparison:
    spec = model.spec(quant)
    v = evaluate(spec, hw, settings, quants)
    image = spec.image is not None
    margin = v.margin if v.level == FitLevel.GREEN else None
    usable = v.capacity.vram_usable
    return Comparison(
        level=v.level, speed_class=speed_class(v), cost=read_cost(spec, v),
        max_fit=(max_side if image else max_context)(spec, hw, settings, quants),
        best_quant=None if image else best_quant(model, hw, settings, quants),
        margin=margin, margin_pct=(margin / usable if margin is not None and usable else None),
        download=spec.weights_bytes or (v.need.weights if image else None))


def rank(comparisons: dict) -> dict:
    """Miejsca w grupie: 1 = najszybszy. Modele, które nie pójdą (albo z błędem), bez miejsca."""
    order = sorted((k for k, c in comparisons.items() if c.error is None and c.speed_class != WONT_RUN),
                   key=lambda k: (comparisons[k].speed_class, comparisons[k].cost))
    out = dict(comparisons)
    for n, key in enumerate(order, 1):
        out[key] = replace(out[key], rank=n)
    return out
