"""Model wpisany ręcznie (SPEC §4).

Poziom podstawowy: parametry (w miliardach), kwantyzacja, opcjonalnie nazwa.
Poziom zaawansowany: warstwy, głowice KV i wymiar głowicy – razem albo wcale,
bo z części tych liczb nie da się policzyć cache KV. Opcjonalnie dane MoE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aibobr.calculator.quants import QuantTable, normalize_quant
from aibobr.domain.enums import ModelSource
from aibobr.sources.base import ArchInfo, LoadedModel, SourceError, Variant


@dataclass(frozen=True)
class ManualInput:
    params_billion: float
    quant: str
    name: str = ""
    n_layers: Optional[int] = None
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_context: Optional[int] = None
    active_params_billion: Optional[float] = None   # MoE
    n_experts: Optional[int] = None
    n_experts_used: Optional[int] = None


def load_manual(data: ManualInput, quants: QuantTable) -> LoadedModel:
    """Sprawdza dane i tworzy LoadedModel z jednym wariantem. Błędy → SourceError
    z kluczem "manual.*", żeby GUI mogło podświetlić konkretne pole."""
    if not data.params_billion or data.params_billion <= 0:
        raise SourceError("manual.params_required")
    quant = normalize_quant(data.quant)
    if quants.bits(quant) is None:
        raise SourceError("manual.unknown_quant", quant)

    arch_fields = (data.n_layers, data.n_kv_heads, data.head_dim)
    given = [v for v in arch_fields if v is not None]
    if given and len(given) != len(arch_fields):
        raise SourceError("manual.incomplete_architecture")
    if any(v <= 0 for v in given) or (data.max_context is not None and data.max_context <= 0):
        raise SourceError("manual.not_positive")

    total = int(data.params_billion * 1e9)
    active = None
    if data.active_params_billion is not None:
        active = int(data.active_params_billion * 1e9)
        if not 0 < active <= total:
            raise SourceError("manual.active_params_range")

    arch = ArchInfo(
        params_total=total, n_layers=data.n_layers, n_kv_heads=data.n_kv_heads,
        head_dim=data.head_dim, max_context=data.max_context,
        n_experts=data.n_experts if data.n_experts and data.n_experts > 1 else None,
        n_experts_used=data.n_experts_used, params_active=active,
    )
    name = data.name.strip() or f"{data.params_billion:g}B"
    return LoadedModel(name=name, source=ModelSource.MANUAL, arch=arch,
                       variants=(Variant(quant, None),), preferred_quant=quant)
