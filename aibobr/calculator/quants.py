"""Tabela kwantyzacji (data/quants.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


def normalize_quant(name: str) -> str:
    """'q4_k_m', ' Q4_K_M ' → 'Q4_K_M'."""
    return name.strip().upper()


@dataclass(frozen=True)
class QuantTable:
    bits_per_weight: Mapping[str, float]

    @classmethod
    def load(cls, path: Path) -> "QuantTable":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        table = {normalize_quant(k): float(v) for k, v in data["bits_per_weight"].items()}
        return cls(table)

    def bits(self, quant: str) -> Optional[float]:
        return self.bits_per_weight.get(normalize_quant(quant))

    def names_by_size(self) -> list[str]:
        """Kwantyzacje od największej do najmniejszej – przyda się podpowiedziom."""
        return sorted(self.bits_per_weight, key=self.bits_per_weight.__getitem__, reverse=True)
