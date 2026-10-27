"""Wyliczenia domeny."""
from __future__ import annotations

from enum import Enum


class FitLevel(Enum):
    """Strefa, w której kończy się pasek modelu (SPEC §6)."""

    GREEN = "green"    # całość w VRAM – pełna prędkość
    YELLOW = "yellow"  # część w RAM (albo komputer bez karty) – wolniej
    RED = "red"        # nie mieści się nawet w VRAM + RAM


class GpuVendor(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    OTHER = "other"


class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    GGUF_FILE = "gguf_file"
    MANUAL = "manual"


class KvCacheType(Enum):
    """Typ elementu cache KV. Wartość = nazwa jak w llama.cpp (--cache-type-k/-v)."""

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"

    @property
    def bytes_per_element(self) -> float:
        # Bloki llama.cpp: q8_0 = 34 B na 32 elementy, q4_0 = 18 B na 32 elementy.
        return {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}[self.value]
