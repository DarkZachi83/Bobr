"""Modele danych: sprzęt, model językowy, ustawienia uruchomienia i wynik.

Wszystkie obiekty są niezmienne (frozen). Wariant „co by było, gdyby” (tryb
Zaawansowany, podpowiedzi) tworzymy przez dataclasses.replace(), więc obliczenia
pozostają czystymi funkcjami. Rozmiary pamięci zawsze w bajtach (zob. units.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from aibobr.domain.enums import FitLevel, GpuVendor, KvCacheType, ModelSource
from aibobr.domain.units import gib


# --------------------------------------------------------------------------- sprzęt

@dataclass(frozen=True)
class GpuInfo:
    """Jedna karta graficzna.

    vram_total = None oznacza „nie udało się odczytać” – to co innego niż 0
    (SPEC §10: nie wolno po cichu przyjąć 0 GB)."""

    name: str
    vendor: GpuVendor
    vram_total: Optional[int]
    vram_used: Optional[int] = None          # tylko informacyjnie, nie wpływa na werdykt
    integrated: bool = False                 # grafika zintegrowana: pamięć = RAM, nie liczymy
    selected: bool = True                    # odznaczenie w trybie Zaawansowanym
    compute_capability: Optional[tuple[int, int]] = None  # tylko NVIDIA


@dataclass(frozen=True)
class HardwareProfile:
    """Wynik skanu (albo ręcznie poprawione / fałszywe dane z --fake-hw)."""

    platform: str                            # "linux", "windows", "darwin", …
    cpu_name: Optional[str]
    ram_total: Optional[int]
    ram_used: Optional[int] = None           # tylko informacyjnie
    ram_description: Optional[str] = None    # np. "DDR4-3200", jeśli system to podaje
    gpus: tuple[GpuInfo, ...] = ()


# --------------------------------------------------------------------------- model

@dataclass(frozen=True)
class ModelSpec:
    """Konkretny model w konkretnej kwantyzacji (SPEC §4).

    Pola architektury (warstwy, głowice KV, wymiar głowicy) są opcjonalne: przy
    ręcznym wpisie na poziomie podstawowym ich nie ma i cache KV jest szacowany."""

    name: str
    source: ModelSource
    quant: str                               # np. "Q4_K_M", "F16"
    params_total: Optional[int] = None       # liczba parametrów (nie w miliardach)
    weights_bytes: Optional[int] = None      # rozmiar pliku GGUF, jeśli znany – ma pierwszeństwo
    n_layers: Optional[int] = None
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_context: Optional[int] = None
    n_experts: Optional[int] = None          # MoE: wszystkich ekspertów
    n_experts_used: Optional[int] = None     # MoE: aktywnych na token
    params_active: Optional[int] = None      # MoE: aktywnych parametrów na token
    origin: Optional[str] = None             # link HF albo ścieżka pliku – do raportu

    @property
    def has_architecture(self) -> bool:
        return None not in (self.n_layers, self.n_kv_heads, self.head_dim)

    @property
    def is_moe(self) -> bool:
        return bool(self.n_experts and self.n_experts > 1)


# --------------------------------------------------------------------------- ustawienia

@dataclass(frozen=True)
class RunSettings:
    """Parametry uruchomienia i progi (SPEC §3, §5, §6). Wartości domyślne = tryb Automatyczny."""

    context_tokens: int = 8192
    kv_cache_type: KvCacheType = KvCacheType.F16
    vram_reserve: int = gib(1.0)             # na każdą kartę
    ram_reserve: int = gib(2.0)              # dla systemu; Windows: 4 GB (ustawia config)
    margin_ratio: float = 0.10               # próg ostrzeżenia o małym zapasie
    margin_min: int = gib(1.0)               # …ale nie mniej niż tyle
    overhead: int = gib(0.5)                 # bufor obliczeń + runtime, do kalibracji (etap 8)


# --------------------------------------------------------------------------- wynik

@dataclass(frozen=True)
class MemoryNeed:
    """Ile pamięci potrzebuje model – trzy części paska modelu."""

    weights: int
    kv_cache: int
    overhead: int
    weights_estimated: bool                  # z bitów na wagę, nie z rozmiaru pliku
    kv_estimated: bool                       # bez danych architektury

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.overhead

    @property
    def is_estimate(self) -> bool:
        return self.weights_estimated or self.kv_estimated


@dataclass(frozen=True)
class Capacity:
    """Odcinki paska systemu: VRAM użytkowy (szybki) i RAM użytkowy (wolny)."""

    vram_usable: int
    ram_usable: int
    gpus_used: tuple[GpuInfo, ...]
    cpu_only: bool                           # brak osobnej karty branej pod uwagę
    missing: tuple[str, ...] = ()            # pola, których nie udało się odczytać

    @property
    def total(self) -> int:
        return self.vram_usable + self.ram_usable

    @property
    def complete(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class Verdict:
    """Werdykt dla jednej pary model + komputer."""

    level: FitLevel
    need: MemoryNeed
    capacity: Capacity
    in_vram: int                             # ile bajtów modelu trafia do VRAM
    in_ram: int                              # ile do RAM
    shortfall: int                           # ile brakuje (tylko RED, inaczej 0)
    margin: Optional[int]                    # zapas do granicy strefy (None dla RED)
    low_margin: bool
    context_exceeds_model: bool              # kontekst większy niż model obsługuje
    warnings: tuple[str, ...] = field(default=())  # klucze i18n, np. "warn.cuda_too_old"

    @property
    def vram_fraction(self) -> float:
        """Jaka część modelu jest w VRAM (0–1). Dla RED liczona względem całej potrzeby."""
        return self.in_vram / self.need.total if self.need.total else 0.0
