"""Kontroler: jedno miejsce, które łączy ustawienia, źródła modelu, detekcję i doradcę.

CLI i GUI rozmawiają tylko z nim. Bez tkintera, więc testy składają całą aplikację
(main.build_app) bez okna – jak w NIUCHACZU.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

from aibobr.app.config import ConfigStore, run_settings
from aibobr.calculator.quants import QuantTable
from aibobr.detection.base import Detector
from aibobr.domain.enums import GpuVendor, KvCacheType
from aibobr.domain.models import GpuInfo, HardwareProfile, ModelSpec, RunSettings, Verdict
from aibobr.domain.units import gib
from aibobr.knowledge import catalog as catalog_mod
from aibobr.knowledge.advisor import Advisor, Explanation
from aibobr.knowledge.catalog import Catalog
from aibobr.sources.base import LoadedModel
from aibobr.sources.cache import ModelCache
from aibobr.sources.factory import load_model
from aibobr.sources.http import HttpClient


@dataclass(frozen=True)
class Overrides:
    """Ręczne poprawki skanu (tryb Zaawansowany; w CLI: --ram, --vram)."""

    ram_gb: Optional[float] = None
    vram_gb: Sequence[float] = ()           # jedna liczba na kartę; zastępuje wykryte karty


class AppController:
    def __init__(self, config: ConfigStore, quants: QuantTable, detector: Detector,
                 cache: Optional[ModelCache] = None, catalog: Optional[Catalog] = None,
                 catalog_download: Optional[Path] = None):
        self.config = config
        self.settings = config.load()
        self.quants = quants
        self.detector = detector
        self.cache = cache
        self.catalog_download = catalog_download
        self.set_catalog(catalog or Catalog(updated="", models=()))

    def set_catalog(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.advisor = Advisor(self.quants, catalog.suggestion_specs())

    def update_catalog(self, client=None) -> bool:
        """Pobiera nowszy katalog z repozytorium projektu (SPEC §9). True = zaktualizowano."""
        if self.catalog_download is None:
            return False
        fresh, newer = catalog_mod.update(client or HttpClient(), self.catalog_download, self.catalog)
        if newer:
            self.set_catalog(fresh)
        return newer

    # ------------------------------------------------------------------ model
    def http(self) -> HttpClient:
        return HttpClient(token=self.settings.get("hf_token") or None)

    def load_model(self, text: str) -> LoadedModel:
        return load_model(text, self.http(), self.cache)

    # ------------------------------------------------------------------ sprzęt
    def scan(self) -> HardwareProfile:
        return self.detector.detect()          # PlatformUnsupported leci wyżej

    @staticmethod
    def apply_overrides(hw: Optional[HardwareProfile], o: Overrides, platform: str) -> HardwareProfile:
        """Poprawki na skanie albo profil od zera (gdy skanu nie ma, np. macOS)."""
        hw = hw or HardwareProfile(platform=platform, cpu_name=None, ram_total=None)
        if o.ram_gb is not None:
            hw = replace(hw, ram_total=gib(o.ram_gb), ram_used=None)
        if o.vram_gb:
            kept = tuple(g for g in hw.gpus if g.integrated)
            vendor = next((g.vendor for g in hw.gpus if not g.integrated), GpuVendor.OTHER)
            manual = tuple(GpuInfo(name=f"GPU {i + 1}", vendor=vendor, vram_total=gib(v))
                           for i, v in enumerate(o.vram_gb) if v > 0)
            hw = replace(hw, gpus=manual + kept)
        return hw

    # ------------------------------------------------------------------ obliczenia
    def run_settings(self, platform: str, context: Optional[int] = None,
                     kv_cache: Optional[str] = None) -> RunSettings:
        s = run_settings(self.settings, platform)
        if context:
            s = replace(s, context_tokens=context)
        if kv_cache:
            s = replace(s, kv_cache_type=KvCacheType(kv_cache))
        return s

    def explain(self, model: LoadedModel, spec: ModelSpec, hw: HardwareProfile,
                settings: RunSettings) -> tuple[Verdict, Explanation]:
        return self.advisor.explain(model, spec, hw, settings)
