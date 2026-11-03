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
from aibobr.domain.units import gib, to_gib
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
                 catalog_download: Optional[Path] = None, audio=None, images=None, saved=None):
        self.config = config
        self.settings = config.load()
        self.quants = quants
        self.detector = detector
        self.cache = cache
        self.catalog_download = catalog_download
        from aibobr.knowledge.audio import AudioCatalog
        self.audio = audio or AudioCatalog()
        from aibobr.knowledge.images import ImageCatalog
        self.images = images or ImageCatalog()
        self.set_catalog(catalog or Catalog(updated="", models=()))
        self.saved = saved                     # SavedStore – modele do porównania (etap 11a)

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
        model = load_model(text, self.http(), self.cache, self.ollama_host())
        model = self.images.annotate(model)         # model obrazów? (etap 10)
        return self.audio.annotate(model)          # model mowy / muzyki? (etap 9)

    def ollama_host(self) -> str:
        return str(self.settings.get("ollama_host") or "http://127.0.0.1:11434")

    def ollama_models(self) -> list[dict]:
        """Modele zainstalowane w Ollamie (tryb Zaawansowany) – SourceError, gdy nie działa."""
        from aibobr.sources.ollama import OllamaSource
        models = OllamaSource(HttpClient(timeout=10), self.ollama_host()).installed()
        return [m for m in models if "embed" not in str(m.get("name", "")).lower()]

    # ------------------------------------------------------------------ modele na dysku
    @property
    def scan_consent(self) -> bool:
        return bool(self.settings.get("scan_consent"))

    def set_scan_consent(self, value: bool) -> None:
        self.settings["scan_consent"] = value
        self.config.save_user({"scan_consent": value})

    def find_models(self, folder: Optional[Path] = None, stop=None, progress=None):
        """Modele na dysku: znane miejsca programów albo wskazany folder (etap 9a)."""
        import sys
        from aibobr.scan.scanner import Scanner
        scanner = Scanner(stop, progress, audio=self.audio)
        return scanner.scan_folder(Path(folder)) if folder else scanner.scan_known(sys.platform)

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

    # ------------------------------------------------------------------ porównanie (etap 11a)
    @staticmethod
    def saved_entry(target: str, spec: ModelSpec, verdict: Verdict, hw: HardwareProfile,
                    settings: RunSettings, now=None):
        """Wpis do zapisanych modeli z bieżącego wyniku."""
        import datetime
        from aibobr.app.saved import SavedModel
        kind = "audio" if spec.audio else "image" if spec.image else "text"
        need = verdict.need
        peak = need.peak if kind == "image" else need.vram_part
        gpus = [g for g in hw.gpus if g.selected and not g.integrated]
        parts = [f"{_short_gpu(g.name)} {to_gib(g.vram_total or 0):.0f} GB" for g in gpus] or ["CPU"]
        if hw.ram_total:
            parts.append(f"RAM {to_gib(hw.ram_total):.0f} GB")
        image = kind == "image"
        return SavedModel(
            kind=kind, name=spec.name, target=target, quant=spec.quant,
            saved_at=(now or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M"),
            level=verdict.level.value, need_gib=round(to_gib(peak), 2),
            context=None if image else settings.context_tokens,
            kv_cache=None if image else settings.kv_cache_type.value,
            resolution=(need.width, need.height) if image else None,
            native_resolution=not settings.image_width,
            batch=need.batch if image else 1,
            hardware=" · ".join(parts))


def _short_gpu(name: str) -> str:
    """„NVIDIA GeForce RTX 5050 Laptop GPU” → „RTX 5050 Laptop” – do wąskiej kolumny tabeli."""
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD ", "Intel(R) ", "Intel "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(" Laptop GPU", " Laptop").replace(" GPU", "").strip()
