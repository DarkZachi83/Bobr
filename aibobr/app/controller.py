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
        self.last_scan = None                  # ostatnie „Modele na dysku…” w tej sesji

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
        if folder:
            return scanner.scan_folder(Path(folder))
        self.last_scan = scanner.scan_known(sys.platform)   # do porównania: „masz już na dysku”
        return self.last_scan

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
            hardware=AppController.hardware_text(hw))


    @staticmethod
    def hardware_text(hw: HardwareProfile) -> str:
        """„RTX 5050 Laptop 8 GB · RAM 15 GB” – krótki opis sprzętu do tabeli."""
        gpus = [g for g in hw.gpus if g.selected and not g.integrated]
        parts = [f"{_short_gpu(g.name)} {to_gib(g.vram_total or 0):.0f} GB" for g in gpus] or ["CPU"]
        if hw.ram_total:
            parts.append(f"RAM {to_gib(hw.ram_total):.0f} GB")
        return " · ".join(parts)

    @staticmethod
    def item_settings(item, base: RunSettings) -> RunSettings:
        """Ustawienia zapisanego modelu na tle bieżących (rezerwy, próg zapasu – z okna)."""
        s = base
        if item.context:
            s = replace(s, context_tokens=item.context)
        if item.kv_cache:
            s = replace(s, kv_cache_type=KvCacheType(item.kv_cache))
        if item.kind == "image":
            custom = item.resolution if not item.native_resolution else None
            s = replace(s, image_width=custom[0] if custom else None,
                        image_height=custom[1] if custom else None, image_batch=item.batch or 1)
        return s

    def _on_disk(self, target: str, size: Optional[int]) -> Optional[str]:
        """Gdzie model już leży: plik wskazany wprost, Ollama, albo – po „Modele na dysku…” –
        plik o tym samym rozmiarze co wariant (rozmiar z HF = rozmiar pobranego pliku)."""
        from aibobr.sources.factory import source_kind
        kind = source_kind(target)
        if kind == "ollama":
            return "Ollama"
        if kind == "file":
            return "file"
        if size and self.last_scan:
            for m in self.last_scan.models:
                if m.size == size:
                    return m.app
        return None

    def compare_saved(self, items, hw: HardwareProfile, base: RunSettings, progress=None) -> dict:
        """Porównanie zapisanych modeli na tym komputerze: {klucz wpisu: Comparison}.
        Modele z Hugging Face idą z pamięci podręcznej (7 dni), pliki – z nagłówka."""
        from aibobr.calculator.compare import Comparison, compare, rank
        from aibobr.sources.base import SourceError
        out = {}
        for n, item in enumerate(items):
            if progress:
                progress(n, len(items), item.name)
            try:
                model = self.load_model(item.target)
                names = [v.quant for v in model.variants]
                quant = item.quant if item.quant in names else model.default_variant().quant
                c = compare(model, quant, hw, self.item_settings(item, base), self.quants)
                out[item.key] = replace(c, on_disk=self._on_disk(item.target, c.download))
            except SourceError as e:
                out[item.key] = Comparison(error=e.code)
            except Exception as e:                   # jeden zepsuty wpis nie psuje porównania
                out[item.key] = Comparison(error="stats.error_other")
        ranked = {}
        for kind in {i.kind for i in items}:
            ranked.update(rank({i.key: out[i.key] for i in items if i.kind == kind}))
        return ranked


    def fetch_hf_stats(self, items, today=None) -> dict:
        """Dane z Hugging Face dla zapisanych modeli z HF (inne – pomijamy). Zapisuje je w liście.
        {klucz: dane} albo {klucz: {"error": kod}} dla repozytorium, którego nie dało się odczytać."""
        import datetime
        from aibobr.sources.base import SourceError
        from aibobr.sources.huggingface import hf_stats, parse_reference
        from aibobr.sources.factory import OLLAMA_PREFIX
        client, out = self.http(), {}
        day = (today or datetime.date.today()).isoformat()
        for item in items:
            target = item.target
            if target.startswith(OLLAMA_PREFIX + "hf.co/"):          # ollama:hf.co/<repo>:Q4_K_M
                target = target[len(OLLAMA_PREFIX):]
            ref = parse_reference(target) if not target.startswith(OLLAMA_PREFIX) else None
            if ref is None or Path(item.target).exists():
                continue
            try:
                out[item.key] = {**hf_stats(client, ref.repo_id), "fetched": day}
            except SourceError as e:
                out[item.key] = {"error": e.code, "fetched": day}
        if self.saved and out:
            self.saved.set_hf({k: v for k, v in out.items() if "error" not in v})
        return out


def _short_gpu(name: str) -> str:
    """„NVIDIA GeForce RTX 5050 Laptop GPU” → „RTX 5050 Laptop” – do wąskiej kolumny tabeli."""
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD ", "Intel(R) ", "Intel "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(" Laptop GPU", " Laptop").replace(" GPU", "").strip()
