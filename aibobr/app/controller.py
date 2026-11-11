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
        self._bench_db = None

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
        self.save_found(self.last_scan)
        return self.last_scan

    @property
    def found_path(self) -> Optional[Path]:
        return self.saved.path.parent / "found_models.json" if self.saved is not None else None

    def save_found(self, report) -> None:
        """Lista z zakładki Benchmark zostaje między uruchomieniami (etap 13c)."""
        import datetime
        if self.found_path is None:
            return
        from aibobr.app import found_store
        try:
            found_store.save(self.found_path, report, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        except OSError:
            pass

    def load_found(self):
        """(ScanReport, kiedy) z poprzedniego szukania albo (None, None)."""
        if self.found_path is None:
            return None, None
        from aibobr.app import found_store
        report, when = found_store.load(self.found_path)
        if report is not None and self.last_scan is None:
            self.last_scan = report                         # „na dysku” w Statystyce bez nowego skanu
        return report, when

    # ------------------------------------------------------------------ sprzęt
    def scan(self) -> HardwareProfile:
        return self.with_ollama_loaded(self.detector.detect())   # PlatformUnsupported leci wyżej

    def ollama_loaded_vram(self) -> int:
        """Ile pamięci kart zajmują modele wczytane teraz w Ollamie (np. 5 minut po benchmarku)."""
        from aibobr.sources.ollama import OllamaSource
        try:
            running = OllamaSource(HttpClient(timeout=2), self.ollama_host()).running()
        except Exception:                          # Ollama nie działa – nic nie jest wczytane
            return 0
        return sum(int(m.get("size_vram") or 0) for m in running)

    def with_ollama_loaded(self, hw: HardwareProfile, loaded: Optional[int] = None) -> HardwareProfile:
        """Odróżnia pamięć zajętą przez system od modelu z Ollamy, który się zwolni.

        Tylko przy jednej karcie z odczytem zajętości – przy kilku nie wiadomo, gdzie leży model."""
        cards = [g for g in hw.gpus if not g.integrated and g.vram_used]
        if len(cards) != 1:
            return hw
        loaded = self.ollama_loaded_vram() if loaded is None else loaded
        if not loaded:
            return hw
        card = cards[0]
        return replace(hw, gpus=tuple(replace(g, vram_loaded=min(loaded, g.vram_used)) if g is card else g
                                      for g in hw.gpus))

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


    # ------------------------------------------------------------------ benchmark (etap 12a)
    @property
    def bench_db(self):
        """benchmarks.sqlite obok zapisanych modeli (tworzony przy pierwszym pomiarze)."""
        if self._bench_db is None and self.saved is not None:
            from aibobr.app.bench_db import BenchDb
            self._bench_db = BenchDb(self.saved.path.parent / "benchmarks.sqlite")
            # tylko szczegółowe próbki telemetrii; domyślnie (None) bezterminowo
            self._bench_db.prune_telemetry(self.settings.get("telemetry_retention_days"))
        return self._bench_db

    @property
    def custom_suites(self):
        """Własne testy jakości – folder quality/ obok zapisanych modeli (edytor testów 0.11)."""
        from aibobr.quality.custom import CustomSuites
        db = self.bench_db
        base = self.saved.path.parent / "quality" if self.saved is not None else None
        return CustomSuites(base, used=db.quality_suite_used if db is not None else None)

    def bench_suite(self) -> dict:
        import json
        from aibobr import resources
        # resources.path – w .exe pliki leżą w katalogu PyInstallera, nie obok kodu
        return json.loads(resources.path("data", "benchmarks", "perf_v1.json").read_text(encoding="utf-8"))

    def bench_predict(self, targets, contexts, hw: HardwareProfile, base: RunSettings) -> dict:
        """Werdykt AI Bóbra dla każdej pary (model, kontekst) – czerwone pomijamy przed startem.
        {(target, ctx): "green"/"yellow"/"red"} albo None, gdy modelu nie dało się policzyć."""
        from aibobr.calculator.verdict import evaluate
        out = {}
        for target in targets:
            try:
                model = self.load_model(target)
                spec = model.spec()
            except Exception:
                spec = None
            for ctx in contexts:
                level = None
                if spec is not None:
                    try:
                        level = evaluate(spec, hw, replace(base, context_tokens=ctx), self.quants).level.value
                    except Exception:
                        level = None
                out[(target, ctx)] = level
        return out

    def quality_suite(self, suite_id: str) -> dict:
        import json
        from aibobr import resources
        return json.loads(resources.path("data", "benchmarks", f"{suite_id}.json").read_text(encoding="utf-8"))

    def run_quality(self, targets, tests, report, stop, *, attempts: int = 1, hw=None, bench=None,
                    sandbox=None, db=None, runner_kw=None) -> dict:
        """Moduł jakości 0.10: tests = [(id testu, język)], np. [("humanistyczny_v1", "pl"),
        ("scisly_v1", "pl"), ("scisly_v1", "en")]; kod Pythona tylko w zweryfikowanej piaskownicy."""
        from aibobr import __version__
        from aibobr.bench.ollama import OllamaBench
        from aibobr.quality import suites
        from aibobr.quality.runner import QualityRunner
        from aibobr.quality.sandbox import Sandbox
        from aibobr.sources.factory import OLLAMA_PREFIX
        bench = bench or OllamaBench(self.ollama_host())
        loaded = {}
        chosen = []
        for suite_id, lang in tests:
            if suite_id not in loaded:      # standardowy z data/quality, własny z folderu quality/ (edytor 0.11)
                loaded[suite_id] = (suites.load(suite_id) if suite_id in suites.STANDARD
                                    else self.custom_suites.load(suite_id))
            chosen.append((loaded[suite_id], lang))
        models = [t[len(OLLAMA_PREFIX):] if t.startswith(OLLAMA_PREFIX) else t for t in targets]
        info = {"app_version": __version__, "runtime_version": bench.version(),
                "hardware": self.hardware_text(hw) if hw is not None else None}
        runner = QualityRunner(bench, sandbox if sandbox is not None else Sandbox(),
                               db if db is not None else self.bench_db, info, **(runner_kw or {}))
        return runner.run(models, chosen, attempts, report, stop)

    def speed_pack(self) -> dict:
        """Benchmark prędkości 0.9: 15 zadań z pakietów polszczyzny i formatu (bez oceniania treści)."""
        from aibobr.bench import speed
        return speed.build_pack([self.quality_suite("pl_jezyk_v1"), self.quality_suite("en_format_v1")])

    def run_speed_benchmark(self, targets, hw: HardwareProfile, report, stop, *, runs: int = 1,
                            contexts=(1024, 8192), db=None, bench=None, collector=None, on_timeout=None,
                            sampler_kw=None, runner=None, skip_contexts=None) -> dict:
        """Benchmark prędkości 0.9 (SPEC §5c): modele z Ollamy po kolei, `runs` przebiegów każdego,
        telemetria co 1 s, wynik do bazy (sessions/runs/tasks/telemetry/long_context).
        on_timeout – pytanie po przekroczeniu czasu stabilizacji (GUI); None = test rusza sam."""
        from aibobr import __version__
        from aibobr.bench.ollama import OllamaBench
        from aibobr.bench.session import SpeedBenchmark
        from aibobr.sources.factory import OLLAMA_PREFIX
        bench = bench or OllamaBench(self.ollama_host())
        models = [t[len(OLLAMA_PREFIX):] if t.startswith(OLLAMA_PREFIX) else t for t in targets]
        info = {"app_version": __version__, "runtime_version": bench.version(), "hardware": self.hardware_text(hw),
                "power": power_source()}
        if db is None:
            db = self.bench_db
        job = SpeedBenchmark(bench, self.speed_pack(), db=db, collector=collector, settings=self.settings,
                             info=info, on_timeout=on_timeout, sampler_kw=sampler_kw, runner=runner)
        skip = {m[len(OLLAMA_PREFIX):] if m.startswith(OLLAMA_PREFIX) else m: v
                for m, v in (skip_contexts or {}).items()}
        return job.run(models, runs, contexts, report, stop, skip_contexts=skip)

    def run_benchmark(self, targets, contexts, predicted: dict, hw: HardwareProfile, report, stop,
                      db=None, now=None, bench=None, power_sampler=None, vram_busy=None,
                      suites=("perf",), repeats: int = 1, mode: str = "standard") -> list:
        """Testy modeli z Ollamy po kolei (nigdy kilku naraz). report(dict) – postęp i dziennik.

        suites: "perf" (wydajność przy długościach z contexts) i/lub pakiety jakości (etap 12c,
        np. "pl_jezyk_v1"). Każdy model: najpierw wydajność, potem jakość, potem zwolnienie z karty.
        Wynik: PerfResult i QualityResult w kolejności pomiaru."""
        import datetime
        from dataclasses import replace as _r
        from aibobr.bench.ollama import OllamaBench
        from aibobr.bench.quality import QualityResult, QualityRunner
        from aibobr.bench.runner import PerfResult, PerfRunner, Stopped
        from aibobr.sources.factory import OLLAMA_PREFIX
        suite = self.bench_suite()
        bench = bench or OllamaBench(self.ollama_host())
        if power_sampler is None and any(g.power_limit_w for g in hw.gpus):
            from aibobr.bench.power import PowerSampler
            power_sampler = PowerSampler
        version = bench.version()
        busy = self.vram_busy_now(hw, bench) if vram_busy is None else vram_busy
        runner = PerfRunner(bench, suite)
        quality = {sid: self.quality_suite(sid) for sid in suites if sid != "perf"}
        hardware, supply = self.hardware_text(hw), power_source()
        stamp = (now or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M")
        results = []
        jobs = [j for tg in targets for j in [(tg, ("perf", c)) for c in (contexts if "perf" in suites else ())]
                + [(tg, ("quality", sid)) for sid in quality]]
        q_jobs = [what for _, (kind, what) in jobs if kind == "quality"]
        report({"event": "start", "version": version, "jobs": len(jobs),
                "perf": len(jobs) - len(q_jobs), "q_suites": len(q_jobs),
                "q_tasks": sum(len(quality[sid].get("tests", ())) for sid in q_jobs),
                "q_runs": max(1, int(repeats)) if mode == "controlled" else 1,
                "busy_mb": round(busy / 1024**2) if busy is not None else None})
        loaded = None                        # model w karcie: zostaje między długościami kontekstu
        chunk = lambda text: report({"event": "chunk", "text": text})      # noqa: E731

        def unload(model):
            # Po teście modelu zwalniamy go od razu – inaczej Ollama trzyma go jeszcze 5 minut,
            # a następny model albo inny program dostaje mniej pamięci karty.
            try:
                bench.unload(model)
                report({"event": "unload", "model": model})
            except Exception:
                pass

        def task(model, ctx, step, prompt):
            report({"event": "p_task", "model": model, "context": ctx, "step": step, "prompt": preview(prompt)})

        try:
            for n, (target, (kind, what)) in enumerate(jobs, 1):
                model = target[len(OLLAMA_PREFIX):] if target.startswith(OLLAMA_PREFIX) else target
                if loaded and loaded != model:
                    unload(loaded)
                if kind == "quality":
                    level = predicted.get((target, 1024))       # jakość liczymy przy 2k
                    if level == "red":
                        results.append(QualityResult(model, what, "", "", "", False, "", "", "",
                                                     skipped="bench.skip_red"))
                        report({"event": "q_skip", "model": model, "suite": what})
                        continue
                    loaded = model
                    try:
                        out = QualityRunner(bench, quality[what], repeats, mode).run(model, report, stop)
                    except Stopped:
                        report({"event": "stopped", "n": n, "model": model})
                        break
                    results += out
                    report({"event": "q_done", "model": model, "suite": what, "results": out})
                    if db is not None:
                        for r in out:
                            db.add_quality(r, measured_at=stamp, runtime="ollama", runtime_version=version,
                                           hardware=hardware,
                                           suite_version=quality[what]["suite_info"].get("version"))
                    continue
                ctx = what
                level = predicted.get((target, ctx))
                if level == "red":
                    results.append(PerfResult(model, ctx, predicted=level, skipped="bench.skip_red"))
                    report({"event": "skip", "n": n, "total": len(jobs), "model": model, "context": ctx})
                    continue
                loaded = model
                try:
                    with (power_sampler() if power_sampler else _NoSampler()) as power:
                        r = runner.run(model, ctx, lambda step: report({"n": n, "total": len(jobs), "model": model,
                                                                        "context": ctx, "step": step}), stop,
                                       measured=lambda step, g: report({"event": "measure", "n": n, "model": model,
                                                                        "context": ctx, "step": step,
                                                                        "pp": g.pp_tps, "ttft": g.ttft_ms,
                                                                        "tg": g.tg_tps}),
                                       on_chunk=chunk,
                                       on_task=lambda step, prompt: task(model, ctx, step, prompt))
                    r = _r(r, power_draw_w=power.peak)
                except Stopped:
                    results.append(PerfResult(model, ctx, predicted=level, stopped=True))
                    report({"event": "stopped", "n": n, "model": model, "context": ctx})
                    break
                r = _r(r, predicted=level)
                results.append(r)
                report({"event": "result", "n": n, "total": len(jobs), "result": r})
                if db is not None:
                    gpu = next((g for g in hw.gpus if g.selected and g.power_limit_w), None)
                    db.add_perf(r, measured_at=stamp, runtime="ollama", runtime_version=version,
                                hardware=hardware, power=supply, suite=suite["suite_info"]["id"],
                                power_limit_w=gpu.power_limit_w if gpu else None,
                                power_max_w=gpu.power_max_w if gpu else None,
                                vram_busy_mb=round(busy / 1024**2) if busy is not None else None)
        finally:
            if loaded:
                unload(loaded)
        return results

    @staticmethod
    def vram_busy_now(hw: HardwareProfile, bench=None, run=None) -> Optional[int]:
        """Pamięć karty zajęta przez system przed testem (bez modeli Ollamy) – tłumaczy różnice
        między komputerami: karta z monitorem ma dla modelu mniej miejsca."""
        from aibobr.detection.base import CommandRunner
        from aibobr.detection.nvidia import from_nvidia_smi
        cards = [g for g in hw.gpus if g.selected and not g.integrated]
        if len(cards) != 1:
            return None
        card, used = cards[0], None
        if card.vendor == GpuVendor.NVIDIA and card.pci_bus:          # świeży odczyt przed testem
            fresh = from_nvidia_smi(run or CommandRunner()) or ()
            used = next((g.vram_used for g in fresh if g.pci_bus == card.pci_bus), None)
        if used is None:
            return card.vram_busy
        loaded = 0
        try:
            data = bench.http.get_json(f"{bench.host}/api/ps") if bench is not None else {}
            loaded = sum(int(m.get("size_vram") or 0) for m in (data or {}).get("models") or [])
        except Exception:
            pass
        return max(0, used - loaded)


def preview(prompt: str, head: int = 400, tail: int = 200) -> str:
    """Długi prompt testu wydajności (85% kontekstu) – do podglądu tylko początek i koniec."""
    if len(prompt) <= head + tail + 40:
        return prompt
    return f"{prompt[:head]}\n… [{len(prompt) - head - tail} …]\n{prompt[-tail:]}"


def _short_gpu(name: str) -> str:
    """„NVIDIA GeForce RTX 5050 Laptop GPU” → „RTX 5050 Laptop” – do wąskiej kolumny tabeli."""
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD ", "Intel(R) ", "Intel "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(" Laptop GPU", " Laptop").replace(" GPU", "").strip()


def power_source() -> Optional[str]:
    """„ac” / „battery” – laptop na baterii zwalnia kartę; None: komputer bez baterii albo nie wiadomo."""
    try:
        import psutil
        battery = psutil.sensors_battery()
    except Exception:
        return None
    if battery is None:
        return None
    return "ac" if battery.power_plugged else "battery"


class _NoSampler:
    peak = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None
