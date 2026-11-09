"""Sesja benchmarku prędkości 0.9: modele po kolei, przebiegi, telemetria i zapis do bazy.

Przed każdym przebiegiem każdego modelu (decyzja autora 9.11.2026):
  1. zwolnienie modelu i sprawdzenie, że zasoby wróciły (VRAM jak przed sesją ± 300 MB),
  2. stabilizacja temperatury CPU i GPU (komunikat dla użytkownika, okno po przekroczeniu czasu),
  3. linia bazowa 10 s (próbka co 1 s),
  4. pakiet 15 zadań, potem osobno długi kontekst; wartości bezwzględne + różnice od linii bazowej.
Kilka przebiegów to pomiar rozrzutu, nie dowód powtarzalności. Pojedynczy przebieg to próbka
wydajności w tych warunkach, nie dowód stabilności.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable, Optional

from aibobr.bench import speed, telemetry
from aibobr.bench.runner import Stopped

RECOVER_WAIT = 10.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class SpeedBenchmark:
    def __init__(self, bench, pack: dict, *, db=None, collector: Optional[telemetry.Collector] = None,
                 settings: Optional[dict] = None, info: Optional[dict] = None,
                 on_timeout: Optional[Callable[[], bool]] = None, runner: Optional[speed.SpeedRunner] = None,
                 sampler_kw: Optional[dict] = None, stamp: Callable[[], str] = now_iso,
                 sleep: Callable[[float], None] = time.sleep):
        self.bench, self.pack, self.db = bench, pack, db
        self.collector = collector
        self.settings = settings or {}
        self.info = info or {}                     # app_version, hardware, hardware_json, power, runtime_version
        self.on_timeout = on_timeout
        self.runner = runner or speed.SpeedRunner(bench, pack)
        self.sampler_kw = sampler_kw or {}
        self.stamp, self.sleep = stamp, sleep
        self.thresholds = speed.thresholds_of(self.settings)
        self.sampler: Optional[telemetry.Sampler] = None
        self.session_id: Optional[str] = None

    # -------------------------------------------------------------- pomocnicze
    def _vram(self) -> Optional[float]:
        if self.sampler is None:
            return None
        sample = self.sampler.sample_once() if not self.sampler.history else self.sampler.history[-1][1]
        return sample.get("vram_used_mb")

    def _recovered(self, reference: Optional[float], stop: threading.Event) -> Optional[bool]:
        """Czy VRAM wrócił do poziomu sprzed sesji. None – brak odczytu karty."""
        if reference is None:
            return None
        waited = 0.0
        while True:
            now = self._vram()
            if now is None:
                return None
            if now <= reference + speed.VRAM_RECOVER_MB:
                return True
            if waited >= RECOVER_WAIT or stop.wait(1.0):
                return False
            waited += 1.0

    def _session_row(self, models: list[str], runs: int, contexts: list[int]) -> dict:
        cfg = telemetry.settings_of(self.settings)
        return {"started_at": self.stamp(), "app_version": self.info.get("app_version"), "runtime": "ollama",
                "runtime_version": self.info.get("runtime_version"), "hardware": self.info.get("hardware"),
                "hardware_json": self.info.get("hardware_json"), "power": self.info.get("power"),
                "suite": self.pack["id"], "suite_version": self.pack["version"], "runs_per_model": runs,
                "models_json": models, "thresholds_version": speed.THRESHOLDS_VERSION,
                "thresholds_json": self.thresholds,
                "settings_json": {**cfg, "context": speed.CONTEXT, "long_contexts": contexts,
                                  "long_reserve": speed.LONG_RESERVE}}

    # -------------------------------------------------------------- przebieg
    def run(self, models: list[str], runs: int = 1, contexts: tuple = (),
            report: Callable[[dict], None] = lambda e: None, stop: Optional[threading.Event] = None,
            skip_contexts: Optional[dict] = None) -> dict:
        """skip_contexts: model → długości, których nie mierzymy (AI Bóbr: nie zmieści się)."""
        stop = stop or threading.Event()
        runs = max(1, int(runs))
        contexts = [int(c) for c in contexts]
        skip_contexts = skip_contexts or {}
        session = self._session_row(models, runs, contexts)
        self.session_id = self.db.start_session(**session) if self.db is not None else speed_id()
        collector = self.collector or telemetry.default_collector()
        self.sampler = telemetry.Sampler(collector, self.session_id, db=self.db, **self.sampler_kw)
        self.sampler.listeners.append(lambda s: report({"event": "telemetry", "sample": s}))
        result = {"session_id": self.session_id, "runs": [], "status": "finished"}
        report({"event": "s_start", "session_id": self.session_id, "models": models, "runs": runs,
                "tasks": len(self.pack["tasks"]), "contexts": contexts,
                "version": self.info.get("runtime_version")})
        reference = None
        self.sampler.start()
        try:
            for model in models:
                for run_no in range(1, runs + 1):
                    if stop.is_set():
                        raise Stopped()
                    released = self.runner.release(model)
                    if reference is None:
                        reference = self._vram()
                    recovered = self._recovered(reference, stop)
                    report({"event": "s_release", "model": model, "run": run_no, "released": released,
                            "recovered": recovered})
                    mine = [c for c in contexts if c not in skip_contexts.get(model, ())]
                    result["runs"].append(self._one(model, run_no, runs, mine, released, recovered,
                                                    report, stop))
                self.runner.release(model)
                report({"event": "unload", "model": model})
        except Stopped:
            result["status"] = "stopped"
            report({"event": "stopped"})
        except Exception as e:
            result["status"] = "error"
            report({"event": "s_error", "error": str(e)})
            raise
        finally:
            self.sampler.stop()
            if self.db is not None:
                self.db.finish_session(self.session_id, status=result["status"], finished_at=self.stamp())
        report({"event": "s_done", "status": result["status"], "session_id": self.session_id})
        return result

    def _one(self, model, run_no, runs, contexts, released, recovered, report, stop) -> dict:
        run = {"model": model, "run_no": run_no, "started_at": self.stamp(), "num_ctx": speed.CONTEXT,
               "thresholds_version": speed.THRESHOLDS_VERSION, "thresholds_json": self.thresholds}
        rid = self.db.start_run(self.session_id, **run) if self.db is not None else speed_id()
        cooling = telemetry.cool_down(self.sampler, self.settings, rid, self.on_timeout, stop,
                                      report=lambda key, kw: report({"event": "s_cooling", "key": key, **kw}))
        report({"event": "s_cooled", "model": model, "run": run_no, "cooling": cooling})
        if cooling.status == "cancelled":
            raise Stopped()
        baseline = telemetry.take_baseline(self.sampler, self.settings, rid, stop)
        if baseline is None:
            raise Stopped()
        report({"event": "s_baseline", "model": model, "run": run_no, "baseline": baseline})
        try:
            tasks = self.runner.run_pack(model, report, stop,
                                         on_task=lambda n: self.sampler.set_phase("task", rid, n))
            share = self._gpu_share(model)
            longs = []
            for ctx in contexts:
                if stop.is_set():
                    raise Stopped()
                self.sampler.set_phase("long_context", rid)
                report({"event": "s_long", "model": model, "run": run_no, "context": ctx})
                lc = self.runner.long_context(model, ctx, run_no,
                                              on_chunk=lambda text: report({"event": "chunk", "text": text}))
                longs.append(lc)
                report({"event": "s_long_done", "model": model, "run": run_no, "result": lc})
        except Stopped:
            if self.db is not None:
                self.db.update_run(rid, status="stopped", finished_at=self.stamp())
            raise
        finally:
            self.sampler.set_phase("idle", rid)
        summary = speed.summarize(tasks, self.thresholds)
        warnings = list(summary.pop("warnings"))
        if cooling.warning:
            warnings.append("stabilization_incomplete")
        if not released:
            warnings.append("previous_not_released")
        if recovered is False:
            warnings.append("resources_not_recovered")
        row = {**summary, "finished_at": self.stamp(), "gpu_share": share, "stabilization": cooling.status,
               "cooling_seconds": cooling.seconds, "baseline_json": baseline, "warnings_json": warnings}
        if self.db is not None:
            self.db.update_run(rid, **row)
            for t in tasks:
                self.db.add_task(rid, **t.row())
            for lc in longs:
                self.db.add_long_context(rid, **lc)
        out = {"id": rid, **run, **row, "tasks": tasks, "long_context": longs}
        report({"event": "s_run_done", "model": model, "run": run_no, "runs": runs, "result": out})
        return out

    def _gpu_share(self, model: str) -> Optional[float]:
        try:
            return self.bench.gpu_share(model)
        except Exception:
            return None


def speed_id() -> str:
    from aibobr.app.bench_db import new_id
    return new_id()
