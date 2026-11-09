"""Telemetria benchmarku 0.9 (SPEC §5c): próbka co 1 s, linia bazowa i stabilizacja temperatury.

Źródła (każde może nie działać – wtedy pole = None, w bazie NULL, nigdy 0):
  procesor  – psutil: obciążenie całe i na rdzeń; temperatura tylko na Linuksie (coretemp, k10temp,
              zenpower); moc z Linux RAPL (/sys/class/powercap, często tylko dla roota – wtedy brak);
  pamięć    – psutil: RAM zajęty i dostępny, swap; aktywność swap (B/s) tylko na Linuksie
              (na Windowsie psutil zwraca tam zawsze 0, więc to nie jest pomiar);
  karta     – NVIDIA: NVML (nvidia-ml-py), bez niego nvidia-smi; obciążenie, temperatura, moc,
              taktowanie, VRAM, powód ograniczenia taktowania; temperatura pamięci karty zwykle N/A.

Przed każdym modelem (decyzja autora 9.11.2026): zwolnienie poprzedniego, stabilizacja temperatury
CPU i GPU (bez jednej docelowej temperatury – zmiana ≤ tolerancji w oknie), linia bazowa 10 s,
dopiero potem zadania (wartości bezwzględne + różnica względem linii bazowej).
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from aibobr.detection.base import CommandRunner

log = logging.getLogger(__name__)
MB = 1024 ** 2
LINUX = sys.platform.startswith("linux")

# Ustawienia (config: bench_*). Okno i tolerancja – propozycja zatwierdzona 9.11.2026.
DEFAULTS = {"bench_stabilization_window_s": 30, "bench_stabilization_tolerance_c": 1.0,
            "bench_stabilization_timeout_s": 300, "bench_baseline_s": 10}

NUMERIC = ("cpu_pct", "cpu_temp_c", "cpu_power_w", "ram_used_mb", "ram_available_mb", "swap_used_mb",
           "swap_in_bps", "swap_out_bps", "gpu_util_pct", "gpu_temp_c", "gpu_power_w", "gpu_clock_mhz",
           "vram_used_mb", "vram_total_mb", "vram_temp_c")

# Powody ograniczenia taktowania (maska NVML / nvidia-smi clocks_event_reasons.active).
THROTTLE = {0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
            0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}


def throttle_names(mask: Optional[int]) -> Optional[str]:
    """Maska → „sw_power_cap,hw_thermal”; 0 → „none” (wiemy, że nic nie ogranicza); None → None."""
    if mask is None:
        return None
    names = [name for bit, name in THROTTLE.items() if mask & bit]
    return ",".join(names) if names else "none"


def _num(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):          # "[N/A]", "[Not Supported]"
        return None


def settings_of(settings: Optional[dict]) -> dict:
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        value = (settings or {}).get(key)
        if isinstance(value, (int, float)) and value > 0:
            out[key] = value
    return out


# ------------------------------------------------------------------ procesor i pamięć
CPU_SENSORS = (("coretemp", ("Package id 0",)), ("k10temp", ("Tctl", "Tdie")), ("zenpower", ("Tdie", "Tctl")),
               ("cpu_thermal", ("",)), ("cpu-thermal", ("",)))


def cpu_temperature(read: Optional[Callable] = None) -> Optional[float]:
    """Temperatura pakietu procesora (Linux); Windows i macOS – None (opisane w README)."""
    if read is None:
        import psutil
        read = getattr(psutil, "sensors_temperatures", None)
        if read is None:
            return None
    try:
        sensors = read() or {}
    except Exception:
        return None
    for chip, labels in CPU_SENSORS:
        entries = sensors.get(chip) or []
        for label in labels:
            for e in entries:
                if (e.label or "") == label or (label == "" and e.current):
                    return float(e.current)
        if entries and chip == "coretemp":              # bez etykiety pakietu – najgorętszy rdzeń
            return max(float(e.current) for e in entries)
    return None


class Rapl:
    """Moc pakietu procesora z licznika energii RAPL (µJ). Brak pliku albo brak uprawnień → None."""

    def __init__(self, root: Path = Path("/sys/class/powercap")):
        self.file: Optional[Path] = None
        self.range: Optional[int] = None
        self.last: Optional[tuple[float, int]] = None
        if not LINUX and root == Path("/sys/class/powercap"):
            return
        for zone in sorted(root.glob("intel-rapl:[0-9]")):
            try:
                name = (zone / "name").read_text().strip()
                if name.startswith("package"):
                    int((zone / "energy_uj").read_text())       # sprawdza uprawnienia
                    self.file = zone / "energy_uj"
                    try:
                        self.range = int((zone / "max_energy_range_uj").read_text())
                    except (OSError, ValueError):
                        self.range = None
                    break
            except (OSError, ValueError):
                continue

    def watts(self, now: float) -> Optional[float]:
        if self.file is None:
            return None
        try:
            energy = int(self.file.read_text())
        except (OSError, ValueError):
            self.file = None
            return None
        prev, self.last = self.last, (now, energy)
        if prev is None or now <= prev[0]:
            return None
        delta = energy - prev[1]
        if delta < 0 and self.range:                      # licznik się przekręcił
            delta += self.range
        return delta / 1e6 / (now - prev[0]) if delta >= 0 else None


class CpuRam:
    def __init__(self, psutil_mod=None, linux: bool = LINUX):
        import psutil
        self.ps = psutil_mod or psutil
        self.linux = linux
        self.ps.cpu_percent(percpu=True)                 # pierwsze wywołanie daje 0 – „zerujemy”
        self.last_swap: Optional[tuple[float, int, int]] = None

    def read(self, now: float) -> dict:
        out: dict = {}
        try:
            cores = self.ps.cpu_percent(percpu=True)
            out["cpu_cores_json"] = [round(c, 1) for c in cores]
            out["cpu_pct"] = round(sum(cores) / len(cores), 1) if cores else None
        except Exception:
            pass
        try:
            vm = self.ps.virtual_memory()
            out["ram_used_mb"] = round((vm.total - vm.available) / MB, 1)
            out["ram_available_mb"] = round(vm.available / MB, 1)
        except Exception:
            pass
        try:
            sw = self.ps.swap_memory()
            out["swap_used_mb"] = round(sw.used / MB, 1)
            if self.linux:
                prev, self.last_swap = self.last_swap, (now, sw.sin, sw.sout)
                if prev and now > prev[0]:
                    out["swap_in_bps"] = max(0.0, (sw.sin - prev[1]) / (now - prev[0]))
                    out["swap_out_bps"] = max(0.0, (sw.sout - prev[2]) / (now - prev[0]))
        except Exception:
            pass
        return out


# ------------------------------------------------------------------ karta graficzna (NVIDIA)
class NvmlGpu:
    """Szybki odczyt bez uruchamiania programu co sekundę. None, gdy brak pakietu albo sterownika."""

    @classmethod
    def create(cls) -> Optional["NvmlGpu"]:
        try:
            import pynvml
            pynvml.nvmlInit()
            return cls(pynvml)
        except Exception:
            return None

    def __init__(self, nvml):
        self.nvml = nvml
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]

    def _try(self, fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    def read(self) -> list[dict]:
        n = self.nvml
        out = []
        for h in self.handles:
            util = self._try(n.nvmlDeviceGetUtilizationRates, h)
            mem = self._try(n.nvmlDeviceGetMemoryInfo, h)
            power = self._try(n.nvmlDeviceGetPowerUsage, h)
            reasons = getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                getattr(n, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
            out.append({
                "gpu_util_pct": float(util.gpu) if util is not None else None,
                "gpu_temp_c": self._try(n.nvmlDeviceGetTemperature, h, 0),     # NVML_TEMPERATURE_GPU
                "gpu_power_w": power / 1000 if power is not None else None,
                "gpu_clock_mhz": self._try(n.nvmlDeviceGetClockInfo, h, 1),     # NVML_CLOCK_SM
                "vram_used_mb": round(mem.used / MB, 1) if mem is not None else None,
                "vram_total_mb": round(mem.total / MB, 1) if mem is not None else None,
                "vram_temp_c": None,                         # NVML nie podaje dla kart konsumenckich
                "gpu_throttle": throttle_names(self._try(reasons, h)) if reasons else None,
            })
        return out

    def close(self) -> None:
        try:
            self.nvml.nvmlShutdown()
        except Exception:
            pass


SMI_FIELDS = ("utilization.gpu", "temperature.gpu", "power.draw", "clocks.sm", "memory.used", "memory.total",
              "temperature.memory", "clocks_event_reasons.active")
SMI_FALLBACK = {"clocks_event_reasons.active": "clocks_throttle_reasons.active"}     # sterowniki < 535
SMI_KEYS = ("gpu_util_pct", "gpu_temp_c", "gpu_power_w", "gpu_clock_mhz", "vram_used_mb", "vram_total_mb",
            "vram_temp_c", "gpu_throttle")


class SmiGpu:
    """nvidia-smi co próbkę. Pole nieznane starszemu sterownikowi → zapytanie bez niego (None)."""

    def __init__(self, run: Callable = CommandRunner()):
        self.run = run
        self.fields: Optional[list[str]] = None

    def _query(self, fields) -> Optional[str]:
        return self.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                        timeout=5)

    def read(self) -> list[dict]:
        if self.fields is None:                          # pierwsza próbka: ustalamy, co sterownik zna
            candidates = [list(SMI_FIELDS), [SMI_FALLBACK.get(f, f) for f in SMI_FIELDS],
                          [f for f in SMI_FIELDS if f not in ("temperature.memory", "clocks_event_reasons.active")]]
            for fields in candidates:
                out = self._query(fields)
                if out is not None:
                    self.fields = fields
                    return self._parse(out)
            self.fields = []
            return []
        if not self.fields:
            return []
        return self._parse(self._query(self.fields))

    def _parse(self, out: Optional[str]) -> list[dict]:
        gpus = []
        for line in (out or "").strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            row = dict.fromkeys(SMI_KEYS)
            for field, value in zip(self.fields, cols):
                field = {v: k for k, v in SMI_FALLBACK.items()}.get(field, field)
                key = SMI_KEYS[SMI_FIELDS.index(field)]
                if key == "gpu_throttle":
                    try:
                        row[key] = throttle_names(int(value, 16))
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = _num(value)
            gpus.append(row)
        return gpus

    def close(self) -> None:
        pass


def gpu_source(run: Callable = CommandRunner(), use_nvml: bool = True):
    return (NvmlGpu.create() if use_nvml else None) or SmiGpu(run)


# ------------------------------------------------------------------ próbnik
@dataclass
class Collector:
    """Jedna próbka ze wszystkich źródeł. `gpu_index` – karta, na której działa model; pozostałe
    karty trafiają do extra_json."""
    cpu: Optional[CpuRam] = None
    gpu: object = None
    rapl: Optional[Rapl] = None
    cpu_temp: Callable[[], Optional[float]] = cpu_temperature
    gpu_index: int = 0

    def collect(self, now: float) -> dict:
        sample = dict.fromkeys(NUMERIC)
        sample.update(cpu_cores_json=None, gpu_throttle=None)
        if self.cpu is not None:
            sample.update(self.cpu.read(now))
        sample["cpu_temp_c"] = self.cpu_temp()
        sample["cpu_power_w"] = self.rapl.watts(now) if self.rapl is not None else None
        gpus = []
        if self.gpu is not None:
            try:
                gpus = self.gpu.read()
            except Exception as e:                       # karta „zniknęła” – zapisujemy brak, nie 0
                log.warning("GPU telemetry failed: %s", e)
        if 0 <= self.gpu_index < len(gpus):
            sample.update(gpus[self.gpu_index])
            others = [g for i, g in enumerate(gpus) if i != self.gpu_index]
            if others:
                sample["extra_json"] = {"gpus": others}
        return sample

    def close(self) -> None:
        if self.gpu is not None and hasattr(self.gpu, "close"):
            self.gpu.close()


def default_collector(gpu_index: int = 0, run: Callable = CommandRunner()) -> Collector:
    return Collector(cpu=CpuRam(), gpu=gpu_source(run), rapl=Rapl(), gpu_index=gpu_index)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class Sampler:
    """Wątek: próbka co `interval` s (bez dryfu), faza ustawiana przez przebieg, paczki do bazy.

    `history` – (czas monotoniczny, próbka) do okna Monitor i do stabilizacji; `listeners` dostają
    każdą próbkę (z wątku próbnika – GUI przekazuje dalej przez `after`)."""

    def __init__(self, collector: Collector, session_id: str, db=None, interval: float = 1.0,
                 clock: Callable[[], float] = time.monotonic, stamp: Callable[[], str] = now_iso,
                 batch: int = 5):
        self.collector, self.session_id, self.db = collector, session_id, db
        self.interval, self.clock, self.stamp, self.batch = interval, clock, stamp, batch
        self.history: list[tuple[float, dict]] = []
        self.listeners: list[Callable[[dict], None]] = []
        self.phase, self.run_id, self.task_no = "idle", None, None
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def set_phase(self, phase: str, run_id: Optional[str] = None, task_no: Optional[int] = None) -> None:
        with self._lock:
            self.phase, self.run_id, self.task_no = phase, run_id, task_no

    def sample_once(self) -> dict:
        now = self.clock()
        data = self.collector.collect(now)
        with self._lock:
            data.update(session_id=self.session_id, run_id=self.run_id, t=self.stamp(),
                        phase=self.phase, task_no=self.task_no)
            self.history.append((now, data))
            if self.db is not None and self.phase != "idle":
                self._pending.append(data)
            flush = len(self._pending) >= self.batch
        for fn in list(self.listeners):
            try:
                fn(data)
            except Exception as e:
                log.warning("telemetry listener failed: %s", e)
        if flush:
            self.flush()
        return data

    def flush(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        if pending and self.db is not None:
            try:
                self.db.add_telemetry(pending)
            except Exception as e:                       # zapis nie może przerwać pomiaru
                log.warning("telemetry write failed: %s", e)

    def since(self, start: float) -> list[tuple[float, dict]]:
        with self._lock:
            return [(m, s) for m, s in self.history if m >= start]

    def _loop(self) -> None:
        next_at = self.clock()
        while not self._stop.is_set():
            self.sample_once()
            next_at += self.interval
            self._stop.wait(max(0.0, next_at - self.clock()))

    def start(self) -> "Sampler":
        self._thread = threading.Thread(target=self._loop, name="bobr-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.flush()
        self.collector.close()

    def __enter__(self) -> "Sampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ------------------------------------------------------------------ stabilizacja i linia bazowa
TEMP_KEYS = ("cpu_temp_c", "gpu_temp_c")


def stability(samples: list[tuple[float, dict]], now: float, window: float, tolerance: float) -> str:
    """'stable' – w ostatnich `window` s każda dostępna temperatura (CPU, GPU) zmieniła się o ≤ `tolerance`;
    'no_sensors' – żadna temperatura nie jest dostępna; 'waiting' – za krótko albo dalej się zmienia."""
    if not samples:
        return "waiting"
    if not any(s.get(k) is not None for _, s in samples for k in TEMP_KEYS):
        return "no_sensors"
    if now - samples[0][0] < window:
        return "waiting"
    recent = [s for m, s in samples if m >= now - window]
    for key in TEMP_KEYS:
        values = [s[key] for s in recent if s.get(key) is not None]
        if values and max(values) - min(values) > tolerance:
            return "waiting"
    return "stable"


@dataclass
class Cooling:
    status: str                  # stable / no_sensors / timeout_continued / cancelled
    seconds: float
    cpu_temp_c: Optional[float] = None
    gpu_temp_c: Optional[float] = None

    @property
    def warning(self) -> bool:
        return self.status == "timeout_continued"


def cool_down(sampler: Sampler, settings: Optional[dict] = None, run_id: Optional[str] = None,
              on_timeout: Optional[Callable[[], bool]] = None, stop: Optional[threading.Event] = None,
              report: Callable[[str, dict], None] = lambda key, kw: None,
              poll: Optional[float] = None) -> Cooling:
    """Czeka na stabilną temperaturę. Po przekroczeniu czasu pyta `on_timeout()`: True – „Kontynuuj test”
    (wynik z ostrzeżeniem), False – „Czekaj dalej” (pytamy znowu po kolejnym takim okresie).
    Bez `on_timeout` (tryb tekstowy, decyzja autora 3a) test rusza sam, z ostrzeżeniem."""
    cfg = settings_of(settings)
    window, tol, timeout = (cfg["bench_stabilization_window_s"], cfg["bench_stabilization_tolerance_c"],
                            cfg["bench_stabilization_timeout_s"])
    stop = stop or threading.Event()
    poll = poll or min(1.0, sampler.interval)
    sampler.set_phase("cooling", run_id)
    start = sampler.clock()
    deadline = start + timeout
    report("bench.cooling.start", {"window": window, "tolerance": tol})

    def result(status: str) -> Cooling:
        last = sampler.since(start)
        tail = last[-1][1] if last else {}
        return Cooling(status, round(sampler.clock() - start, 1), tail.get("cpu_temp_c"), tail.get("gpu_temp_c"))

    while True:
        now = sampler.clock()
        state = stability(sampler.since(start), now, window, tol)
        if state in ("stable", "no_sensors"):
            return result(state)
        if now >= deadline:
            if on_timeout is None or on_timeout():
                return result("timeout_continued")
            deadline = sampler.clock() + timeout
            report("bench.cooling.wait_more", {})
        if stop.wait(poll):
            return result("cancelled")


def summarize(samples: list[dict]) -> dict:
    """Średnia i maksimum każdego pola liczbowego; pole bez odczytów → None (nie 0)."""
    mean, peak = {}, {}
    for key in NUMERIC:
        values = [s[key] for s in samples if s.get(key) is not None]
        mean[key] = round(sum(values) / len(values), 2) if values else None
        peak[key] = max(values) if values else None
    cores = [s["cpu_cores_json"] for s in samples if s.get("cpu_cores_json")]
    if cores and all(len(c) == len(cores[0]) for c in cores):
        mean["cpu_cores"] = [round(sum(c[i] for c in cores) / len(cores), 1) for i in range(len(cores[0]))]
    return {"samples": len(samples), "mean": mean, "max": peak}


def take_baseline(sampler: Sampler, settings: Optional[dict] = None, run_id: Optional[str] = None,
                  stop: Optional[threading.Event] = None) -> Optional[dict]:
    """Linia bazowa: `bench_baseline_s` s próbek w spoczynku, po stabilizacji. None przy przerwaniu."""
    seconds = settings_of(settings)["bench_baseline_s"]
    needed = max(1, round(seconds / sampler.interval))
    stop = stop or threading.Event()
    sampler.set_phase("baseline", run_id)
    start = sampler.clock()

    def taken():
        return [s for _, s in sampler.since(start) if s.get("phase") == "baseline"]
    while len(taken()) < needed and sampler.clock() < start + seconds + 3 * sampler.interval:
        if stop.wait(sampler.interval / 4):
            return None
    return {"seconds": seconds, **summarize(taken()[:needed])}


def delta(sample: dict, baseline: Optional[dict]) -> dict:
    """Różnica względem średniej z linii bazowej; brak którejkolwiek wartości → None."""
    base = (baseline or {}).get("mean") or {}
    return {k: (round(sample[k] - base[k], 2) if sample.get(k) is not None and base.get(k) is not None else None)
            for k in NUMERIC}


def as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False)
