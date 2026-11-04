"""Pobór mocy karty NVIDIA w trakcie benchmarku (etap 12a) – nvidia-smi co sekundę.

Pozwala zauważyć kartę „na ścianie” limitu mocy: pomiar autora 4.11.2026 – laptop RTX 5050
cały test na 35 W (z 100 W możliwych), taktowanie skakało 600–2000 MHz. Bez nvidia-smi – nic.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from aibobr.detection.base import CommandRunner

AT_LIMIT = 0.9          # pobór ≥ 90% limitu – karta „na ścianie”
ABOVE_LIMIT = 1.05      # pobór wyraźnie ponad odczytany limit – limit jest dynamiczny (Dynamic Boost)


def capped_during_test(gpu, peak_w: Optional[float]) -> bool:
    """Czy karta przez test stała na niskim limicie mocy.

    Pomiar autora 4.11.2026: z działającym nvidia-powerd sterownik podaje w spoczynku limit 55 W,
    a pod obciążeniem karta bierze 68–77 W – to nie blokada, więc bez ostrzeżenia."""
    if gpu is None or not peak_w or not gpu.power_capped or not gpu.power_limit_w:
        return False
    if gpu.dynamic_boost is True or peak_w > ABOVE_LIMIT * gpu.power_limit_w:
        return False
    return peak_w >= AT_LIMIT * gpu.power_limit_w


QUERY = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]


class PowerSampler:
    def __init__(self, run: Callable = CommandRunner(), interval: float = 1.0):
        self.run, self.interval = run, interval
        self.peak: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        out = self.run(QUERY, timeout=5)
        for line in (out or "").splitlines():
            try:
                value = float(line.strip())
            except ValueError:
                continue
            self.peak = max(self.peak or 0.0, value)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "PowerSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6)
