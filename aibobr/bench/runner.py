"""Pomiar wydajności jednego modelu przy kilku długościach kontekstu (etap 12a, pakiet perf_v1).

Dla każdej długości: rozgrzewka (Ollama wczytuje model z tym num_ctx – zmiana num_ctx to
przeładowanie, więc grupujemy po kontekście), potem `runs` przebiegów przetwarzania promptu
(~85% kontekstu) i `runs` przebiegów generowania `generate_tokens` tokenów. Wynik = mediana.
"""
from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from aibobr.bench import prompts
from aibobr.bench.ollama import OllamaBench


@dataclass(frozen=True)
class PerfResult:
    model: str
    context: int
    pp_tps: Optional[float] = None          # przetwarzanie promptu, tokeny/s
    ttft_ms: Optional[float] = None         # czas do pierwszego tokenu przy pełnym prompcie
    tg_tps: Optional[float] = None          # generowanie, tokeny/s
    prompt_tokens: int = 0
    gen_tokens: int = 0
    gpu_share: Optional[float] = None       # część modelu w karcie wg Ollamy
    predicted: Optional[str] = None         # werdykt AI Bóbra dla tego kontekstu
    skipped: Optional[str] = None           # klucz i18n, gdy pominięty (np. nie zmieści się)
    stopped: bool = False


class Stopped(Exception):
    pass


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


class PerfRunner:
    def __init__(self, bench: OllamaBench, suite: dict):
        self.bench, self.suite = bench, suite
        self.runs = int(suite.get("runs", 3))
        self.warmup = int(suite.get("warmup_runs", 1))
        self.gen_tokens = int(suite.get("generate_tokens", 256))

    def steps_per_context(self) -> int:
        return self.warmup + 2 * self.runs

    def run(self, model: str, context: int, report: Callable[[str], None] = lambda s: None,
            stop: Optional[threading.Event] = None) -> PerfResult:
        def check():
            if stop is not None and stop.is_set():
                raise Stopped()

        for n in range(self.warmup):
            check()
            report("warmup")
            self.bench.generate(model, prompts.generation(-1 - n), context, 16)
        pp, ttft, ptok = [], [], []
        for n in range(1, self.runs + 1):
            check()
            report(f"prompt {n}/{self.runs}")
            g = self.bench.generate(model, prompts.prompt_processing(context, n), context, 8)
            pp.append(g.pp_tps)
            ttft.append(g.ttft_ms)
            ptok.append(g.prompt_tokens)
        tg, gtok = [], []
        for n in range(1, self.runs + 1):
            check()
            report(f"generate {n}/{self.runs}")
            g = self.bench.generate(model, prompts.generation(n), context, self.gen_tokens)
            tg.append(g.tg_tps)
            gtok.append(g.gen_tokens)
        share = self.bench.gpu_share(model)
        return PerfResult(model=model, context=context, pp_tps=_median(pp), ttft_ms=_median(ttft),
                          tg_tps=_median(tg), prompt_tokens=int(_median(ptok) or 0),
                          gen_tokens=int(_median(gtok) or 0), gpu_share=share)


# Domyślne prędkości do szacunku czasu, gdy modelu jeszcze nie mierzono na tym komputerze –
# z pomiarów autora (RTX 5050 Laptop 8 GB): w karcie gemma2:9b ok. 900–1000 tok/s promptu
# i ~22 tok/s generowania; karta + RAM qwen2.5-coder:14b ok. 470–510 i 9–12 tok/s.
DEFAULT_SPEED = {"green": (950.0, 20.0), "yellow": (480.0, 9.0), None: (480.0, 9.0)}
LOAD_SECONDS = 12


def estimate_seconds(context: int, predicted: Optional[str], runs: int = 3, gen_tokens: int = 256,
                     measured: Optional[tuple[float, float]] = None) -> float:
    """Szacunek czasu jednej długości kontekstu: wczytanie + `runs` promptów (~85% kontekstu)
    + `runs` generowań. measured = (prompt tok/s, generowanie tok/s) z wcześniejszego pomiaru."""
    pp, tg = measured if measured and all(measured) else DEFAULT_SPEED.get(predicted, DEFAULT_SPEED[None])
    return LOAD_SECONDS + runs * (context * 0.85 / pp + 8 / tg) + runs * gen_tokens / tg + gen_tokens / tg / 16
