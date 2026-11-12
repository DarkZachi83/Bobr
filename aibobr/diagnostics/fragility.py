"""Kruchość promptu: odstęp między pierwszym a drugim kandydatem przy każdym wygenerowanym tokenie.

Skąd: wskazówka z r/LocalLLaMA po eksperymencie BOS (11.11.2026) – „zapisuj odstęp między najlepszym
kandydatem a drugim; krok z odstępem bliskim zera oznacza kruchy prompt, zanim porówna się silniki”.
Różnica wyniku na zadaniu kruchym może być przeważonym remisem; na stabilnym – rzeczywistą różnicą wejścia.

Zasady (decyzje autora 12.11.2026):
- osobny przebieg z `logprobs` – nigdy w trakcie benchmarku ani testu jakości (sama prośba o logprobs
  zmieniała zachowanie pamięci promptu w eksperymencie z 7.11);
- każde zadanie od zera (model zwolniony), te same ustawienia co test jakości (num_ctx, limit tokenów,
  temperatura 0, seed 42) – inaczej odstępy dotyczyłyby innej odpowiedzi;
- progi ustawiane przez użytkownika; domyślnie: kruche < 0,10, remis < 0,02 (różnica prawdopodobieństw);
- odstęp liczy się tylko na ścieżce, którą model wybrał – najwięcej mówi pierwszy prawie remis.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional

FRAGILE = 0.10
TIE = 0.02
TOP = 2
STABLE, FRAGILE_LABEL, TIE_LABEL, UNSUPPORTED, NOT_RUN = "stable", "fragile", "tie", "unsupported", "not_run"
CONTEXT_CHARS = 40


@dataclass(frozen=True)
class Thresholds:
    fragile: float = FRAGILE
    tie: float = TIE

    def __post_init__(self):
        if not (0 <= self.tie <= self.fragile <= 1):
            raise ValueError("progi: 0 ≤ remis ≤ kruche ≤ 1")


@dataclass(frozen=True)
class Step:
    pos: int
    token: str
    first: Optional[tuple[str, float]]       # (token, prawdopodobieństwo) – najlepszy kandydat
    second: Optional[tuple[str, float]]      # drugi kandydat; None – Ollama podała tylko jednego
    margin: float                            # p1 − p2 (bez drugiego: p1)


@dataclass(frozen=True)
class TaskFragility:
    task_id: str
    label: str                               # stable / fragile / tie / unsupported / not_run
    steps: int = 0
    min_margin: Optional[float] = None
    min_pos: Optional[int] = None
    min_pair: Optional[tuple] = None         # ((token1, p1), (token2, p2)) w najwęższym miejscu
    fragile_steps: int = 0
    tie_steps: int = 0
    first_pos: Optional[int] = None          # pierwszy krok poniżej progu „kruche”
    first_pair: Optional[tuple] = None
    first_context: Optional[str] = None      # tekst odpowiedzi tuż przed pierwszym prawie remisem
    answer: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _p(entry: dict) -> Optional[float]:
    lp = entry.get("logprob")
    return math.exp(lp) if isinstance(lp, (int, float)) else None


def steps(logprobs: Iterable[dict]) -> list[Step]:
    """Z odpowiedzi Ollamy (logprobs + top_logprobs) – odstęp przy każdym tokenie."""
    out = []
    for pos, e in enumerate(logprobs or []):
        cands = [(c.get("token"), _p(c)) for c in (e.get("top_logprobs") or []) if _p(c) is not None]
        cands.sort(key=lambda c: c[1], reverse=True)
        if not cands:                             # bez kandydatów – tylko wybrany token
            p = _p(e)
            if p is None:
                continue
            cands = [(e.get("token"), p)]
        first = cands[0]
        second = cands[1] if len(cands) > 1 else None
        out.append(Step(pos, str(e.get("token") or ""), first, second, first[1] - (second[1] if second else 0.0)))
    return out


def classify(task_id: str, logprobs: Optional[list], answer: str = "",
             thresholds: Thresholds = Thresholds()) -> TaskFragility:
    if logprobs is None:
        return TaskFragility(task_id, UNSUPPORTED, answer=answer, reason="no_logprobs")
    st = steps(logprobs)
    if not st:
        return TaskFragility(task_id, NOT_RUN, answer=answer, reason="empty")
    low = min(st, key=lambda s: s.margin)
    fragile = [s for s in st if s.margin < thresholds.fragile]
    ties = [s for s in st if s.margin < thresholds.tie]
    label = TIE_LABEL if ties else FRAGILE_LABEL if fragile else STABLE
    first = fragile[0] if fragile else None
    context = None
    if first is not None:
        context = "".join(s.token for s in st[:first.pos])[-CONTEXT_CHARS:]
    return TaskFragility(task_id, label, len(st), round(low.margin, 6), low.pos, (low.first, low.second),
                         len(fragile), len(ties), first.pos if first else None,
                         (first.first, first.second) if first else None, context, answer)


@dataclass
class FragilityRun:
    model: str
    thresholds: Thresholds
    tasks: list[TaskFragility] = field(default_factory=list)

    def counts(self) -> dict:
        out = {STABLE: 0, FRAGILE_LABEL: 0, TIE_LABEL: 0, UNSUPPORTED: 0, NOT_RUN: 0}
        for t in self.tasks:
            out[t.label] = out.get(t.label, 0) + 1
        return out


class FragilityProbe:
    """Przebieg dla jednego modelu i listy zadań [(id, prompt, limit tokenów)]."""

    def __init__(self, bench, clock: Callable[[], float], sleep: Callable[[float], None],
                 unload_timeout: float = 30.0, num_ctx: int = 2048):
        from aibobr.diagnostics.bos import BosProbe           # to samo zwalnianie modelu co pomiar tokenu początku
        self.bench, self.num_ctx = bench, num_ctx
        self._release = BosProbe(bench, clock, sleep, unload_timeout).release

    def run(self, model: str, tasks: list[tuple[str, str, int]], thresholds: Thresholds = Thresholds(),
            report: Callable[[dict], None] = lambda e: None,
            stop: Callable[[], bool] = lambda: False) -> FragilityRun:
        result = FragilityRun(model, thresholds)
        try:
            for n, (task_id, prompt, max_tokens) in enumerate(tasks, 1):
                if stop():
                    break
                report({"event": "f_task", "model": model, "task_id": task_id, "n": n, "total": len(tasks)})
                if not self._release(model):
                    item = TaskFragility(task_id, NOT_RUN, reason="not_released")
                else:
                    try:
                        g = self.bench.generate(model, prompt, self.num_ctx, int(max_tokens), top_logprobs=TOP)
                        item = classify(task_id, g.logprobs, g.text, thresholds)
                        if g.prompt_cached:                  # model miał być zwolniony – wynik nieważny
                            item = TaskFragility(task_id, NOT_RUN, answer=g.text, reason="cached")
                    except Exception as e:
                        item = TaskFragility(task_id, NOT_RUN, reason=f"error: {e}")
                result.tasks.append(item)
                report({"event": "f_task_done", "model": model, "task": item.as_dict()})
                if item.label == UNSUPPORTED:               # ta Ollama nie podaje prawdopodobieństw – dalej bez sensu
                    break
        finally:
            try:
                self._release(model)
            except Exception:
                pass
        return result


def suite_tasks(suite: dict, lang: str) -> list[tuple[str, str, int]]:
    """Zadania testu jakości w tej samej postaci, w jakiej idą do modelu (treść w danym języku, limit tokenów)."""
    return [(t["id"], t["prompt"][lang], int(t.get("max_tokens") or 256)) for t in suite["tasks"]
            if lang in t.get("prompt", {})]
