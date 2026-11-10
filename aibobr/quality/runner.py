"""Przebieg testów jakości 0.10: każde zadanie od zera, 1 albo 3 próby, ocena kodem, zapis do bazy.

Procedura próby (jak w benchmarku prędkości – zbadana powtarzalność odpowiedzi „od zera”):
zwolnienie modelu (potwierdzone w /api/ps) → wczytanie pustym zapytaniem → zadanie → sprawdzenie
pamięci promptu. Prompt częściowo z pamięci = próba nieważna i powtórzona raz; drugi raz z pamięci
→ zadanie nieocenione („invalid”), nie ocenione na siłę. Błąd wykonania (HTTP, brak pamięci) też
nieocenione – to nie wina odpowiedzi modelu.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Optional

from aibobr.bench.runner import Stopped
from aibobr.bench.speed import SpeedRunner
from aibobr.quality import checks, score

CONTEXT = 2048
ATTEMPTS = (1, 3)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class AttemptResult:
    answer: str
    correct: Optional[bool]
    format_compliant: Optional[bool]
    reason: str
    value: object = None
    prompt_tokens: Optional[int] = None
    gen_tokens: Optional[int] = None
    cached: Optional[int] = None
    released: bool = True
    done_reason: Optional[str] = None


@dataclass
class TaskResult:
    task_no: int
    task_id: str
    category: str
    level: str
    lang: str
    prompt: str
    translation: Optional[str]
    expected: str
    attempts: list = field(default_factory=list)
    correct: Optional[bool] = None
    format_compliant: Optional[bool] = None
    reason: str = ""
    value: object = None
    differing: bool = False
    manual: bool = False

    @property
    def answer(self) -> str:
        return self.attempts[0].answer if self.attempts else ""


def aggregate(t: TaskResult) -> TaskResult:
    """Jedna próba – jej wynik; trzy – zaliczone przy 2 z 3, format według zaliczonych prób."""
    t.correct = score.majority([a.correct for a in t.attempts])
    good = [a for a in t.attempts if a.correct]
    if t.correct:
        t.format_compliant = score.majority([a.format_compliant for a in good]) if len(good) > 1 \
            else good[0].format_compliant
    rep = next((a for a in t.attempts if a.correct == t.correct), t.attempts[0])
    t.reason, t.value = rep.reason, rep.value
    if t.correct and t.format_compliant is False:
        t.reason = checks.FORMAT
    t.differing = len({a.answer.strip() for a in t.attempts}) > 1
    return t


class QualityRunner:
    def __init__(self, bench, sandbox=None, db=None, info: Optional[dict] = None,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                 unload_timeout: float = 30.0):
        self.bench, self.sandbox, self.db, self.info = bench, sandbox, db, info or {}
        self.speed = SpeedRunner(bench, {"tasks": []}, clock=clock, sleep=sleep, unload_timeout=unload_timeout)

    def _python(self):
        if self.sandbox is None:
            return None
        return self.sandbox.run_tests if self.sandbox.status().ready else None

    def attempt(self, model: str, task: dict, lang: str, on_chunk: Callable[[str], None]) -> AttemptResult:
        prompt = task["prompt"][lang]
        for retry in range(2):
            released = self.speed.release(model)
            try:
                self.bench.load(model, CONTEXT)
                g = self.bench.generate(model, prompt, CONTEXT, int(task.get("max_tokens") or 256), on_chunk=on_chunk)
            except Stopped:
                raise
            except Exception as e:                                       # nie wina odpowiedzi modelu
                return AttemptResult("", None, None, "error", f"{e}"[:300], released=released)
            if not g.prompt_cached:
                break
        else:
            return AttemptResult(g.text, None, None, "invalid", None, g.prompt_tokens, g.gen_tokens,
                                 g.prompt_cached, released)
        c = checks.check(g.text, task["check"], lang, run_python=self._python())
        reason = c.reason
        if c.correct is False and g.done_reason == "length":
            reason = checks.TRUNCATED              # nie zdążył – osobny powód (decyzja autora, propozycja A)
        return AttemptResult(g.text, c.correct, c.format_compliant, reason, c.value, g.prompt_tokens,
                             g.gen_tokens, g.prompt_cached, released, g.done_reason)

    def run(self, models: list[str], tests: list[tuple[dict, str]], attempts: int = 1,
            report: Callable[[dict], None] = lambda e: None, stop: Optional[threading.Event] = None) -> dict:
        """tests: [(plik testu, język)] – np. humanistyczny/pl, ścisły/pl, ścisły/en."""
        stop = stop or threading.Event()
        attempts = attempts if attempts in ATTEMPTS else 1
        sandbox = self.sandbox.status().reason if self.sandbox is not None else "none"
        sid = None
        if self.db is not None:
            sid = self.db.start_quality_session(
                started_at=now_iso(), attempts=attempts, app_version=self.info.get("app_version"),
                runtime_version=self.info.get("runtime_version"), hardware=self.info.get("hardware"),
                models_json=models, sandbox=sandbox,
                tests_json=[f"{s['suite_info']['id']}/{lang}" for s, lang in tests])
        total = len(models) * sum(len(s["tasks"]) for s, _ in tests) * attempts
        report({"event": "q_start", "session_id": sid, "models": models, "attempts": attempts, "steps": total,
                "tests": [f"{s['suite_info']['id']}/{lang}" for s, lang in tests], "sandbox": sandbox,
                "version": self.info.get("runtime_version")})
        result = {"session_id": sid, "status": "finished", "runs": []}
        try:
            for model in models:
                for suite, lang in tests:
                    result["runs"].append(self._suite(sid, model, suite, lang, attempts, report, stop))
                self.speed.release(model)
                report({"event": "unload", "model": model})
        except Stopped:
            result["status"] = "stopped"
            report({"event": "stopped"})
            for model in models:
                if self.bench.is_loaded(model):
                    self.speed.release(model)
        finally:
            if self.db is not None and sid:
                self.db.finish_quality_session(sid, status=result["status"], finished_at=now_iso())
        report({"event": "q_done", "status": result["status"], "session_id": sid})
        return result

    def _suite(self, sid, model, suite, lang, attempts, report, stop) -> dict:
        info = suite["suite_info"]
        rid = self.db.start_quality_run(sid, model=model, suite=info["id"], suite_version=info.get("version"),
                                        lang=lang, started_at=now_iso()) if self.db is not None else None
        report({"event": "q_run", "model": model, "suite": info["id"], "lang": lang, "name": info["name"]})
        tasks = []
        for n, task in enumerate(suite["tasks"], 1):
            translation = task["prompt"].get("pl") if lang != "pl" else None
            t = TaskResult(n, task["id"], task["category"], task["level"], lang, task["prompt"][lang], translation,
                           checks.expected_text(task["check"], lang))
            for a in range(attempts):
                if stop.is_set():
                    if rid:
                        self.db.update_quality_run(rid, status="stopped", finished_at=now_iso())
                    raise Stopped()
                report({"event": "q_task", "model": model, "suite": info["id"], "lang": lang, "n": n,
                        "total": len(suite["tasks"]), "attempt": a + 1, "attempts": attempts, "task": task["id"],
                        "prompt": t.prompt, "translation": translation})
                t.attempts.append(self.attempt(model, task, lang, lambda s: report({"event": "chunk", "text": s})))
            aggregate(t)
            tasks.append(t)
            if rid:
                self.db.add_quality_task(
                    rid, task_no=n, task_id=t.task_id, category=t.category, level=t.level, prompt_text=t.prompt,
                    expected=t.expected, correct=None if t.correct is None else int(t.correct),
                    format_compliant=None if t.format_compliant is None else int(t.format_compliant),
                    reason=t.reason, value=_text(t.value), answer=t.answer, differing=int(t.differing),
                    attempts_json=[{**asdict(a), "value": _text(a.value)} for a in t.attempts])
            report({"event": "q_task_done", "model": model, "suite": info["id"], "lang": lang, "result": t})
        s = score.summarize([vars(t) for t in tasks])
        row = {"status": "finished", "finished_at": now_iso(), "total": s.overall.total,
               "evaluated": s.overall.evaluated, "correct": s.overall.correct,
               "format_noncompliant": s.format_noncompliant, "not_evaluated": s.overall.total - s.overall.evaluated,
               "manual": s.manual, "by_category_json": {k: v.as_dict() for k, v in s.by_category.items()},
               "by_level_json": {k: v.as_dict() for k, v in s.by_level.items()}}
        if rid:
            self.db.update_quality_run(rid, **row)
        out = {"id": rid, "model": model, "suite": info["id"], "lang": lang, "summary": s, "tasks": tasks, **row}
        report({"event": "q_run_done", "model": model, "suite": info["id"], "lang": lang, "result": out})
        return out


def _text(value) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _bool(v) -> Optional[bool]:
    return None if v is None else bool(v)


def set_manual(db, run_id: str, task_row: int, correct: bool) -> dict:
    """Ocena ręczna (zadania „oceniam sam” albo kod bez piaskownicy): zapis i przeliczenie wyniku
    przebiegu. Ręczne oceny mają własny licznik w wyniku."""
    db.update_quality_task(task_row, correct=int(bool(correct)), manual=1)
    rows = db.quality_tasks(run_id)
    s = score.summarize([{"category": r["category"], "level": r["level"], "correct": _bool(r["correct"]),
                          "format_compliant": _bool(r["format_compliant"]), "manual": r["manual"]} for r in rows])
    values = {"evaluated": s.overall.evaluated, "correct": s.overall.correct, "manual": s.manual,
              "not_evaluated": s.overall.total - s.overall.evaluated,
              "format_noncompliant": s.format_noncompliant,
              "by_category_json": {k: v.as_dict() for k, v in s.by_category.items()},
              "by_level_json": {k: v.as_dict() for k, v in s.by_level.items()}}
    db.update_quality_run(run_id, **values)
    return values
