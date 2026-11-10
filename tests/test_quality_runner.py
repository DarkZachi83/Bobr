"""Przebieg jakości 0.10: próby od zera, pamięć promptu, 1/3 próby, wynik, etykieta, baza, ocena ręczna."""
import tempfile
import threading
import unittest
from pathlib import Path

from aibobr.app.bench_db import BenchDb
from aibobr.bench.ollama import Generation
from aibobr.quality import checks, score, suites
from aibobr.quality.runner import QualityRunner, set_manual


class Bench:
    """Ollama w pamięci: odpowiedź = answers(prompt, wywołanie), pamięć promptu, gdy model nie zwolniony."""

    def __init__(self, answers, sticky=False, fail=()):
        self.answers, self.sticky, self.fail = answers, sticky, set(fail)
        self.loaded, self.last, self.calls = None, None, []

    def version(self):
        return "0.99"

    def unload(self, model):
        self.calls.append("unload")
        if not self.sticky:
            self.loaded, self.last = None, None

    def is_loaded(self, model):
        return self.loaded == model

    def load(self, model, num_ctx):
        self.calls.append(("load", num_ctx))
        self.loaded = model

    def generate(self, model, prompt, num_ctx, num_predict, on_chunk=None):
        self.calls.append(("generate", num_predict))
        if prompt in self.fail:
            raise OSError("connection reset")
        cached = 30 if self.last == prompt else 0
        self.last = prompt
        text = self.answers(prompt, len(self.calls))
        if on_chunk:
            on_chunk(text)
        return Generation(40, 0.01, 5, 0.1, 0.0, text, prompt_cached=cached, total_seconds=0.2, done_reason="stop")


def by_prompt(data, lang="pl"):
    """Wzorcowe odpowiedzi z pliku testu (Python – kod wzorcowy z tests.test_sandbox)."""
    from tests.test_sandbox import REFERENCE
    out = {}
    for task in data["tasks"]:
        spec = task["check"]
        for code_lang, prompt in task["prompt"].items():
            if spec["type"] == "python":
                out[prompt] = REFERENCE[spec["function"]]
            else:
                out[prompt] = checks.expected_text(spec, code_lang).split(" | ")[0]
    return out


class NoSandbox:
    def status(self):
        from aibobr.quality.sandbox import Status
        return Status(False, "unsupported")


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.h, self.s = suites.standard()
        self.perfect = {**by_prompt(self.h), **by_prompt(self.s)}
        self.db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")

    def runner(self, bench, sandbox=None):
        t = [0.0]                                                        # zegar idzie razem z czekaniem
        return QualityRunner(bench, sandbox or NoSandbox(), self.db, {"app_version": "0.10.0"},
                             clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s), unload_timeout=1)

    def test_perfect_model_without_sandbox_partial_coverage(self):
        bench = Bench(lambda p, n: self.perfect[p])
        events = []
        out = self.runner(bench).run(["m"], [(self.h, "pl"), (self.s, "pl"), (self.s, "en")], 1, events.append)
        h, spl, sen = out["runs"]
        self.assertEqual((h["correct"], h["evaluated"], h["total"]), (15, 15, 15))
        self.assertEqual((spl["correct"], spl["evaluated"], spl["not_evaluated"]), (12, 12, 3))   # Python nieoceniony
        self.assertEqual(spl["by_category_json"]["python"], {"total": 3, "evaluated": 0, "correct": 0})
        self.assertEqual(sen["by_level_json"]["hard"], {"total": 5, "evaluated": 4, "correct": 4})
        lab = score.label(h["summary"].overall, spl["summary"].overall)
        self.assertEqual((lab.kind, lab.partial), ("similar", True))
        q = [e for e in events if e["event"] == "q_task"]
        self.assertEqual(len(q), 45)
        en = next(e for e in q if e["lang"] == "en" and e["task"] == "S1")
        self.assertEqual(en["translation"], "Oblicz 17 × 23. Podaj samą liczbę.")   # polskie tłumaczenie pytania
        self.assertIsNone(next(e for e in q if e["lang"] == "pl")["translation"])
        session = self.db.quality_sessions()[0]
        self.assertEqual((session["status"], session["sandbox"], session["attempts"]), ("finished", "unsupported", 1))
        self.assertEqual(len(self.db.quality_runs(session["id"])), 3)

    def test_every_attempt_from_zero(self):
        bench = Bench(lambda p, n: self.perfect[p])
        self.runner(bench).run(["m"], [(self.h, "pl")], 1)
        self.assertEqual([c if isinstance(c, str) else c[0] for c in bench.calls[:3]], ["unload", "load", "generate"])
        self.assertEqual(bench.calls[1], ("load", 2048))

    def test_cached_prompt_is_retried_then_invalid(self):
        bench = Bench(lambda p, n: self.perfect[p], sticky=True)                # model nigdy się nie zwalnia
        h1 = {"suite_info": self.h["suite_info"], "tasks": [self.h["tasks"][0], self.h["tasks"][0]]}
        out = self.runner(bench).run(["m"], [(h1, "pl")], 1)
        first, second = out["runs"][0]["tasks"]
        self.assertEqual(first.correct, True)                                   # pierwsze od zera
        self.assertEqual((second.correct, second.reason), (None, "invalid"))    # dwa razy z pamięci – nieocenione

    def test_three_attempts_majority_and_differing(self):
        task = self.s["tasks"][0]                                               # 17 × 23
        one = {"suite_info": self.s["suite_info"], "tasks": [task]}
        answers = iter(["391", "390", "391", "390", "390", "391"])
        bench = Bench(lambda p, n: next(answers))
        out = self.runner(bench).run(["m"], [(one, "pl"), (one, "en")], 3)
        pl, en = (r["tasks"][0] for r in out["runs"])
        self.assertEqual((pl.correct, pl.differing, len(pl.attempts)), (True, True, 3))     # 2 z 3
        self.assertEqual((en.correct, en.differing), (False, True))                          # 1 z 3
        row = self.db.quality_tasks(out["runs"][0]["id"])[0]
        self.assertEqual((row["correct"], row["differing"]), (1, 1))
        self.assertIn('"answer": "390"', row["attempts_json"])

    def test_execution_error_not_counted_against_model(self):
        prompt = self.h["tasks"][0]["prompt"]["pl"]
        bench = Bench(lambda p, n: self.perfect[p], fail=[prompt])
        out = self.runner(bench).run(["m"], [(self.h, "pl")], 1)
        t = out["runs"][0]["tasks"][0]
        self.assertEqual((t.correct, t.reason), (None, "error"))
        self.assertEqual(out["runs"][0]["evaluated"], 14)

    def test_manual_grade_recomputes_run(self):
        bench = Bench(lambda p, n: self.perfect[p])
        out = self.runner(bench).run(["m"], [(self.s, "pl")], 1)
        rid = out["runs"][0]["id"]
        rows = [r for r in self.db.quality_tasks(rid) if r["reason"] == "not_run"]
        for r in rows:
            set_manual(self.db, rid, r["id"], True)
        run = self.db.quality_runs(out["session_id"])[0]
        self.assertEqual((run["evaluated"], run["correct"], run["manual"], run["not_evaluated"]), (15, 15, 3, 0))

    def test_stop(self):
        stop = threading.Event()
        bench = Bench(lambda p, n: (stop.set(), self.perfect[p])[1])
        out = self.runner(bench).run(["m"], [(self.h, "pl")], 1, stop=stop)
        self.assertEqual(out["status"], "stopped")
        self.assertEqual(self.db.quality_sessions()[0]["status"], "stopped")
        self.assertIsNone(bench.loaded)


class ScoreTest(unittest.TestCase):
    def tally(self, correct, evaluated, total=15):
        return score.Tally(total, evaluated, correct)

    def test_label_rules(self):
        self.assertEqual(score.label(self.tally(12, 15), self.tally(9, 15)).kind, "text")       # 80 vs 60
        self.assertEqual(score.label(self.tally(9, 15), self.tally(12, 15)).kind, "science")
        self.assertEqual(score.label(self.tally(12, 15), self.tally(11, 15)).kind, "similar")   # 6,7 pkt
        self.assertEqual(score.label(self.tally(6, 15), self.tally(3, 15)).kind, "weak")        # oba < 50%
        self.assertEqual(score.label(self.tally(12, 15), self.tally(10, 12)).partial, True)
        self.assertEqual(score.label(None, self.tally(1, 15)).kind, "none")

    def test_majority_and_estimate(self):
        self.assertEqual([score.majority(a) for a in ([True], [False, True, True], [True, False, None], [None])],
                         [True, True, False, None])
        self.assertAlmostEqual(score.estimate_seconds(45, 1, load_s=3, tps=60, avg_tokens=60), 45 * 4.5)
        self.assertAlmostEqual(score.estimate_seconds(45, 3, load_s=3, tps=60, avg_tokens=60), 3 * 45 * 4.5)


if __name__ == "__main__":
    unittest.main()
