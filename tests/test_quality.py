"""Testy Zacharskiego – jakość (etap 12c): ocena odpowiedzi bez Ollamy."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from aibobr.bench import quality as q
from tests.fake_http import FakeHttp

ROOT = Path(__file__).resolve().parent.parent
PL = json.loads((ROOT / "data/benchmarks/pl_jezyk_v1.json").read_text(encoding="utf-8"))
EN = json.loads((ROOT / "data/benchmarks/en_format_v1.json").read_text(encoding="utf-8"))
TESTS = {t["id"]: t for t in PL["tests"] + EN["tests"]}


def ev(test_id):
    return TESTS[test_id]["evaluator"]


class PolishTest(unittest.TestCase):
    def test_patterns_and_cleaning(self):
        self.assertTrue(q.evaluate("„Zażółć gęślą jaźń.”", ev("pl_diacritics_restore")).passed)
        self.assertFalse(q.evaluate("Zazolc gesla jazn", ev("pl_diacritics_restore")).passed)
        self.assertTrue(q.evaluate("<think>hm</think>Chrząszcze brzmiały w trzcinie.",
                                   ev("pl_chrzaszcz_plural_past")).passed)
        self.assertTrue(q.evaluate("Ci dobrzy lekarze przyszli.", ev("pl_masculine_personal")).passed)
        self.assertFalse(q.evaluate("Te dobre lekarze przyszły.", ev("pl_masculine_personal")).passed)
        self.assertTrue(q.evaluate("Tomaszu", ev("pl_vocative")).passed)
        self.assertTrue(q.evaluate("jabłek", ev("pl_genitive_plural")).passed)

    def test_henrykowska_knowledge(self):
        good = "Daj, niech ja pomielę, a ty odpoczywaj – mąż mówi do żony przy żarnach."
        self.assertTrue(q.evaluate(good, ev("pl_henrykowska")).passed)
        bad = q.evaluate("To zdanie o modlitwie.", ev("pl_henrykowska"))
        self.assertFalse(bad.passed)
        self.assertTrue(bad.reason.startswith("quality.reason.missing:"))


class JsonTest(unittest.TestCase):
    def test_strict_and_lenient(self):
        v = q.evaluate('{"cpu": "Ryzen 7 5700X", "ram_gb": 32, "vram_gb": 8}', ev("json_hardware"))
        self.assertEqual((v.passed, v.strict), (True, True))
        v = q.evaluate('Here you go:\n```json\n{"cpu": "AMD Ryzen 7 5700X", "ram_gb": 32.0, "vram_gb": 8}\n```',
                       ev("json_hardware"))
        self.assertEqual((v.passed, v.strict), (True, False))           # łagodnie: z bloku ```
        v = q.evaluate('The primes: [2, 3, 5, 7, 11].', ev("json_primes"))
        self.assertEqual((v.passed, v.strict), (True, False))           # łagodnie: z tekstu

    def test_values_and_escaping(self):
        v = q.evaluate('{"name": "Anna", "age": 34, "skills": ["Python", "SQL"]}', ev("json_nested"))
        self.assertTrue(v.passed)                                        # bez rozróżniania wielkości liter
        self.assertFalse(q.evaluate('{"cpu": "Ryzen 7", "ram_gb": 16, "vram_gb": 8}', ev("json_hardware")).passed)
        good = json.dumps({"quote": 'She said "hi" and left \\ fast.'})
        self.assertTrue(q.evaluate(good, ev("json_escaping")).passed)
        self.assertEqual(q.evaluate("no json here", ev("json_hardware")).reason, "quality.reason.no_json")
        # Bielik-Minitron 7B v3.0, 5.11.2026: "\ " to niedozwolona sekwencja – JSON jest, ale błędny
        minitron = '{"quote": "She said \'hi\' and left \\ fast."}'
        self.assertEqual(q.evaluate(minitron, ev("json_escaping")).reason, "quality.reason.bad_json")


class PythonTest(unittest.TestCase):
    def test_syntax_only_never_runs(self):
        code = "```python\ndef is_palindrome(s):\n    s = s.replace(' ', '').lower()\n    return s == s[::-1]\n```"
        self.assertTrue(q.evaluate(code, ev("py_palindrome")).passed)
        self.assertFalse(q.evaluate("def is_palindrome(s)\n  return", ev("py_palindrome")).passed)
        boom = "import os\nos.remove('/nie/ma')\ndef fibonacci(n):\n    return n"
        self.assertTrue(q.evaluate(boom, ev("py_fibonacci")).passed)    # tylko parsowanie – nic się nie dzieje
        # Bielik-Minitron 7B v3.0, 5.11.2026: „Code:” przed poprawną funkcją – łagodnie ✓, ściśle ✗
        v = q.evaluate("Code:\n\ndef is_palindrome(s):\n    s = s.replace(' ', '').lower()\n"
                       "    return s == s[::-1]", ev("py_palindrome"))
        self.assertEqual((v.passed, v.strict), (True, False))
        v = q.evaluate("def fibonacci(n):\n    return [0, 1][:n]", ev("py_fibonacci"))
        self.assertEqual((v.passed, v.strict), (True, True))
        v = q.evaluate("```python\nclass Stack:\n    def push(self, i): pass\n    def pop(self): pass\n"
                       "    def is_empty(self): pass\n```\n\nThis class implements a stack.", ev("py_class"))
        self.assertEqual((v.passed, v.strict), (True, False))           # Bielik 11B: opis pod kodem
        stack = "class Stack:\n    def push(self, x): pass\n    def pop(self): pass\n"
        v = q.evaluate(stack, ev("py_class"))
        self.assertFalse(v.passed)
        self.assertIn("is_empty", v.reason)


class Server:
    """Udawana Ollama: /api/generate (zadania i zwalnianie), /api/ps. Pamięć promptu jak w
    eksperymencie kontrolnym: ten sam prompt drugi raz na wczytanym modelu = z pamięci."""

    def __init__(self, answer, cached_field=True, stuck=False):
        self.answer, self.cached_field, self.stuck = answer, cached_field, stuck
        self.loaded, self.seen, self.log = set(), set(), []

    def generate(self, payload):
        if "prompt" not in payload:                                  # keep_alive 0
            self.log.append("unload")
            if not self.stuck:
                self.loaded.discard(payload["model"])
                self.seen.clear()                                    # pamięć promptu znika z modelem
            return {"done": True}
        prompt = payload["prompt"]
        hit = payload["model"] in self.loaded and prompt in self.seen
        self.loaded.add(payload["model"])
        self.seen.add(prompt)
        self.log.append(("gen", prompt, hit, payload))
        out = {"response": self.answer(prompt, hit), "prompt_eval_count": 40, "load_duration": 0 if hit else 3 * 10**9}
        if self.cached_field:
            out["prompt_eval_cached_count"] = 39 if hit else 0
        return out

    def ps(self):
        return {"models": [{"name": m} for m in self.loaded]}

    def bench(self):
        from aibobr.bench.ollama import OllamaBench
        return OllamaBench("http://h", FakeHttp(json={"http://h/api/generate": self.generate,
                                                     "http://h/api/ps": self.ps}))


def gens(server):
    return [e for e in server.log if e != "unload"]


class RunnerTest(unittest.TestCase):
    def test_standard_releases_once_and_runs_every_task_once(self):
        answers = {t["prompt"]: "Zażółć gęślą jaźń" for t in PL["tests"]}
        server = Server(lambda p, hit: answers.get(p, ""))
        server.loaded.add("m")                                       # model po pomiarze wydajności
        events = []
        out = q.QualityRunner(server.bench(), PL, repeats=5).run("m", events.append)   # standard: zawsze 1
        self.assertEqual(len(out), 8)
        self.assertEqual(server.log[0], "unload")                    # raz, przed pakietem
        self.assertEqual(server.log.count("unload"), 1)
        self.assertEqual(len(gens(server)), 8)
        self.assertTrue(all(not hit for _, _, hit, _ in gens(server)))   # żadne zadanie z pamięci
        self.assertTrue(all(c["options"]["num_ctx"] == q.CONTEXT and c["options"]["temperature"] == 0
                            for *_, c in gens(server)))
        self.assertEqual(sum(e["event"] == "chunk" for e in events), 8)  # podgląd odpowiedzi
        self.assertTrue(all(r.mode == "standard" and r.runs == 1 and not r.cache_warn for r in out))
        self.assertEqual(out[0].attempts[0].released, None)
        scores = q.score(out)
        self.assertEqual(scores["spelling"], (1, 1))                     # „Zażółć gęślą jaźń”
        self.assertEqual(scores["inflection"], (0, 4))
        self.assertEqual(scores["numerals"], (0, 1))                     # osobno (uwaga z Reddita)
        self.assertEqual(scores["knowledge"], (0, 1))
        self.assertNotIn("agree", scores)                                # jeden przebieg

    def test_controlled_releases_before_every_attempt(self):
        """Odpowiedź z pamięci byłaby inna – w trybie kontrolowanym żadna próba jej nie dostaje."""
        server = Server(lambda p, hit: "Tomku" if hit else "Tomaszu")
        one = dict(PL, tests=[TESTS["pl_vocative"]])
        events = []
        r = q.QualityRunner(server.bench(), one, repeats=3, mode="controlled",
                            sleep=lambda s: None).run("m", events.append)[0]
        self.assertEqual(server.log, ["unload", gens(server)[0], "unload", gens(server)[1], "unload",
                                      gens(server)[2]])
        self.assertEqual((r.runs, r.identical, r.passes, r.passed, r.invalid), (3, 3, 3, True, 0))
        self.assertEqual([a.cached for a in r.attempts], [0, 0, 0])
        self.assertTrue(all(a.released and a.valid for a in r.attempts))
        self.assertEqual([a.n for a in r.attempts], [1, 2, 3])
        self.assertEqual(q.score([r])["agree"], (1, 1))
        self.assertEqual(sum(e["event"] == "chunk" for e in events), 1)  # podgląd: pierwsza próba

    def test_standard_with_cache_shows_why_repeats_need_release(self):
        """Bez zwalniania (jak w 0.8) kolejne próby trafiają w pamięć – widać to w cached i ostrzeżeniu."""
        server = Server(lambda p, hit: "Tomku" if hit else "Tomaszu")
        one = dict(PL, tests=[TESTS["pl_vocative"]])
        runner = q.QualityRunner(server.bench(), one, mode="standard")
        runner.repeats = 3                                               # wymuszone jak w starej wersji
        r = runner.run("m")[0]
        self.assertEqual((r.identical, r.passes, r.passed), (1, 1, False))
        self.assertEqual((r.cached, r.cache_warn), (39, True))

    def test_controlled_rejects_attempts_that_break_conditions(self):
        one = dict(PL, tests=[TESTS["pl_vocative"]])
        stuck = Server(lambda p, hit: "Tomaszu", stuck=True)             # model nie chce się zwolnić
        stuck.loaded.add("m")
        events = []
        r = q.QualityRunner(stuck.bench(), one, repeats=3, mode="controlled", unload_timeout=0.05,
                            sleep=lambda s: None).run("m", events.append)[0]
        self.assertEqual((r.runs, r.invalid, r.passed, r.reason), (0, 3, False, "quality.reason.no_valid"))
        self.assertTrue(any(e["event"] == "q_release_failed" for e in events))
        old = Server(lambda p, hit: "Tomaszu", cached_field=False)       # starsza Ollama – bez pola
        r = q.QualityRunner(old.bench(), one, repeats=3, mode="controlled", sleep=lambda s: None).run("m")[0]
        self.assertEqual((r.runs, r.invalid, r.cached, r.cache_warn), (3, 0, None, False))
        self.assertTrue(all(a.cached is None for a in r.attempts))      # brak pola to None, nie 0
        with tempfile.TemporaryDirectory() as d:
            import json
            import sqlite3
            from contextlib import closing
            from aibobr.app.bench_db import BenchDb
            db = BenchDb(Path(d) / "b.sqlite")
            db.add_quality(r, measured_at="t", runtime="ollama", runtime_version="x", hardware="h")
            with closing(sqlite3.connect(db.path)) as con:          # Windows: plik zamknięty przed sprzątaniem
                row = con.execute("SELECT prompt_tokens, prompt_cached, cache_warn, mode, invalid, attempts"
                                  " FROM quality").fetchone()
            self.assertEqual(row[:5], (40, None, 0, "controlled", 0))
            attempts = json.loads(row[5])
            self.assertEqual([a["n"] for a in attempts], [1, 2, 3])
            self.assertEqual(attempts[0]["text"], "Tomaszu")
            self.assertIsNone(attempts[0]["cached"])

    def test_controller_saves_quality(self):
        import main
        from tests.fixtures import DESKTOP_3060
        from tests.test_bench import fake_bench
        with tempfile.TemporaryDirectory() as d:
            c = main.build_app(user_dir=Path(d) / "cfg", cache_dir=Path(d) / "cache")
            bench, gen = fake_bench()
            out = c.run_benchmark(["ollama:llama3.1:8b"], [1024], {}, DESKTOP_3060, lambda e: None,
                                  threading.Event(), db=c.bench_db, bench=bench, vram_busy=0,
                                  suites=("perf", "en_format_v1"))
            self.assertEqual(sum(1 for r in out if isinstance(r, q.QualityResult)), 7)
            self.assertEqual(c.bench_db.quality_count(), 7)
            self.assertEqual(c.bench_db.count(), 1)


if __name__ == "__main__":
    unittest.main()
