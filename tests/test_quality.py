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


class RunnerTest(unittest.TestCase):
    def test_runs_every_task_once_with_preview(self):
        from aibobr.bench.ollama import OllamaBench
        answers = {t["prompt"]: "Zażółć gęślą jaźń" for t in PL["tests"]}
        calls = []

        def gen(payload):
            calls.append(payload)
            return {"response": answers.get(payload["prompt"], ""), "eval_count": 5, "eval_duration": 10**8}
        bench = OllamaBench("http://h", FakeHttp(json={"http://h/api/generate": gen}))
        events = []
        out = q.QualityRunner(bench, PL).run("m", events.append)
        self.assertEqual(len(out), 8)
        self.assertTrue(all(c["options"]["num_ctx"] == q.CONTEXT and c["options"]["temperature"] == 0
                            for c in calls))
        self.assertEqual(sum(e["event"] == "chunk" for e in events), 8)  # podgląd odpowiedzi
        self.assertTrue(out[0].passed)
        scores = q.score(out)
        self.assertEqual(scores["spelling"], (1, 1))                     # „Zażółć gęślą jaźń”
        self.assertEqual(scores["inflection"], (0, 4))
        self.assertEqual(scores["numerals"], (0, 1))                     # osobno (uwaga z Reddita)
        self.assertEqual(scores["knowledge"], (0, 1))
        self.assertNotIn("agree", scores)                                # jeden przebieg

    def test_repeats_report_agreement(self):
        from aibobr.bench.ollama import OllamaBench
        n = {"i": 0}

        def gen(payload):
            n["i"] += 1
            text = "Tomaszu" if n["i"] % 5 else "Tomku"                  # co piąta odpowiedź inna
            return {"response": text, "eval_count": 2, "eval_duration": 10**8}
        bench = OllamaBench("http://h", FakeHttp(json={"http://h/api/generate": gen}))
        one = dict(PL, tests=[TESTS["pl_vocative"]])
        events = []
        r = q.QualityRunner(bench, one, repeats=5).run("m", events.append)[0]
        self.assertEqual((r.runs, r.identical, r.passes, r.passed), (5, 4, 4, True))   # większość
        self.assertEqual(n["i"], 5)
        self.assertEqual(sum(e["event"] == "chunk" for e in events), 1)  # podgląd: pierwszy przebieg
        self.assertEqual(q.score([r])["agree"], (0, 1))                  # nie wszystkie identyczne

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
