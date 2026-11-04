"""Benchmark przez Ollamę (etap 12a) – bez Ollamy: udawane odpowiedzi API."""
import tempfile
import threading
import unittest
from pathlib import Path

from aibobr.app.bench_db import BenchDb
from aibobr.bench import prompts
from aibobr.bench.ollama import OllamaBench
from aibobr.bench.runner import PerfRunner, estimate_seconds
from tests.fake_http import FakeHttp
from tests.fixtures import DESKTOP_3060

HOST = "http://127.0.0.1:11434"
SUITE = {"suite_info": {"id": "perf_v1"}, "runs": 3, "warmup_runs": 1, "generate_tokens": 256}


class FakeOllama:
    """Odpowiedzi /api/generate: prompt 800 tok/s, generowanie 40 tok/s (z drobnym rozrzutem)."""

    def __init__(self):
        self.calls = []
        self.unloaded = []

    def __call__(self, payload):
        if "prompt" not in payload:                      # zwolnienie modelu (keep_alive 0)
            self.unloaded.append((payload["model"], payload["keep_alive"]))
            return {"done": True}
        self.calls.append(payload)
        n = len(self.calls)
        prompt_tokens = max(10, len(payload["prompt"]) // 4)
        gen = payload["options"]["num_predict"]
        return {"prompt_eval_count": prompt_tokens, "prompt_eval_duration": int(prompt_tokens / (800 + n) * 1e9),
                "eval_count": gen, "eval_duration": int(gen / (40 + n % 3) * 1e9),
                "load_duration": 2_000_000_000, "response": "beaver"}


def fake_bench():
    gen = FakeOllama()
    http = FakeHttp(json={f"{HOST}/api/generate": gen, f"{HOST}/api/version": {"version": "0.40.11"},
                          f"{HOST}/api/ps": {"models": [{"name": "llama3.1:8b", "size": 6_000, "size_vram": 6_000}]}})
    return OllamaBench(HOST, http), gen


class PromptTest(unittest.TestCase):
    def test_every_run_differs_and_fits_context(self):
        a, b = prompts.prompt_processing(8192, 1), prompts.prompt_processing(8192, 2)
        self.assertNotEqual(a[:40], b[:40])                       # inny początek – bez prompt cache
        self.assertLess(len(a) / prompts.CHARS_PER_TOKEN, 8192)
        self.assertGreater(len(a) / prompts.CHARS_PER_TOKEN, 8192 * 0.7)


class RunnerTest(unittest.TestCase):
    def test_medians_and_steps(self):
        bench, gen = fake_bench()
        steps = []
        r = PerfRunner(bench, SUITE).run("llama3.1:8b", 8192, steps.append)
        self.assertEqual(steps, ["warmup", "prompt 1/3", "prompt 2/3", "prompt 3/3",
                                 "generate 1/3", "generate 2/3", "generate 3/3"])
        self.assertEqual(len(gen.calls), 7)
        self.assertTrue(all(c["options"]["num_ctx"] == 8192 and c["think"] is False for c in gen.calls))
        self.assertTrue(all(c["options"]["temperature"] == 0 for c in gen.calls))
        self.assertAlmostEqual(r.pp_tps, 803, delta=3)
        self.assertAlmostEqual(r.tg_tps, 41, delta=1.5)
        self.assertEqual((r.gen_tokens, r.gpu_share), (256, 1.0))

    def test_stop_between_steps(self):
        bench, gen = fake_bench()
        stop = threading.Event()
        from aibobr.bench.runner import Stopped
        with self.assertRaises(Stopped):
            PerfRunner(bench, SUITE).run("llama3.1:8b", 1024, lambda s: stop.set() if s == "prompt 2/3" else None, stop)
        self.assertEqual(len(gen.calls), 3)      # rozgrzewka + 2 prompty – bieżący krok się kończy

    def test_estimate_grows_with_context_and_split(self):
        self.assertLess(estimate_seconds(1024, "green"), estimate_seconds(8192, "green"))
        self.assertLess(estimate_seconds(8192, "green"), estimate_seconds(8192, "yellow"))

    def test_estimate_matches_authors_measurements(self):
        # gemma2:9b na RTX 5050 Laptop (4.11.2026): 1k – 46 s, 8k – 62 s
        self.assertAlmostEqual(estimate_seconds(1024, "green", measured=(1000, 21.6)), 46, delta=10)
        self.assertAlmostEqual(estimate_seconds(8192, "yellow", measured=(896, 23.5)), 62, delta=12)

    def test_prompt_differs_between_sessions(self):
        import importlib
        first = prompts.prompt_processing(1024, 1)
        importlib.reload(prompts)                                  # nowa sesja programu
        self.assertNotEqual(first[:30], prompts.prompt_processing(1024, 1)[:30])


class ControllerTest(unittest.TestCase):
    def test_runs_models_one_by_one_skips_red_and_saves(self):
        import main
        with tempfile.TemporaryDirectory() as d:
            c = main.build_app(user_dir=Path(d) / "cfg", cache_dir=Path(d) / "cache")
            bench, gen = fake_bench()
            targets = ["ollama:llama3.1:8b", "ollama:qwen2.5-coder:14b"]
            predicted = {(targets[0], 1024): "green", (targets[0], 32768): "yellow",
                         (targets[1], 1024): "green", (targets[1], 32768): "red"}
            reports = []
            out = c.run_benchmark(targets, [1024, 32768], predicted, DESKTOP_3060, reports.append,
                                  threading.Event(), db=c.bench_db, bench=bench)
            self.assertEqual([(r.model, r.context, r.skipped) for r in out],
                             [("llama3.1:8b", 1024, None), ("llama3.1:8b", 32768, None),
                              ("qwen2.5-coder:14b", 1024, None), ("qwen2.5-coder:14b", 32768, "bench.skip_red")])
            self.assertEqual(c.bench_db.count(), 3)                    # czerwony – bez wiersza
            rows = c.bench_db.latest("llama3.1:8b", c.hardware_text(DESKTOP_3060))
            self.assertEqual([r["context"] for r in rows], [1024, 32768])
            self.assertEqual((rows[0]["runtime"], rows[0]["runtime_version"], rows[1]["predicted"]),
                             ("ollama", "0.40.11", "yellow"))
            self.assertEqual(gen.unloaded, [("llama3.1:8b", 0), ("qwen2.5-coder:14b", 0)])   # karta wolna po teście
            models_in_order = [cl["model"] for cl in gen.calls]
            self.assertEqual(models_in_order, sorted(models_in_order, key=targets_order(targets)))


def targets_order(targets):
    names = [t.split(":", 1)[1] for t in targets]
    return names.index


class DbTest(unittest.TestCase):
    def test_latest_per_context(self):
        from aibobr.bench.runner import PerfResult
        with tempfile.TemporaryDirectory() as d:
            db = BenchDb(Path(d) / "b.sqlite")
            for tps, when in ((30.0, "2026-11-03 10:00"), (35.0, "2026-11-04 10:00")):
                db.add_perf(PerfResult("m", 8192, tg_tps=tps), measured_at=when, runtime="ollama",
                            runtime_version="1", hardware="hw", power="ac", suite="perf_v1")
            self.assertEqual([r["tg_tps"] for r in db.latest("m", "hw")], [35.0])
            self.assertEqual(db.latest("m", "inny sprzęt"), [])


if __name__ == "__main__":
    unittest.main()
