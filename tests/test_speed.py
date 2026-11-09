"""Benchmark prędkości 0.9: progi (każda granica), sprawdzenia wiarygodności, procedura zadania,
długi kontekst z kalibracją, sesja z telemetrią i zapisem do bazy."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from aibobr.app.bench_db import BenchDb
from aibobr.bench import speed, telemetry
from aibobr.bench.ollama import Generation, OllamaBench
from aibobr.bench.runner import Stopped
from aibobr.bench.session import SpeedBenchmark
from aibobr.bench.speed import TaskResult
from tests.fake_http import FakeHttp

ROOT = Path(__file__).resolve().parents[1]


def suites():
    return [json.loads((ROOT / "data" / "benchmarks" / f"{s}.json").read_text(encoding="utf-8"))
            for s in ("pl_jezyk_v1", "en_format_v1")]


class ThresholdTest(unittest.TestCase):
    def test_every_generation_boundary(self):
        cases = [(30.0, "fast"), (29.99, "useful"), (15.0, "useful"), (14.99, "slow"), (6.0, "slow"),
                 (5.99, "too_slow"), (0.0, "too_slow"), (250.0, "fast")]
        self.assertEqual([speed.gen_level(v) for v, _ in cases], [lvl for _, lvl in cases])
        self.assertIsNone(speed.gen_level(None))

    def test_every_ttft_boundary(self):
        cases = [(0.2, "fast"), (1.0, "fast"), (1.001, "useful"), (3.0, "useful"), (3.001, "slow"),
                 (8.0, "slow"), (8.001, "too_slow")]
        self.assertEqual([speed.ttft_level(v) for v, _ in cases], [lvl for _, lvl in cases])
        self.assertIsNone(speed.ttft_level(None))

    def test_worse_measure_decides(self):
        self.assertEqual(speed.verdict(54.2, 0.4), ("fast", "both", "fast", "fast"))
        self.assertEqual(speed.verdict(54.2, 4.0), ("slow", "ttft", "fast", "slow"))
        self.assertEqual(speed.verdict(10.0, 0.5), ("slow", "gen", "slow", "fast"))
        self.assertEqual(speed.verdict(20.0, 2.0), ("useful", "both", "useful", "useful"))
        self.assertEqual(speed.verdict(None, 2.0), ("useful", "ttft", None, "useful"))
        self.assertEqual(speed.verdict(5.0, None), ("too_slow", "gen", "too_slow", None))
        self.assertEqual(speed.verdict(None, None), (None, None, None, None))

    def test_thresholds_from_settings_and_recompute(self):
        self.assertEqual(speed.thresholds_of({}), speed.DEFAULT_THRESHOLDS)
        bad = speed.thresholds_of({"bench_thresholds": {"gen_tps": [6, 15, 30], "ttft_s": "x"}})
        self.assertEqual(bad, speed.DEFAULT_THRESHOLDS)                  # rosnące tok/s – odrzucone
        mine = speed.thresholds_of({"bench_thresholds": {"gen_tps": [40, 20, 8]}})
        self.assertEqual(mine, {"gen_tps": [40.0, 20.0, 8.0], "ttft_s": [1.0, 3.0, 8.0]})
        run = {"status": "finished", "gen_tps": 35.0, "ttft_wall_median_ms": 800.0}
        self.assertEqual(speed.recompute(run, speed.DEFAULT_THRESHOLDS), "fast")   # werdykt z dnia testu
        self.assertEqual(speed.recompute(run, mine), "useful")                    # według nowych progów
        self.assertIsNone(speed.recompute({**run, "status": "unreliable"}, mine))


def task(n=1, **kw):
    base = dict(prompt_tokens=60, prompt_cached=0, gen_tokens=100, load_ms=0.0, prompt_eval_ms=50.0,
                eval_ms=2000.0, total_ms=2100.0, ttft_wall_ms=120.0, wall_ms=2200.0, done_reason="stop",
                response_text="odpowiedź", released=True, load_wall_ms=1500.0, preload_ms=1400.0)
    base.update(kw)
    return speed.check(TaskResult(n, f"t{n}", **base))


class ValidityTest(unittest.TestCase):
    def test_errors_are_excluded(self):
        for kw, err in (({"gen_tokens": 0}, "no_tokens"), ({"response_text": "  "}, "no_tokens"),
                        ({"eval_ms": 0.0}, "bad_timing"), ({"eval_ms": 1.0}, "bad_timing"),
                        ({"error": "execution: timed out"}, "execution: timed out")):
            r = task(**kw)
            self.assertEqual(r.error, err)
            self.assertEqual((r.counted_gen, r.counted_pp, r.counted_ttft), (False, False, False))

    def test_flags(self):
        r = task(gen_tokens=12)
        self.assertIn("few_tokens", r.flags)
        self.assertTrue(r.counted_gen)                                    # krótka – wchodzi do średniej
        r = task(prompt_cached=40)
        self.assertEqual((r.counted_gen, r.counted_pp, "cached" in r.flags), (True, False, True))
        r = task(prompt_cached=None)
        self.assertEqual((r.counted_pp, "cache_unverified" in r.flags), (True, True))
        r = task(released=False)
        self.assertEqual((r.counted_pp, "not_released" in r.flags), (False, True))
        r = task(load_ms=2500.0)
        self.assertEqual((r.counted_ttft, "loaded_during_task" in r.flags), (False, True))
        self.assertIn("length", task(done_reason="length").flags)

    def test_summary_weighted_mean_and_basis(self):
        tasks = [task(1, gen_tokens=1000, eval_ms=18000.0), task(2, gen_tokens=214, eval_ms=4400.0,
                                                                  ttft_wall_ms=300.0),
                 task(3, ttft_wall_ms=200.0, prompt_cached=50, gen_tokens=0)]          # błąd: brak tokenów
        s = speed.summarize(tasks)
        self.assertEqual((s["gen_tokens_total"], s["gen_ms_total"], s["gen_tps"]), (1214, 22400.0, 54.2))
        self.assertEqual(speed.basis(s), {"tps": 54.2, "tokens": 1214, "seconds": 22.4})
        self.assertEqual((s["ttft_wall_median_ms"], s["ttft_wall_max_ms"]), (210.0, 300.0))
        self.assertEqual((s["pp_tokens_total"], s["pp_tps"]), (120, 1200.0))
        self.assertEqual((s["tasks_counted"], s["tasks_error"], s["status"]), (2, 1, "finished"))
        self.assertEqual((s["verdict"], s["verdict_measure"]), ("fast", "both"))
        self.assertEqual((s["load_ms"], s["load_wall_ms"]), (1400.0, 1500.0))
        self.assertEqual(s["warnings"], [])

    def test_summary_warnings_and_unreliable(self):
        s = speed.summarize([task(1, gen_tokens=100, wall_ms=3000.0)])
        self.assertEqual(set(s["warnings"]), {"few_tokens_total", "overhead"})      # 100 < 300; narzut 30%
        self.assertAlmostEqual(s["overhead_ratio"], 0.3)
        tasks = [task(n) for n in range(1, 13)] + [task(n, error="execution: x") for n in (13, 14, 15)]
        s = speed.summarize(tasks)
        self.assertEqual((s["status"], s["verdict"]), ("unreliable", None))
        self.assertIn("too_many_errors", s["warnings"])
        two = [task(n) for n in range(1, 14)] + [task(n, error="execution: x") for n in (14, 15)]
        self.assertEqual(speed.summarize(two)["status"], "finished")                # 2 błędy – jeszcze tak

    def test_spread(self):
        runs = [{"gen_tps": 54.0, "status": "finished"}, {"gen_tps": 57.0, "status": "finished"},
                {"gen_tps": None, "status": "unreliable"}]
        self.assertEqual(speed.spread(runs), (54.0, 57.0))
        self.assertIsNone(speed.spread([]))


class FakeBench:
    """Ollama w pamięci: pamięć promptu, zwalnianie, wczytywanie z num_ctx, obcinanie promptu,
    tokenizator `cpt` znaków na token. Zegar aplikacji przesuwa się o czasy „silnika”."""

    def __init__(self, clock, cpt=3.1, tps=50.0, pp=1000.0, load_s=1.5, sticky=False, fail=(), real_delay=0.0):
        self.clock, self.cpt, self.tps, self.pp, self.load_s = clock, cpt, tps, pp, load_s
        self.real_delay = real_delay                                  # prawdziwy czas – żeby próbnik zdążył
        self.sticky, self.fail = sticky, set(fail)
        self.loaded, self.ctx, self.last = None, None, None
        self.calls = []

    def version(self):
        return "0.99-test"

    def unload(self, model):
        self.calls.append(("unload", model))
        if not self.sticky:
            self.loaded, self.last = None, None

    def is_loaded(self, model):
        return self.loaded == model

    def gpu_share(self, model):
        return 1.0 if self.loaded == model else None

    def _ensure(self, model, num_ctx):
        if self.loaded == model and self.ctx == num_ctx:
            return 0.0
        self.loaded, self.ctx, self.last = model, num_ctx, None
        self.clock.now += self.load_s
        return self.load_s

    def load(self, model, num_ctx):
        self.calls.append(("load", model, num_ctx))
        return self._ensure(model, num_ctx)

    def generate(self, model, prompt, num_ctx, num_predict, on_chunk=None):
        self.calls.append(("generate", model, num_ctx, num_predict))
        if any(f in prompt for f in self.fail):
            raise OSError("connection reset")
        load = self._ensure(model, num_ctx)
        if self.real_delay:
            time.sleep(self.real_delay)
        tokens = min(num_ctx, int(len(prompt) / self.cpt) + 12)          # Ollama obcina za długi prompt
        cached = tokens - 1 if self.last == prompt else 0
        self.last = prompt
        gen = min(num_predict, max(1, num_ctx - tokens))
        p_s, g_s = (tokens - cached) / self.pp, gen / self.tps
        self.clock.now += 0.01 + p_s
        if on_chunk:
            on_chunk("Odp")
        self.clock.now += g_s + 0.02
        return Generation(tokens, p_s, gen, g_s, load, "Odpowiedź " * 3, prompt_cached=cached,
                          total_seconds=load + p_s + g_s + 0.01, done_reason="length" if gen == num_predict else "stop")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.pack = speed.build_pack(suites())

    def test_pack_has_15_tasks_from_both_suites(self):
        self.assertEqual(len(self.pack["tasks"]), 15)
        self.assertEqual(self.pack["version"], "pl_jezyk_v1@1.1+en_format_v1@1.0")
        self.assertEqual({t["num_predict"] for t in self.pack["tasks"]}, {256, 512})

    def test_task_procedure_release_load_then_stream(self):
        bench = FakeBench(self.clock)
        r = speed.SpeedRunner(bench, self.pack, clock=self.clock, sleep=lambda s: None)
        bench.loaded = "m"                                       # poprzednio wczytany
        out = r.task("m", 1, self.pack["tasks"][0])
        self.assertEqual([c[0] for c in bench.calls], ["unload", "load", "generate"])
        self.assertEqual(bench.calls[1], ("load", "m", 2048))     # ten sam num_ctx co zadanie
        self.assertEqual((out.released, out.prompt_cached, out.load_ms), (True, 0, 0.0))
        self.assertEqual(out.load_wall_ms, 1500.0)
        self.assertAlmostEqual(out.ttft_wall_ms, 10 + out.prompt_eval_ms, places=1)   # zegar aplikacji
        self.assertGreater(out.wall_ms, out.total_ms)
        self.assertEqual((out.error, out.counted_gen, out.counted_pp), (None, True, True))

    def test_sticky_model_is_not_released(self):
        bench = FakeBench(self.clock, sticky=True)
        r = speed.SpeedRunner(bench, self.pack, clock=self.clock, sleep=lambda s: setattr(
            self.clock, "now", self.clock.now + s), unload_timeout=2)
        r.task("m", 1, self.pack["tasks"][0])
        out = r.task("m", 1, self.pack["tasks"][0])               # ten sam prompt, model nie zwolniony
        self.assertFalse(out.released)
        self.assertGreater(out.prompt_cached, 0)
        self.assertFalse(out.counted_pp)
        self.assertTrue({"cached", "not_released"} <= set(out.flags))

    def test_execution_error_is_recorded(self):
        bench = FakeBench(self.clock, fail=[self.pack["tasks"][2]["prompt"][:30]])
        r = speed.SpeedRunner(bench, self.pack, clock=self.clock, sleep=lambda s: None)
        out = r.run_pack("m")
        self.assertEqual(out[2].error, "execution: connection reset")
        self.assertEqual(sum(t.error is None for t in out), 14)

    def test_long_context_measures_real_input_after_calibration(self):
        for cpt in (2.2, 3.1, 4.6):                              # różne tokenizatory
            bench = FakeBench(Clock(), cpt=cpt)
            r = speed.SpeedRunner(bench, self.pack, clock=bench.clock, sleep=lambda s: None)
            lc = r.long_context("m", 8192)
            usable = 8192 - speed.LONG_RESERVE
            self.assertEqual(lc["valid"], 1, (cpt, lc))
            self.assertTrue(0.80 * usable <= lc["prompt_tokens"] <= 0.95 * usable, (cpt, lc["prompt_tokens"]))
            self.assertEqual((lc["num_ctx"], lc["reserve_tokens"], lc["gen_tokens"]), (8192, 128, 128))
            self.assertEqual(lc["free_tokens"], 8192 - lc["prompt_tokens"] - 128)
            self.assertIn("chars_per_token", lc["calibration_json"])
            self.assertEqual(lc["prompt_cached"], 0)
            self.assertGreater(lc["tg_tps"], 0)

    def test_long_context_truncation_and_short_input_are_invalid(self):
        from dataclasses import replace

        def runner_seeing(tokens_for_long):
            bench = FakeBench(Clock(), cpt=3.0)
            real = bench.generate

            def generate(m, p, c, n, on_chunk=None):
                g = real(m, p, c, n, on_chunk)
                return replace(g, prompt_tokens=tokens_for_long(c)) if len(p) > 2000 else g
            bench.generate = generate
            return speed.SpeedRunner(bench, self.pack, clock=bench.clock, sleep=lambda s: None)
        lc = runner_seeing(lambda c: 1000).long_context("m", 8192)        # model „widzi” tylko część
        self.assertEqual((lc["valid"], lc["invalid_reason"]), (0, "input_too_short"))
        self.assertEqual(len(lc["calibration_json"]["attempts"]), 2)      # jedna poprawka, potem koniec
        lc = runner_seeing(lambda c: c - 10).long_context("m", 8192)      # prawie cały kontekst – obcięcie
        self.assertEqual((lc["valid"], lc["invalid_reason"]), (0, "possible_truncation"))


class SessionTest(unittest.TestCase):
    SETTINGS = {"bench_stabilization_window_s": 0.05, "bench_stabilization_timeout_s": 0.2,
                "bench_baseline_s": 0.1}

    def collector(self, temps=lambda: None):
        class Gpu:
            def read(self):
                return [{"vram_used_mb": 900.0, "gpu_temp_c": temps()}]
        return telemetry.Collector(gpu=Gpu(), cpu_temp=temps)

    def run_session(self, stop=None, **kw):
        db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        clock = Clock()
        bench = FakeBench(clock, real_delay=0.03)
        runner = speed.SpeedRunner(bench, speed.build_pack(suites()), clock=clock, sleep=lambda s: None)
        events = []
        job = SpeedBenchmark(bench, runner.pack, db=db, collector=self.collector(), settings=self.SETTINGS,
                             info={"app_version": "0.9.0", "runtime_version": "0.99"}, runner=runner,
                             sampler_kw={"interval": 0.02}, **kw)
        out = job.run(["a", "b"], runs=2, contexts=(1024,), report=events.append, stop=stop)
        return db, out, events

    def test_two_models_two_runs_saved(self):
        db, out, events = self.run_session()
        self.assertEqual(out["status"], "finished")
        session = db.sessions()[0]
        self.assertEqual((session["status"], session["run_count"], session["runs_per_model"]), ("finished", 4, 2))
        self.assertEqual(json.loads(session["thresholds_json"]), speed.DEFAULT_THRESHOLDS)
        runs = db.runs(session["id"])
        self.assertEqual([(r["model"], r["run_no"]) for r in runs], [("a", 1), ("a", 2), ("b", 1), ("b", 2)])
        r = runs[0]
        self.assertEqual((r["status"], r["verdict"], r["stabilization"]), ("finished", "fast", "no_sensors"))
        self.assertEqual(r["gen_tps"], 50.0)                         # tempo udawanego silnika
        self.assertEqual(json.loads(r["thresholds_json"]), speed.DEFAULT_THRESHOLDS)
        self.assertEqual(json.loads(r["warnings_json"]), [])
        self.assertGreater(json.loads(r["baseline_json"])["samples"], 0)
        tasks = db.tasks(r["id"])
        self.assertEqual(len(tasks), 15)
        self.assertTrue(all(t["response_text"] and t["prompt_cached"] == 0 for t in tasks))
        lc = db.long_context(r["id"])
        self.assertEqual((len(lc), lc[0]["num_ctx"], lc[0]["valid"]), (1, 1024, 1))
        phases = {s["phase"] for s in db.telemetry(run_id=r["id"])}
        self.assertTrue({"cooling", "baseline", "task", "long_context"} <= phases, phases)
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds.count("s_run_done"), 4)
        self.assertIn("telemetry", kinds)
        self.assertEqual(kinds[-1], "s_done")

    def test_stop_marks_session_and_run(self):
        stop = threading.Event()
        db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        clock = Clock()
        bench = FakeBench(clock)
        runner = speed.SpeedRunner(bench, speed.build_pack(suites()), clock=clock, sleep=lambda s: None)
        original = runner.task

        def task_then_stop(model, n, t, on_chunk=lambda s: None):
            if n == 3:
                stop.set()
            return original(model, n, t, on_chunk)
        runner.task = task_then_stop
        job = SpeedBenchmark(bench, runner.pack, db=db, collector=self.collector(), settings=self.SETTINGS,
                             runner=runner, sampler_kw={"interval": 0.02})
        out = job.run(["a"], runs=1, stop=stop)
        self.assertEqual(out["status"], "stopped")
        session = db.sessions()[0]
        self.assertEqual(session["status"], "stopped")
        self.assertEqual(db.runs(session["id"])[0]["status"], "stopped")

    def test_cooling_timeout_without_dialog_gives_warning(self):
        flip = {"n": 0}

        def temps():
            flip["n"] += 1                                           # temperatura ciągle skacze
            return 40.0 + 40.0 * ((flip["n"] // 2) % 2)
        db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        clock = Clock()
        bench = FakeBench(clock)
        runner = speed.SpeedRunner(bench, speed.build_pack(suites()), clock=clock, sleep=lambda s: None)
        settings = {**self.SETTINGS, "bench_stabilization_window_s": 0.15, "bench_stabilization_timeout_s": 0.4}
        job = SpeedBenchmark(bench, runner.pack, db=db, collector=self.collector(temps), settings=settings,
                             runner=runner, sampler_kw={"interval": 0.02})
        job.run(["a"], runs=1)
        run = db.runs(db.sessions()[0]["id"])[0]
        self.assertEqual(run["stabilization"], "timeout_continued")
        self.assertIn("stabilization_incomplete", json.loads(run["warnings_json"]))


class ControllerSpeedTest(unittest.TestCase):
    def test_controller_runs_and_saves_to_app_database(self):
        import main
        from tests.fixtures import DESKTOP_3060
        with tempfile.TemporaryDirectory() as d:
            c = main.build_app(user_dir=Path(d) / "cfg", cache_dir=Path(d) / "cache")
            c.settings.update(SessionTest.SETTINGS)
            clock = Clock()
            bench = FakeBench(clock)
            runner = speed.SpeedRunner(bench, c.speed_pack(), clock=clock, sleep=lambda s: None)
            out = c.run_speed_benchmark(["ollama:llama3.1:8b"], DESKTOP_3060, lambda e: None, threading.Event(),
                                        contexts=(), bench=bench, collector=SessionTest().collector(),
                                        sampler_kw={"interval": 0.02}, runner=runner)
            self.assertEqual(out["status"], "finished")
            session = c.bench_db.sessions()[0]
            self.assertEqual((session["app_version"], session["runtime_version"], session["suite"]),
                             (__import__("aibobr").__version__, "0.99-test", "speed_v1"))
            self.assertEqual(c.bench_db.runs(session["id"])[0]["model"], "llama3.1:8b")


class OllamaLoadTest(unittest.TestCase):
    def test_load_request_and_extra_fields(self):
        host = "http://o"
        http = FakeHttp(json={f"{host}/api/generate": lambda p: (
            {"done": True, "load_duration": 1_500_000_000} if "prompt" not in p else
            {"response": "x", "done": True, "prompt_eval_count": 10, "prompt_eval_duration": 10_000_000,
             "eval_count": 5, "eval_duration": 100_000_000, "total_duration": 200_000_000,
             "done_reason": "stop"})})
        b = OllamaBench(host, http)
        self.assertEqual(b.load("m", 2048), 1.5)
        self.assertEqual(http.posts[0][1], {"model": "m", "keep_alive": "5m", "options": {"num_ctx": 2048}})
        g = b.generate("m", "p", 2048, 16)
        self.assertEqual((g.total_seconds, g.done_reason, g.prompt_cached), (0.2, "stop", None))


if __name__ == "__main__":
    unittest.main()
