"""Telemetria 0.9: źródła z brakami (None, nie 0), próbnik, stabilizacja temperatury, linia bazowa."""
import tempfile
import threading
import time
import unittest
from collections import namedtuple
from pathlib import Path

from aibobr.app.bench_db import BenchDb
from aibobr.bench import telemetry as tm

Temp = namedtuple("Temp", "label current high critical")
Mem = namedtuple("Mem", "total available")
Swap = namedtuple("Swap", "used sin sout")


class FakePsutil:
    def __init__(self):
        self.swap = Swap(512 * tm.MB, 0, 0)

    def cpu_percent(self, percpu=False):
        return [10.0, 30.0, 50.0, 70.0]

    def virtual_memory(self):
        return Mem(16000 * tm.MB, 9000 * tm.MB)

    def swap_memory(self):
        return self.swap


class SourcesTest(unittest.TestCase):
    def test_throttle_names(self):
        self.assertEqual(tm.throttle_names(0x4 | 0x40), "sw_power_cap,hw_thermal")
        self.assertEqual(tm.throttle_names(0), "none")
        self.assertIsNone(tm.throttle_names(None))

    def test_cpu_temperature_per_vendor(self):
        intel = {"coretemp": [Temp("Core 0", 50, 0, 0), Temp("Package id 0", 61, 0, 0)]}
        amd = {"k10temp": [Temp("Tctl", 72.5, 0, 0)], "nvme": [Temp("Composite", 40, 0, 0)]}
        self.assertEqual(tm.cpu_temperature(lambda: intel), 61.0)
        self.assertEqual(tm.cpu_temperature(lambda: amd), 72.5)
        self.assertIsNone(tm.cpu_temperature(lambda: {"nvme": [Temp("Composite", 40, 0, 0)]}))
        self.assertIsNone(tm.cpu_temperature(lambda: {}))

    def test_rapl_power_wraparound_and_no_access(self):
        root = Path(tempfile.mkdtemp())
        zone = root / "intel-rapl:0"
        zone.mkdir()
        (zone / "name").write_text("package-0\n")
        (zone / "max_energy_range_uj").write_text("1000000000\n")
        (zone / "energy_uj").write_text("999000000\n")
        rapl = tm.Rapl(root)
        self.assertIsNone(rapl.watts(0.0))                       # pierwsza próbka – brak różnicy
        (zone / "energy_uj").write_text("14000000\n")            # +15 J po przekręceniu licznika
        self.assertAlmostEqual(rapl.watts(1.0), 15.0)
        self.assertIsNone(tm.Rapl(Path(tempfile.mkdtemp())).watts(1.0))     # brak RAPL

    def test_swap_activity_only_on_linux(self):
        ps = FakePsutil()
        linux, windows = tm.CpuRam(ps, linux=True), tm.CpuRam(ps, linux=False)
        first = linux.read(0.0)
        self.assertEqual((first["cpu_pct"], first["cpu_cores_json"]), (40.0, [10.0, 30.0, 50.0, 70.0]))
        self.assertEqual((first["ram_used_mb"], first["ram_available_mb"], first["swap_used_mb"]),
                         (7000.0, 9000.0, 512.0))
        self.assertNotIn("swap_in_bps", first)
        ps.swap = Swap(512 * tm.MB, 4096, 0)
        self.assertEqual(linux.read(2.0)["swap_in_bps"], 2048.0)
        self.assertNotIn("swap_in_bps", windows.read(2.0))       # Windows: aktywność swap niedostępna

    def test_nvidia_smi_new_and_old_driver(self):
        line = "87, 64, 72.40, 2505, 6123, 8188, [N/A], 0x0000000000000004\n"
        calls = []

        def new(cmd, timeout=None):
            calls.append(cmd[1])
            return line
        gpu = tm.SmiGpu(new)
        row = gpu.read()[0]
        self.assertEqual((row["gpu_util_pct"], row["gpu_temp_c"], row["gpu_power_w"], row["vram_used_mb"]),
                         (87.0, 64.0, 72.4, 6123.0))
        self.assertIsNone(row["vram_temp_c"])
        self.assertEqual(row["gpu_throttle"], "sw_power_cap")
        gpu.read()
        self.assertEqual(len(calls), 2)

        def old(cmd, timeout=None):
            if "clocks_event_reasons.active" in cmd[1]:
                return None                                       # sterownik < 535 nie zna pola
            return line
        self.assertEqual(tm.SmiGpu(old).read()[0]["gpu_throttle"], "sw_power_cap")

        def oldest(cmd, timeout=None):
            return None if "reasons" in cmd[1] else "50, 40, 20.0, 1500, 1000, 8188\n"
        row = tm.SmiGpu(oldest).read()[0]
        self.assertEqual((row["gpu_util_pct"], row["gpu_throttle"], row["vram_temp_c"]), (50.0, None, None))
        none = tm.SmiGpu(lambda cmd, timeout=None: None)
        self.assertEqual((none.read(), none.read()), ([], []))

    def test_collector_missing_sources_are_none(self):
        class Gpus:
            def read(self):
                return [{"gpu_temp_c": 50.0}, {"gpu_temp_c": 70.0}]
        c = tm.Collector(cpu=None, gpu=Gpus(), rapl=None, cpu_temp=lambda: None, gpu_index=1)
        s = c.collect(0.0)
        self.assertEqual(s["gpu_temp_c"], 70.0)
        self.assertEqual(s["extra_json"], {"gpus": [{"gpu_temp_c": 50.0}]})
        self.assertTrue(all(s[k] is None for k in ("cpu_pct", "cpu_temp_c", "cpu_power_w", "ram_used_mb")))

        class Broken:
            def read(self):
                raise RuntimeError("GPU has fallen off the bus")
        s = tm.Collector(gpu=Broken(), cpu_temp=lambda: 40.0).collect(0.0)
        self.assertEqual((s["gpu_temp_c"], s["cpu_temp_c"]), (None, 40.0))


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Script:
    """Temperatury CPU/GPU zależne od czasu – udawany komputer, który stygnie."""

    def __init__(self, clock, cpu, gpu):
        self.clock, self.cpu, self.gpu = clock, cpu, gpu

    def collector(self):
        script = self

        class Gpu:
            def read(self):
                value = script.gpu(script.clock.now)
                return [{"gpu_temp_c": value}]
        return tm.Collector(gpu=Gpu(), cpu_temp=lambda: script.cpu(script.clock.now))


class Ticks:
    """Zamiast prawdziwego czekania: przesuwa zegar i robi próbki (1 na sekundę)."""

    def __init__(self, sampler, clock, cancel_at=None):
        self.sampler, self.clock, self.cancel_at = sampler, clock, cancel_at
        self.next = clock.now + 1.0

    def wait(self, seconds):
        self.clock.now += seconds
        while self.clock.now + 1e-9 >= self.next:
            self.sampler.sample_once()
            self.next += 1.0
        return self.cancel_at is not None and self.clock.now >= self.cancel_at

    def is_set(self):
        return False


def sampler_for(cpu, gpu, db=None, session="s"):
    clock = Clock()
    s = tm.Sampler(Script(clock, cpu, gpu).collector(), session, db=db, clock=clock,
                   stamp=lambda: f"2026-11-09T10:00:{int(clock.now) % 60:02d}+01:00")
    return s, clock


class StabilizationTest(unittest.TestCase):
    SET = {"bench_stabilization_window_s": 30, "bench_stabilization_tolerance_c": 1.0,
           "bench_stabilization_timeout_s": 300}

    def test_stability_states(self):
        rows = [(float(i), {"cpu_temp_c": 60.0, "gpu_temp_c": None}) for i in range(31)]
        self.assertEqual(tm.stability(rows, 30.0, 30, 1.0), "stable")
        self.assertEqual(tm.stability(rows[:10], 9.0, 30, 1.0), "waiting")          # za krótko
        rows[25] = (25.0, {"cpu_temp_c": 61.5, "gpu_temp_c": None})
        self.assertEqual(tm.stability(rows, 30.0, 30, 1.0), "waiting")              # skok 1,5 °C
        none = [(0.0, {"cpu_temp_c": None, "gpu_temp_c": None})]
        self.assertEqual(tm.stability(none, 0.0, 30, 1.0), "no_sensors")
        self.assertEqual(tm.stability([], 0.0, 30, 1.0), "waiting")

    def test_cooling_until_stable(self):
        # GPU stygnie 2 °C/s do 45 °C w 20 s, CPU stałe: stabilnie po 20 + 30 s okna
        s, clock = sampler_for(lambda t: 50.0, lambda t: max(45.0, 85.0 - 2 * t))
        msgs = []
        res = tm.cool_down(s, self.SET, run_id="r", stop=Ticks(s, clock), report=lambda k, kw: msgs.append(k))
        self.assertEqual(res.status, "stable")
        self.assertFalse(res.warning)
        self.assertTrue(50 <= res.seconds <= 52)
        self.assertEqual((res.gpu_temp_c, res.cpu_temp_c), (45.0, 50.0))
        self.assertEqual(msgs, ["bench.cooling.start"])
        self.assertTrue(all(x["phase"] == "cooling" and x["run_id"] == "r" for _, x in s.history))

    def test_timeout_without_dialog_continues_with_warning(self):
        s, clock = sampler_for(lambda t: 50.0 + (t % 4), lambda t: 60.0)          # CPU ciągle skacze
        res = tm.cool_down(s, self.SET, stop=Ticks(s, clock))
        self.assertEqual(res.status, "timeout_continued")
        self.assertTrue(res.warning)
        self.assertTrue(300 <= res.seconds <= 302)

    def test_wait_more_then_continue(self):
        s, clock = sampler_for(lambda t: 50.0 + (t % 4), lambda t: 60.0)
        answers, msgs = [False, True], []
        res = tm.cool_down(s, self.SET, on_timeout=lambda: answers.pop(0), stop=Ticks(s, clock),
                           report=lambda k, kw: msgs.append(k))
        self.assertEqual((res.status, answers), ("timeout_continued", []))
        self.assertTrue(600 <= res.seconds <= 603)                                 # drugi okres 300 s
        self.assertIn("bench.cooling.wait_more", msgs)

    def test_cancel_and_no_sensors(self):
        s, clock = sampler_for(lambda t: 50.0 + (t % 4), lambda t: None)
        self.assertEqual(tm.cool_down(s, self.SET, stop=Ticks(s, clock, cancel_at=5)).status, "cancelled")
        s, clock = sampler_for(lambda t: None, lambda t: None)
        res = tm.cool_down(s, self.SET, stop=Ticks(s, clock))
        self.assertEqual((res.status, res.warning), ("no_sensors", False))


class BaselineAndSamplerTest(unittest.TestCase):
    def test_baseline_summary_and_delta(self):
        s, clock = sampler_for(lambda t: 40.0 + (t % 2), lambda t: None)
        base = tm.take_baseline(s, {"bench_baseline_s": 10}, run_id="r", stop=Ticks(s, clock))
        self.assertEqual((base["seconds"], base["samples"]), (10, 10))
        self.assertEqual(base["mean"]["cpu_temp_c"], 40.5)
        self.assertIsNone(base["mean"]["gpu_temp_c"])                              # brak czujnika ≠ 0
        d = tm.delta({"cpu_temp_c": 70.0, "gpu_temp_c": 60.0}, base)
        self.assertEqual((d["cpu_temp_c"], d["gpu_temp_c"]), (29.5, None))
        s2, clock2 = sampler_for(lambda t: 40.0, lambda t: None)
        self.assertIsNone(tm.take_baseline(s2, None, stop=Ticks(s2, clock2, cancel_at=3)))

    def test_settings_fall_back_to_defaults(self):
        cfg = tm.settings_of({"bench_baseline_s": 0, "bench_stabilization_timeout_s": "x",
                              "bench_stabilization_window_s": 60})
        self.assertEqual((cfg["bench_baseline_s"], cfg["bench_stabilization_timeout_s"],
                          cfg["bench_stabilization_window_s"]), (10, 300, 60))

    def test_samples_saved_in_batches_with_nulls(self):
        db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        sid = db.start_session(started_at="2026-11-09T10:00:00+01:00")
        rid = db.start_run(sid, model="m", run_no=1, started_at="2026-11-09T10:00:00+01:00")
        s, clock = sampler_for(lambda t: 55.0, lambda t: None, db=db, session=sid)
        seen = []
        s.listeners.append(seen.append)
        s.sample_once()                                     # faza „idle” – do Monitora, nie do bazy
        s.set_phase("task", rid, task_no=3)
        for _ in range(7):
            clock.now += 1
            s.sample_once()
        self.assertEqual(len(db.telemetry(run_id=rid)), 5)  # paczka po 5
        s.stop()
        rows = db.telemetry(run_id=rid)
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(seen), 8)
        self.assertEqual((rows[0]["phase"], rows[0]["task_no"], rows[0]["cpu_temp_c"]), ("task", 3, 55.0))
        self.assertIsNone(rows[0]["gpu_temp_c"])
        self.assertIsNone(rows[0]["cpu_power_w"])

    def test_thread_samples_every_interval(self):
        s = tm.Sampler(tm.Collector(cpu_temp=lambda: 40.0), "s", interval=0.05)
        with s:
            time.sleep(0.32)
        self.assertTrue(4 <= len(s.history) <= 9, len(s.history))

    def test_real_collector_on_this_machine(self):
        c = tm.default_collector(run=lambda cmd, timeout=None: None)             # bez nvidia-smi
        try:
            sample = c.collect(time.monotonic())
        finally:
            c.close()
        self.assertIsNotNone(sample["ram_available_mb"])
        self.assertTrue(set(tm.NUMERIC) <= set(sample))


def _display_available() -> bool:
    try:
        import tkinter
        tkinter.Tk().destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(_display_available(), "brak tkintera albo ekranu")
class CoolingDialogTest(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from aibobr.i18n import set_language
        set_language("pl")
        self.root = tk.Tk()
        self.addCleanup(self.root.destroy)

    def test_buttons_and_texts(self):
        from aibobr.gui.dialogs.cooling import CoolingDialog
        d = CoolingDialog(self.root)
        self.assertEqual(d.title(), "Stabilizacja temperatury trwa dłużej niż oczekiwano")
        self.assertEqual((d.continue_button["text"], d.wait_button["text"]), ("Kontynuuj test", "Czekaj dalej"))
        d.continue_button.invoke()
        self.assertTrue(d.result)
        d = CoolingDialog(self.root)
        d.wait_button.invoke()
        self.assertFalse(d.result)

    def test_worker_thread_gets_answer_from_tk_thread(self):
        from aibobr.gui.dialogs.cooling import ThreadsafeAsk
        ask = ThreadsafeAsk(self.root, ask=lambda root: True)
        out = {}
        worker = threading.Thread(target=lambda: out.setdefault("v", ask()))
        worker.start()
        deadline = time.time() + 5
        while worker.is_alive() and time.time() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertEqual(out.get("v"), True)
        ask.close()
        no = ThreadsafeAsk(self.root, ask=lambda root: False)
        worker = threading.Thread(target=lambda: out.__setitem__("v", no()))
        worker.start()
        while worker.is_alive() and time.time() < deadline + 5:
            self.root.update()
            time.sleep(0.01)
        self.assertIs(out["v"], False)
        stop = threading.Event()
        stop.set()
        self.assertTrue(ThreadsafeAsk(self.root, ask=lambda root: False, stop=stop)())  # zatrzymany test


if __name__ == "__main__":
    unittest.main()
