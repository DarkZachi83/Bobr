"""tools/measure_vram.py: szczyty pamięci procesów z udawanego nvidia-smi."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import measure_vram as mv  # noqa: E402


class FakeSmi:
    """Kolejne odpowiedzi nvidia-smi: przed startem, potem rosnące zużycie."""

    def __init__(self, frames):
        self.frames, self.calls = frames, {"gpu": 0, "apps": 0}

    def __call__(self, args, timeout=None):
        kind = "gpu" if "--query-gpu=memory.used" in args[1] else "apps"
        total, apps = self.frames[min(self.calls[kind], len(self.frames) - 1)]
        self.calls[kind] += 1            # każde zapytanie ma własną kolejkę klatek
        if kind == "gpu":
            return str(total)
        return "\n".join(f"{pid}, {name}, {mem}" for pid, name, mem in apps)


FRAMES = [
    (600, [(1111, "/usr/lib/Xorg", 120)]),
    (3600, [(1111, "/usr/lib/Xorg", 120), (2222, "/usr/bin/llama-server", 2950)]),
    (4300, [(1111, "/usr/lib/Xorg", 125), (2222, "/usr/bin/llama-server", 3010),
            (3333, "/usr/bin/python3", 640)]),
    (3700, [(1111, "/usr/lib/Xorg", 120), (2222, "/usr/bin/llama-server", 3010)]),
]


class ParseTest(unittest.TestCase):
    def test_apps_and_missing_memory(self):
        apps = mv.parse_apps("1234, /usr/bin/llama-server, 2950\n99, C:\\a, b\\x.exe, [N/A]\nzła linia")
        self.assertEqual(apps, {1234: ("/usr/bin/llama-server", 2950), 99: ("C:\\a, b\\x.exe", None)})
        self.assertEqual(mv.parse_total("1200\n800"), 2000)            # dwie karty
        self.assertIsNone(mv.parse_total(None))


class SamplerTest(unittest.TestCase):
    def test_peaks_per_process(self):
        cmd = {2222: "llama-server -m orpheus.gguf -ngl 99", 3333: "python gguf_orpheus.py --text hej"}
        s = mv.Sampler(run=FakeSmi(FRAMES), cmdline=cmd.get)
        total_before, apps = s.baseline()
        for _ in range(3):
            s.sample()
        r = mv.Result("orpheus", 10, total_before, s.total_peak, list(s.processes.values()))
        self.assertEqual((r.total_before_mib, r.total_peak_mib, r.total_delta_mib), (600, 4300, 3700))
        new = r.new_processes()
        self.assertEqual([(p.pid, p.peak_mib) for p in new], [(2222, 3010), (3333, 640)])
        self.assertEqual(new[1].command, "python gguf_orpheus.py --text hej")   # dekoder osobno
        out = []
        mv.report(r, out.append)
        self.assertTrue(any("0.62 GB" in line and "gguf_orpheus.py" in line for line in out))

    def test_process_running_before_counts_its_growth(self):
        frames = [(3000, [(4444, "python launch.py", 2900)]),
                  (6900, [(4444, "python launch.py", 6800)]),
                  (3100, [(4444, "python launch.py", 3000)])]
        s = mv.Sampler(run=FakeSmi(frames), cmdline=lambda pid: "python launch.py --forge")
        s.baseline()
        for _ in range(2):
            s.sample()
        r = mv.Result("forge sdxl 1024", 30, 3000, s.total_peak, list(s.processes.values()))
        self.assertEqual(r.new_processes(), [])
        self.assertEqual([(p.pid, p.peak_mib) for p in r.grown_processes()], [(4444, 6800)])
        out = []
        mv.report(r, out.append)
        self.assertTrue(any("6.64 GB" in x and "przed 2.83 GB" in x for x in out))

    def test_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.json"
            sampler = mv.Sampler(run=FakeSmi(FRAMES), interval=0.01, cmdline=lambda pid: None)
            code = mv.main_cli(["--label", "test", "--output", str(out)], stdin=io.StringIO("\n"),
                               sampler=sampler)
            self.assertEqual(code, 0)
            self.assertIn('"label": "test"', out.read_text(encoding="utf-8"))


class TimelineTest(unittest.TestCase):
    def test_stages_of_forge_run(self):
        # Forge na ciepło: 2,9 GB → UNet 5,6 GB (10 s) → po zwolnieniu połowy UNet VAE 5,4 GB (2 s)
        mems = [2900] * 3 + [5600, 5650, 5620] * 7 + [3100, 5400, 5450, 5420, 5380] + [3000] * 3
        frames = [(m + 100, [(4444, "python launch.py", m)]) for m in [2900] + mems]
        clock = iter(i * 0.5 for i in range(1000))
        s = mv.Sampler(run=FakeSmi(frames), cmdline=lambda pid: "python launch.py", clock=lambda: next(clock))
        s.baseline()
        for _ in mems:
            s.sample()
        stages = mv.plateaus(mv.series(s.timeline, 4444))
        self.assertEqual([peak for _, _, peak in stages], [2900, 5650, 5450, 3000])
        self.assertLessEqual(len(mv.sparkline(mv.series(s.timeline, 4444), width=10)), 10)
        r = mv.Result("forge", 17, 3000, s.total_peak, list(s.processes.values()))
        out = []
        mv.report(r, out.append, timeline=s.timeline)
        self.assertTrue(any("przebieg pid 4444" in x for x in out))
        self.assertTrue(any("5.52 GB" in x for x in out))          # 5650 MiB – etap UNet


if __name__ == "__main__":
    unittest.main()
