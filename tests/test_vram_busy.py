"""Pamięć karty zajęta przez system (pomiar autora 4.11.2026: RTX 2060 SUPER z monitorem – pulpit
Windows 1151 MiB; gemma2:9b przy 8k – 72% w karcie, laptop z monitorem na grafice Intela – 87%)."""
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from aibobr.calculator.verdict import busy_vram, capacity
from aibobr.domain.enums import GpuVendor
from aibobr.domain.models import GpuInfo, RunSettings
from tests.fixtures import pc
from tests.test_advisor import LLAMA, explain
from tests.test_bench import fake_bench

MIB, GIB = 1024**2, 1024**3
DESKTOP = GpuInfo("NVIDIA GeForce RTX 2060 SUPER", GpuVendor.NVIDIA, 8 * GIB, vram_used=1151 * MIB,
                  pci_bus="0000:07:00.0")
IGPU = GpuInfo("Intel UHD Graphics", GpuVendor.INTEL, None, integrated=True)


class BusyTest(unittest.TestCase):
    def test_rounded_up_and_small_ignored(self):
        self.assertEqual(busy_vram(DESKTOP), int(1.25 * GIB))
        self.assertEqual(busy_vram(replace(DESKTOP, vram_used=180 * MIB)), 0)      # laptop, Linux
        self.assertEqual(busy_vram(replace(DESKTOP, vram_used=None)), 0)

    def test_ollama_model_is_not_system(self):
        self.assertEqual(busy_vram(replace(DESKTOP, vram_used=6 * GIB, vram_loaded=5 * GIB)), GIB)

    def test_capacity_takes_larger_of_reserve_and_busy(self):
        cap, _ = capacity(pc(gpus=[DESKTOP]), RunSettings())
        self.assertEqual(cap.vram_usable, int(6.75 * GIB))
        self.assertEqual(cap.busy, ((DESKTOP.name, int(1.25 * GIB), int(6.75 * GIB)),))
        cap, _ = capacity(pc(gpus=[replace(DESKTOP, vram_used=600 * MIB)]), RunSettings())
        self.assertEqual((cap.vram_usable, cap.busy), (7 * GIB, ()))                  # rezerwa wystarcza


class NoteTest(unittest.TestCase):
    def test_note_and_igpu_tip(self):
        _, e = explain(LLAMA, pc(gpus=[DESKTOP]))
        keys = [n.key for n in e.notes]
        self.assertIn("note.vram_busy", keys)
        self.assertNotIn("note.vram_busy_igpu", keys)                  # Ryzen 7 5700X – bez grafiki
        _, e = explain(LLAMA, pc(gpus=[DESKTOP, IGPU]))
        self.assertIn("note.vram_busy_igpu", [n.key for n in e.notes])

    def test_no_note_when_reserve_covers_it(self):
        _, e = explain(LLAMA, pc(gpus=[replace(DESKTOP, vram_used=180 * MIB)]))
        self.assertNotIn("note.vram_busy", [n.key for n in e.notes])


class SplitTipTest(unittest.TestCase):
    """Przy podziale karta + RAM liczy się każde ~100 MB (pomiar autora: 194 → 101 MiB, 85% → 87%, +6%)."""

    def test_tip_only_for_split(self):
        small = replace(DESKTOP, vram_total=4 * GIB, vram_used=None)            # 8B nie zmieści się w 3 GB
        _, e = explain(LLAMA, pc(gpus=[small]))
        self.assertIn("note.split_free_vram", [n.key for n in e.notes])
        _, e = explain(LLAMA, pc(gpus=[replace(small, vram_total=24 * GIB)]))   # całość w karcie
        self.assertNotIn("note.split_free_vram", [n.key for n in e.notes])
        _, e = explain(LLAMA, pc(gpus=[]))                                       # sam procesor
        self.assertNotIn("note.split_free_vram", [n.key for n in e.notes])


class ControllerTest(unittest.TestCase):
    def setUp(self):
        import main
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.c = main.build_app(user_dir=d / "cfg", cache_dir=d / "cache")

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_marks_ollama_model(self):
        hw = self.c.with_ollama_loaded(pc(gpus=[replace(DESKTOP, vram_used=6 * GIB), IGPU]), loaded=5 * GIB)
        self.assertEqual(hw.gpus[0].vram_loaded, 5 * GIB)
        self.assertEqual(hw.gpus[0].vram_busy, GIB)
        two = pc(gpus=[DESKTOP, replace(DESKTOP, pci_bus="0000:08:00.0")])
        self.assertIs(self.c.with_ollama_loaded(two, loaded=GIB), two)         # kilka kart – nie wiadomo

    def test_benchmark_saves_busy(self):
        bench, _ = fake_bench()
        target = "ollama:llama3.1:8b"
        hw = pc(gpus=[DESKTOP])
        self.c.run_benchmark([target], [1024], {(target, 1024): "green"}, hw, lambda p: None,
                             threading.Event(), db=self.c.bench_db, bench=bench, vram_busy=1151 * MIB)
        row = self.c.bench_db.latest("llama3.1:8b", self.c.hardware_text(hw))[0]
        self.assertEqual(row["vram_busy_mb"], 1151)

    def test_busy_now_fresh_read_minus_ollama(self):
        bench, _ = fake_bench()                                          # Ollama: wczytane 6000 B
        run = lambda args, timeout=None: "RTX 2060 SUPER, 8192, 1200, 00000000:07:00.0, 581.57, 7.5"  # noqa: E731
        self.assertEqual(self.c.vram_busy_now(pc(gpus=[DESKTOP]), bench, run), 1200 * MIB - 6000)
        no_smi = lambda args, timeout=None: None                          # noqa: E731
        self.assertEqual(self.c.vram_busy_now(pc(gpus=[DESKTOP]), None, no_smi), 1151 * MIB)


if __name__ == "__main__":
    unittest.main()
