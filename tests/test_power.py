"""Limit mocy kart NVIDIA i Dynamic Boost (pomiar autora 4.11.2026: laptop RTX 5050, 35 z 100 W)."""
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest import mock

from aibobr.detection import nvidia
from aibobr.domain.enums import GpuVendor
from aibobr.domain.models import GpuInfo, RunSettings
from tests.fixtures import QUANTS, pc
from tests.test_advisor import LLAMA, explain

POWER_Q = """==============NVSMI LOG==============
Attached GPUs                             : 1
GPU 00000000:01:00.0
    GPU Power Readings
        Average Power Draw                             : 4.52 W
        GPU Ceiling Power Limit
            Current Power Limit                        : 35.00 W
            Requested Power Limit                      : 35.00 W
            Default Power Limit                        : 35.00 W
        Min Power Limit                                : 5.00 W
        Max Power Limit                                : 100.00 W
    Module Power Readings
        Current Power Limit                            : N/A
        Max Power Limit                                : N/A
"""
LAPTOP = GpuInfo("NVIDIA GeForce RTX 5050 Laptop GPU", GpuVendor.NVIDIA, 8 * 1024**3, pci_bus="0000:01:00.0")


class FakeRun:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, args, timeout=None):
        self.calls.append(" ".join(args))
        for key, value in self.answers.items():
            if key in " ".join(args):
                return value
        return None


class ParseTest(unittest.TestCase):
    def test_first_values_of_gpu_block(self):
        self.assertEqual(nvidia.parse_power(POWER_Q), {"0000:01:00.0": (35.0, 100.0)})
        self.assertEqual(nvidia.parse_power(None), {})

    def test_capped_threshold(self):
        self.assertTrue(replace(LAPTOP, power_limit_w=35, power_max_w=100).power_capped)
        self.assertFalse(replace(LAPTOP, power_limit_w=80, power_max_w=100).power_capped)
        self.assertFalse(replace(LAPTOP, power_limit_w=175, power_max_w=175).power_capped)   # 2060 SUPER
        self.assertFalse(LAPTOP.power_capped)                                                 # brak danych


@unittest.skipUnless(sys.platform.startswith("linux"), "nvidia-powerd – tylko Linux")
class BoostTest(unittest.TestCase):
    def test_powerd_inactive(self):
        run = FakeRun({"-q -d POWER": POWER_Q, "list-unit-files": "nvidia-powerd.service disabled disabled"})
        g = nvidia.with_power([LAPTOP], run)[0]
        self.assertEqual((g.power_limit_w, g.power_max_w, g.dynamic_boost), (35.0, 100.0, False))

    def test_powerd_active(self):
        run = FakeRun({"-q -d POWER": POWER_Q, "is-active": "active\n"})
        self.assertTrue(nvidia.with_power([LAPTOP], run)[0].dynamic_boost)

    def test_no_systemctl_means_unknown(self):
        g = nvidia.with_power([LAPTOP], FakeRun({"-q -d POWER": POWER_Q}))[0]
        self.assertIsNone(g.dynamic_boost)


class DuringTestTest(unittest.TestCase):
    """Ostrzeżenie w oknie benchmarku (pomiar autora 4.11.2026, nvidia-powerd włączony: 55 W → 68–77 W)."""

    def test_capped_without_boost(self):
        from aibobr.bench.power import capped_during_test
        capped = replace(LAPTOP, power_limit_w=35, power_max_w=100, dynamic_boost=False)
        self.assertTrue(capped_during_test(capped, 34.6))
        self.assertFalse(capped_during_test(capped, 20.0))                   # daleko od limitu

    def test_no_warning_with_dynamic_boost(self):
        from aibobr.bench.power import capped_during_test
        boosted = replace(LAPTOP, power_limit_w=55, power_max_w=100, dynamic_boost=True)
        self.assertFalse(capped_during_test(boosted, 77.0))
        unknown = replace(boosted, dynamic_boost=None)                       # Windows: stanu usługi nie znamy
        self.assertFalse(capped_during_test(unknown, 70.0))                  # pobór ponad limit = limit dynamiczny
        self.assertTrue(capped_during_test(unknown, 54.0))


class NoteTest(unittest.TestCase):
    def test_note_only_when_capped_and_boost_off(self):
        capped = replace(LAPTOP, power_limit_w=35, power_max_w=100, dynamic_boost=False)
        _, e = explain(LLAMA, pc(gpus=[capped]))
        note = next(n for n in e.notes if n.key == "note.power_capped")
        self.assertEqual((note.fields["limit"], note.fields["max"]), ("35", "100"))
        for gpu in (replace(capped, dynamic_boost=True), replace(capped, dynamic_boost=None),
                    replace(capped, power_limit_w=95)):
            _, e = explain(LLAMA, pc(gpus=[gpu]))
            self.assertNotIn("note.power_capped", [n.key for n in e.notes])


class DbMigrationTest(unittest.TestCase):
    def test_old_database_gets_power_columns(self):
        from aibobr.app.bench_db import BenchDb
        from aibobr.bench.runner import PerfResult
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "b.sqlite"
            with closing(sqlite3.connect(path)) as db, db:          # baza z wersji 12a bez kolumn mocy
                db.execute("CREATE TABLE perf (id INTEGER PRIMARY KEY, measured_at TEXT NOT NULL, model TEXT NOT NULL,"
                           " runtime TEXT NOT NULL, runtime_version TEXT, hardware TEXT, power TEXT,"
                           " context INTEGER NOT NULL, pp_tps REAL, ttft_ms REAL, tg_tps REAL, prompt_tokens INTEGER,"
                           " gen_tokens INTEGER, gpu_share REAL, predicted TEXT, suite TEXT)")
                db.execute("INSERT INTO perf (measured_at, model, runtime, context, tg_tps) VALUES ('x','m','ollama',1024,21.6)")
            b = BenchDb(path)
            b.add_perf(PerfResult("m", 1024, tg_tps=56.4, power_draw_w=69.9), measured_at="y", runtime="ollama",
                       runtime_version="1", hardware="hw", power="ac", suite="perf_v1",
                       power_limit_w=35, power_max_w=100)
            with closing(sqlite3.connect(path)) as db:
                rows = db.execute("SELECT tg_tps, power_draw_w, power_max_w FROM perf ORDER BY id").fetchall()
            self.assertEqual(rows, [(21.6, None, None), (56.4, 69.9, 100.0)])


if __name__ == "__main__":
    unittest.main()
