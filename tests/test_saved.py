"""Zapisane modele do porównania (etap 11a): plik, limit 10 na grupę, odświeżanie wpisu."""
import datetime
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aibobr.app.controller import AppController
from aibobr.app.saved import MAX_PER_KIND, ListFull, SavedModel, SavedStore
from aibobr.calculator.verdict import evaluate
from aibobr.domain.models import RunSettings
from tests.fixtures import DESKTOP_3060, LLAMA_31_8B, QUANTS


def item(n=0, kind="text", **kw):
    base = dict(kind=kind, name=f"model {n}", target=f"https://huggingface.co/x/m{n}", quant="Q4_K_M",
                saved_at=f"2026-11-03 18:{n:02d}", level="green", need_gib=5.5, context=8192, kv_cache="f16")
    base.update(kw)
    return SavedModel(**base)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "saved_models.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_and_newest_first(self):
        s = SavedStore(self.path)
        s.add(item(1))
        s.add(item(2, kind="image", context=None, kv_cache=None, resolution=(1024, 1024)))
        s.add(item(3))
        again = SavedStore(self.path)
        self.assertEqual([i.name for i in again.of_kind("text")], ["model 3", "model 1"])
        self.assertEqual(again.of_kind("image")[0].resolution, (1024, 1024))

    def test_limit_per_group(self):
        s = SavedStore(self.path)
        for n in range(MAX_PER_KIND):
            s.add(item(n))
        with self.assertRaises(ListFull) as e:
            s.add(item(99))
        self.assertEqual(e.exception.kind, "text")
        s.add(item(99, kind="audio"))                      # inna grupa – osobny limit
        s.remove(s.of_kind("text")[0])
        self.assertTrue(s.add(item(99)))                  # po usunięciu jest miejsce

    def test_same_model_and_settings_is_refreshed_not_duplicated(self):
        s = SavedStore(self.path)
        self.assertTrue(s.add(item(1)))
        self.assertFalse(s.add(item(1, saved_at="2026-11-04 10:00", level="yellow")))
        self.assertEqual(len(s.items), 1)
        self.assertEqual(s.items[0].level, "yellow")
        self.assertTrue(s.add(item(1, context=32768)))    # inne ustawienia = osobny wpis

    def test_broken_file_means_empty_list(self):
        self.path.write_text("{to nie json", encoding="utf-8")
        self.assertEqual(SavedStore(self.path).items, [])
        self.path.write_text(json.dumps({"models": [{"kind": "text", "obce": 1}, item(1).__dict__]},
                                        default=list), encoding="utf-8")
        self.assertEqual([i.name for i in SavedStore(self.path).items], ["model 1"])


class EntryTest(unittest.TestCase):
    def test_entry_from_result(self):
        settings = RunSettings(context_tokens=16384)
        v = evaluate(LLAMA_31_8B, DESKTOP_3060, settings, QUANTS)
        e = AppController.saved_entry("hf.co/x/llama", LLAMA_31_8B, v, DESKTOP_3060, settings,
                                      now=datetime.datetime(2026, 11, 3, 18, 30))
        self.assertEqual((e.kind, e.quant, e.context, e.saved_at, e.level),
                         ("text", LLAMA_31_8B.quant, 16384, "2026-11-03 18:30", v.level.value))
        self.assertIn("RTX 3060 12 GB", e.hardware)
        self.assertIn("RAM 32 GB", e.hardware)


if __name__ == "__main__":
    unittest.main()
