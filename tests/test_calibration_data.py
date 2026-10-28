"""Pomiary z prawdziwego sprzętu (etap 8) jako test regresyjny obliczeń.

Każda zmiana w liczeniu pamięci musi nadal: (1) trafiać werdykt „całość w karcie” tam,
gdzie Ollama naprawdę zmieściła model w VRAM, i (2) nigdy nie przewidywać mniej pamięci,
niż zmierzył nvidia-smi – program może być ostrożny, ale nie może obiecywać za dużo.
"""
import json
import unittest
from pathlib import Path

from aibobr.calculator.verdict import evaluate
from aibobr.detection.profiles import load_profile
from aibobr.domain.enums import FitLevel, ModelSource
from aibobr.domain.models import ModelSpec, RunSettings
from aibobr.domain.units import GIB, gib
from tests.fixtures import QUANTS

HERE = Path(__file__).resolve().parent
DATASETS = [(HERE / "calibration" / "rtx5050_laptop_2026-10-28.json",
             HERE / "fake_hardware" / "real_laptop_rtx5050_linux.json")]


class MeasuredHardwareTest(unittest.TestCase):
    def test_measurements(self):
        for data_path, profile_path in DATASETS:
            data = json.loads(data_path.read_text(encoding="utf-8"))
            hw = load_profile(profile_path)
            for r in data["results"]:
                a = data["models"][r["model"]]
                spec = ModelSpec(name=r["model"], source=ModelSource.GGUF_FILE, quant="Q4_K_M",
                                 weights_bytes=gib(a["weights_gib"]), n_layers=a["n_layers"],
                                 n_kv_heads=a["n_kv_heads"], head_dim=a["head_dim"],
                                 sliding_window=a["sliding_window"], swa_layers=a["swa_layers"])
                v = evaluate(spec, hw, RunSettings(context_tokens=r["context"]), QUANTS)
                with self.subTest(model=r["model"], ctx=r["context"]):
                    self.assertEqual(v.level == FitLevel.GREEN, r["fits"])
                    if r["fits"]:
                        self.assertGreaterEqual(v.need.total / GIB + 0.005, r["nvidia_gib"])


if __name__ == "__main__":
    unittest.main()
