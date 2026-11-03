"""Porównanie zapisanych modeli (etap 11b): klasa szybkości, miejsca, kontekst i kwantyzacja w karcie."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aibobr.app.saved import SavedModel
from aibobr.calculator import compare as cmp
from aibobr.calculator.verdict import evaluate
from aibobr.domain.enums import FitLevel
from aibobr.domain.models import RunSettings
from aibobr.domain.units import gib
from tests.fixtures import DESKTOP_3060, QUANTS, gpu, pc
from tests.test_advisor import LLAMA, LLAMA70, MOE, QWEN32

AUTO = RunSettings()


def run(model, hw=DESKTOP_3060, settings=AUTO, quant="Q4_K_M"):
    return cmp.compare(model, quant, hw, settings, QUANTS)


class SpeedTest(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(run(LLAMA).speed_class, cmp.IN_VRAM)                 # 8B w 12 GB
        self.assertEqual(run(QWEN32).speed_class, cmp.SPLIT)                  # 32B – pół na pół
        self.assertEqual(run(LLAMA70).speed_class, cmp.WONT_RUN)              # 70B przy 32 GB RAM
        self.assertEqual(run(LLAMA, hw=pc(ram_gb=32)).speed_class, cmp.CPU)

    def test_rank_class_first_then_bytes_read(self):
        comps = {"llama": run(LLAMA), "qwen": run(QWEN32), "moe": run(MOE), "70b": run(LLAMA70)}
        ranked = cmp.rank(comps)
        self.assertEqual(ranked["llama"].rank, 1)                            # całość w karcie
        self.assertIsNone(ranked["70b"].rank)                                # nie pójdzie – bez miejsca
        # MoE 30B-A3B i 32B dzielą kartę z RAM, ale MoE czyta na token ~10× mniej
        self.assertLess(ranked["moe"].rank, ranked["qwen"].rank)
        self.assertLess(run(MOE).cost, run(QWEN32).cost / 5)

    def test_errors_not_ranked(self):
        ranked = cmp.rank({"a": run(LLAMA), "b": cmp.Comparison(error="source.not_found")})
        self.assertEqual((ranked["a"].rank, ranked["b"].rank), (1, None))


class FitTest(unittest.TestCase):
    def test_max_context_is_the_green_edge(self):
        c = run(LLAMA)
        self.assertIsNotNone(c.max_fit)
        green = lambda ctx: evaluate(LLAMA.spec("Q4_K_M"), DESKTOP_3060,                 # noqa: E731
                                     replace(AUTO, context_tokens=ctx), QUANTS).level == FitLevel.GREEN
        self.assertTrue(green(c.max_fit))
        if c.max_fit < LLAMA.spec().max_context:
            self.assertFalse(green(c.max_fit + cmp.CONTEXT_STEP))

    def test_no_context_fits_when_weights_dont(self):
        self.assertIsNone(run(QWEN32).max_fit)

    def test_best_quant_in_vram(self):
        self.assertEqual(run(LLAMA).best_quant, "Q8_0")                      # 12 GB: nawet Q8_0
        small = run(LLAMA, hw=pc(gpus=[gpu(vram_gb=6)]))
        # 6 GB (5 użytkowe): IQ4_XS 4,1 GB + pamięć rozmowy 1 GB + narzut > 5 GB, IQ3_M 3,5 GB już tak
        self.assertEqual(small.best_quant, "IQ3_M")
        self.assertIsNone(run(LLAMA70).best_quant)

    def test_margin_only_when_green(self):
        c = run(LLAMA)
        self.assertGreater(c.margin, 0)
        self.assertAlmostEqual(c.margin_pct, c.margin / gib(11), places=3)  # 12 GB − 1 GB rezerwy
        self.assertIsNone(run(QWEN32).margin)
        self.assertEqual(run(LLAMA).download, gib(4.58))


class ControllerTest(unittest.TestCase):
    def test_compare_saved_loads_and_ranks_per_group(self):
        import main
        from tests.gguf_writer import llama_like
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "Test-8B-Q4_K_M.gguf"
            path.write_bytes(llama_like())
            c = main.build_app(user_dir=Path(d) / "cfg", cache_dir=Path(d) / "cache")
            ok = SavedModel("text", "test", str(path), "Q4_K_M", "2026-11-03 18:00", "green", 1.0, 8192, "f16")
            gone = replace(ok, target=str(Path(d) / "nie-ma.gguf"), name="nie ma")
            out = c.compare_saved([ok, gone], DESKTOP_3060, AUTO)
            self.assertEqual(out[ok.key].rank, 1)
            self.assertEqual(out[ok.key].on_disk, "file")
            self.assertIsNotNone(out[gone.key].error)


if __name__ == "__main__":
    unittest.main()
