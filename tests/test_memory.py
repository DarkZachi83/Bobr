import unittest
from dataclasses import replace

from aibobr.calculator.memory import MissingModelData, memory_need
from aibobr.domain.enums import KvCacheType, ModelSource
from aibobr.domain.models import ModelSpec, RunSettings
from aibobr.domain.units import GIB, gib
from tests.fixtures import LLAMA_31_8B, QUANTS, QWEN_25_32B


class KvCacheTest(unittest.TestCase):
    def test_llama_8b_is_exactly_1_gib_at_8k(self):
        # 2 × 32 warstwy × 8 głowic × 128 × 2 B = 128 KiB na token; × 8192 = 1 GiB
        need = memory_need(LLAMA_31_8B, RunSettings(context_tokens=8192), QUANTS)
        self.assertEqual(need.kv_cache, GIB)
        self.assertFalse(need.kv_estimated)

    def test_kv_grows_linearly_with_context(self):
        need = memory_need(LLAMA_31_8B, RunSettings(context_tokens=32768), QUANTS)
        self.assertEqual(need.kv_cache, 4 * GIB)

    def test_kv_quantization_shrinks_cache(self):
        q8 = memory_need(LLAMA_31_8B, RunSettings(kv_cache_type=KvCacheType.Q8_0), QUANTS)
        q4 = memory_need(LLAMA_31_8B, RunSettings(kv_cache_type=KvCacheType.Q4_0), QUANTS)
        self.assertEqual(q8.kv_cache, int(GIB * 34 / 64))
        self.assertEqual(q4.kv_cache, int(GIB * 18 / 64))

    def test_kv_estimate_without_architecture_is_flagged_and_conservative(self):
        bare = replace(LLAMA_31_8B, n_layers=None, n_kv_heads=None, head_dim=None)
        exact = memory_need(LLAMA_31_8B, RunSettings(), QUANTS)
        guess = memory_need(bare, RunSettings(), QUANTS)
        self.assertTrue(guess.kv_estimated)
        self.assertTrue(guess.is_estimate)
        self.assertGreater(guess.kv_cache, exact.kv_cache)  # ostrożnie = zawyżone


class WeightsTest(unittest.TestCase):
    def test_file_size_wins_over_estimate(self):
        need = memory_need(QWEN_25_32B, RunSettings(), QUANTS)
        self.assertEqual(need.weights, 19_850_000_000)
        self.assertFalse(need.weights_estimated)

    def test_estimate_from_bits_per_weight(self):
        manual = ModelSpec(name="x", source=ModelSource.MANUAL, quant="q4_k_m",
                           params_total=8_000_000_000)
        need = memory_need(manual, RunSettings(), QUANTS)
        self.assertEqual(need.weights, int(8e9 * 4.85 / 8))
        self.assertTrue(need.weights_estimated)

    def test_f16_is_two_bytes_per_weight(self):
        manual = ModelSpec(name="x", source=ModelSource.MANUAL, quant="F16",
                           params_total=3_000_000_000)
        self.assertEqual(memory_need(manual, RunSettings(), QUANTS).weights, 6_000_000_000)

    def test_total_is_sum_of_parts(self):
        need = memory_need(LLAMA_31_8B, RunSettings(overhead=gib(0.5)), QUANTS)
        self.assertEqual(need.total, 4_920_000_000 + GIB + gib(0.5))

    def test_missing_data_raises(self):
        with self.assertRaises(MissingModelData):
            memory_need(ModelSpec(name="x", source=ModelSource.MANUAL, quant="Q4_K_M"),
                        RunSettings(), QUANTS)
        with self.assertRaises(MissingModelData):
            memory_need(ModelSpec(name="x", source=ModelSource.MANUAL, quant="Q9_ZZ",
                                  params_total=1_000_000_000), RunSettings(), QUANTS)


if __name__ == "__main__":
    unittest.main()
