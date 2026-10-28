"""Okno przesuwne (SPEC §16, etap 8): Gemma 3, gpt-oss – reguły z llama.cpp."""
import io
import unittest
from dataclasses import replace

from aibobr.calculator.memory import memory_need, swa_cells
from aibobr.domain.enums import ModelSource
from aibobr.domain.models import ModelSpec, RunSettings
from aibobr.domain.units import GIB, MIB
from aibobr.sources.base import count_swa_layers
from aibobr.sources.gguf_reader import LocalFile, arch_from_headers, read_header
from aibobr.sources.huggingface import arch_from_config
from tests.fixtures import QUANTS
from tests.gguf_writer import gguf_bytes

# Gemma 3 12B: 48 warstw, 8 głowic KV × 256, okno 1024, wzór 6 (5 z oknem + 1 pełna)
GEMMA3_12B = ModelSpec(name="Gemma 3 12B", source=ModelSource.HUGGINGFACE, quant="Q4_K_M",
                       params_total=11_770_000_000, weights_bytes=int(6.8 * GIB),
                       n_layers=48, n_kv_heads=8, head_dim=256, max_context=131072,
                       sliding_window=1024, swa_layers=40)
# gpt-oss-20b: 24 warstwy, 8 głowic KV × 64, okno 128, co druga warstwa
GPT_OSS_20B = ModelSpec(name="gpt-oss-20b", source=ModelSource.HUGGINGFACE, quant="MXFP4",
                        params_total=20_910_000_000, weights_bytes=int(11.3 * GIB),
                        n_layers=24, n_kv_heads=8, head_dim=64, max_context=131072,
                        sliding_window=128, swa_layers=12)


def kv(spec, ctx):
    return memory_need(spec, RunSettings(context_tokens=ctx), QUANTS)


class RulesTest(unittest.TestCase):
    def test_pattern_like_llama_cpp(self):
        self.assertEqual(count_swa_layers(48, 6), 40)              # Gemma 3
        self.assertEqual(count_swa_layers(24, 2), 12)              # gpt-oss
        self.assertEqual(count_swa_layers(10, 0), 10)              # 0 = wszystkie z oknem
        self.assertEqual(count_swa_layers(4, [True, False, True, True, True]), 3)

    def test_cells_window_plus_ubatch_padded(self):
        self.assertEqual(swa_cells(1024, 8192), 1536)
        self.assertEqual(swa_cells(128, 8192), 768)                # 640 → w górę do 256
        self.assertEqual(swa_cells(1024, 1000), 1000)              # krótki kontekst


class MemoryTest(unittest.TestCase):
    def test_gemma3_12b_at_8k(self):
        n = kv(GEMMA3_12B, 8192)
        # 8 warstw × 8192 + 40 warstw × 1536 komórek, po 8 KiB (2 × 8 × 256 × 2 B)
        self.assertEqual(n.kv_cache, (8 * 8192 + 40 * 1536) * 8192)
        self.assertTrue(n.kv_sliding)
        without = kv(replace(GEMMA3_12B, sliding_window=None, swa_layers=None), 8192)
        self.assertEqual(without.kv_cache, 3 * GIB)
        self.assertLess(n.kv_cache, without.kv_cache / 3)          # „nawet kilkukrotnie”

    def test_gpt_oss_grows_only_with_full_layers(self):
        k8, k32 = kv(GPT_OSS_20B, 8192).kv_cache, kv(GPT_OSS_20B, 32768).kv_cache
        self.assertEqual(k8, (12 * 8192 + 12 * 768) * 2048)
        self.assertAlmostEqual((k32 - k8) / MIB, 12 * (32768 - 8192) * 2048 / MIB)

    def test_short_context_is_not_marked_sliding(self):
        self.assertFalse(kv(GEMMA3_12B, 1024).kv_sliding)


def header(md):
    return arch_from_headers([read_header(LocalFile(io.BytesIO(gguf_bytes(md))))])


GEMMA_MD = [("general.architecture", "gemma3", "str"), ("gemma3.block_count", 48, "u32"),
            ("gemma3.attention.head_count", 16, "u32"), ("gemma3.attention.head_count_kv", 8, "u32"),
            ("gemma3.attention.key_length", 256, "u32"), ("gemma3.attention.value_length", 256, "u32"),
            ("gemma3.embedding_length", 3840, "u32"), ("gemma3.attention.sliding_window", 1024, "u32")]


class GgufTest(unittest.TestCase):
    def test_gemma3_default_pattern(self):
        a = header(GEMMA_MD)
        self.assertEqual((a.sliding_window, a.swa_layers), (1024, 40))

    def test_pattern_from_file_wins(self):
        md = GEMMA_MD + [("gemma3.attention.sliding_window_pattern", [True, False] * 24, "arr:bool")]
        self.assertEqual(header(md).swa_layers, 24)

    def test_unknown_architecture_keeps_full_context(self):
        md = [(k.replace("gemma3", "mystery"), v, t) if k != "general.architecture"
              else (k, "mystery", t) for k, v, t in GEMMA_MD]
        a = header(md)
        self.assertIsNone(a.sliding_window)                         # bezpieczniej zawyżyć

    def test_no_window_key(self):
        self.assertIsNone(header(GEMMA_MD[:-1]).swa_layers)


class ConfigTest(unittest.TestCase):
    def test_gpt_oss_layer_types(self):
        cfg = {"num_hidden_layers": 24, "num_attention_heads": 64, "num_key_value_heads": 8,
               "head_dim": 64, "hidden_size": 2880, "sliding_window": 128,
               "layer_types": ["sliding_attention", "full_attention"] * 12}
        a = arch_from_config(cfg)
        self.assertEqual((a.sliding_window, a.swa_layers), (128, 12))

    def test_gemma3_text_config_pattern(self):
        cfg = {"text_config": {"num_hidden_layers": 48, "num_attention_heads": 16,
                               "num_key_value_heads": 8, "head_dim": 256, "hidden_size": 3840,
                               "sliding_window": 1024, "sliding_window_pattern": 6}}
        self.assertEqual(arch_from_config(cfg).swa_layers, 40)

    def test_qwen_disabled_window_is_ignored(self):
        cfg = {"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
               "hidden_size": 4096, "sliding_window": 32768, "use_sliding_window": False}
        self.assertIsNone(arch_from_config(cfg).swa_layers)

    def test_window_without_layer_information_is_ignored(self):
        cfg = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
               "hidden_size": 4096, "sliding_window": 4096}                # stary Mistral
        self.assertIsNone(arch_from_config(cfg).swa_layers)


if __name__ == "__main__":
    unittest.main()
