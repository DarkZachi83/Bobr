"""Hybrydy i modele SSM (v0.3.1, v0.3.2: mapa warstw z pliku, MTP): pamięć KV tylko w warstwach uwagi, reszta to stały stan.

Zgłoszenie z Reddita (r/Oobabooga, 30.10.2026): Swift-1.5-Qwen3.8-27B (architektura
qwen35) IQ4_XS, kontekst 90 000, KV q8_0. AI Bóbr liczył pamięć rozmowy dla 64 warstw
(11,9 GB, werdykt żółty), a llama-server zajął 18,4 GB karty bez wylewania do RAM –
pamięć KV ma tylko co czwarta warstwa."""
import io
import json
import unittest
from pathlib import Path
from dataclasses import replace

from aibobr.calculator.memory import memory_need, mtp_extra_bytes
from aibobr.calculator.verdict import evaluate
from aibobr.domain.enums import FitLevel, GpuVendor, KvCacheType, ModelSource
from aibobr.domain.models import RunSettings
from aibobr.domain.units import GIB, MIB, gib
from aibobr.knowledge.advisor import Advisor, render
from aibobr.sources.gguf_reader import LocalFile, arch_from_headers, read_header
from aibobr.sources.huggingface import arch_from_config
from tests.fixtures import QUANTS, gpu, pc
from tests.gguf_writer import gguf_bytes


def header(md):
    return arch_from_headers([read_header(LocalFile(io.BytesIO(gguf_bytes(md))))])


# Qwen 3.5 / 3.8 27B: 64 warstwy, co 4. z uwagą (4 głowice KV × 256), 48 warstw Gated DeltaNet
QWEN35_MD = [("general.architecture", "qwen35", "str"), ("qwen35.block_count", 64, "u32"),
             ("qwen35.attention.head_count", 24, "u32"), ("qwen35.attention.head_count_kv", 4, "u32"),
             ("qwen35.attention.key_length", 256, "u32"), ("qwen35.attention.value_length", 256, "u32"),
             ("qwen35.embedding_length", 5120, "u32"), ("qwen35.full_attention_interval", 4, "u32"),
             ("qwen35.ssm.conv_kernel", 4, "u32"), ("qwen35.ssm.state_size", 128, "u32"),
             ("qwen35.ssm.group_count", 16, "u32"), ("qwen35.ssm.inner_size", 6144, "u32"),
             ("qwen35.ssm.time_step_rank", 48, "u32")]
# Stan na warstwę: splot 3 × (6144 + 2·16·128) + macierz 128 × 6144, po 4 bajty
QWEN35_STATE = 48 * (3 * 10240 + 128 * 6144) * 4

QWEN35_CONFIG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {
    "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
    "hidden_size": 5120, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
    "linear_num_value_heads": 48, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 248320}}

# Plik ze zgłoszenia: 64 warstwy + 1 warstwa MTP na końcu i gotowa mapa warstw liniowych
SWIFT_MD = [x for x in QWEN35_MD if x[0] != "qwen35.block_count"] + [
    ("qwen35.block_count", 65, "u32"), ("qwen35.nextn_predict_layers", 1, "u32"),
    ("qwen35.attention.recurrent_layers", [True, True, True, False] * 16 + [False], "arr:bool")]
SWIFT_SETTINGS = RunSettings(context_tokens=90000, kv_cache_type=KvCacheType.Q8_0)
CALIBRATION = json.loads((Path(__file__).parent / "calibration" / "swift27b_rx7900xt_2026-10-30.json")
                         .read_text(encoding="utf-8"))


def swift_spec(md=QWEN35_MD):
    arch = header(md)
    return arch.to_spec(name="Swift-1.5-Qwen3.8-27B-Uncensored-MTP", source=ModelSource.GGUF_FILE,
                        quant="IQ4_XS", weights_bytes=gib(13.3), origin=None)


class GgufTest(unittest.TestCase):
    def test_qwen35_every_fourth_layer_has_kv(self):
        a = header(QWEN35_MD)
        self.assertEqual((a.n_layers, a.kv_layers, a.n_kv_heads, a.head_dim), (64, 16, 4, 256))
        self.assertEqual(a.recurrent_bytes, QWEN35_STATE)

    def test_interval_default_when_key_missing(self):
        md = [x for x in QWEN35_MD if x[0] != "qwen35.full_attention_interval"]
        self.assertEqual(header(md).kv_layers, 16)

    def test_per_layer_kv_heads_with_zeros(self):
        # LFM2: tablica głowic KV na warstwę, 0 = warstwa splotu (krótki stan)
        kv = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0]
        md = [("general.architecture", "lfm2", "str"), ("lfm2.block_count", 16, "u32"),
              ("lfm2.attention.head_count", 32, "u32"), ("lfm2.attention.head_count_kv", kv, "arr:u32"),
              ("lfm2.embedding_length", 2048, "u32"), ("lfm2.shortconv.l_cache", 3, "u32")]
        a = header(md)
        self.assertEqual((a.kv_layers, a.n_kv_heads, a.head_dim), (6, 8, 64))
        self.assertEqual(a.recurrent_bytes, 10 * 2048 * 2 * 4)

    def test_pure_mamba_has_no_kv(self):
        md = [("general.architecture", "mamba", "str"), ("mamba.block_count", 64, "u32"),
              ("mamba.embedding_length", 2560, "u32"), ("mamba.ssm.conv_kernel", 4, "u32"),
              ("mamba.ssm.inner_size", 5120, "u32"), ("mamba.ssm.state_size", 16, "u32")]
        a = header(md)
        self.assertEqual(a.kv_layers, 0)
        self.assertEqual(a.recurrent_bytes, 64 * (3 * 5120 + 16 * 5120) * 4)
        spec = a.to_spec(name="mamba", source=ModelSource.GGUF_FILE, quant="Q8_0",
                         weights_bytes=gib(3), origin=None)
        short = memory_need(spec, RunSettings(context_tokens=2048), QUANTS).kv_cache
        long = memory_need(spec, RunSettings(context_tokens=131072), QUANTS).kv_cache
        self.assertEqual(short, long)                        # nie rośnie z kontekstem
        self.assertEqual(short, a.recurrent_bytes)

    def test_ordinary_model_unchanged(self):
        md = [("general.architecture", "llama", "str"), ("llama.block_count", 32, "u32"),
              ("llama.attention.head_count", 32, "u32"), ("llama.attention.head_count_kv", 8, "u32"),
              ("llama.embedding_length", 4096, "u32")]
        a = header(md)
        self.assertEqual((a.kv_layers, a.recurrent_bytes), (None, None))


class HuggingFaceTest(unittest.TestCase):
    def test_qwen35_layer_types_match_gguf(self):
        a = arch_from_config(QWEN35_CONFIG)
        self.assertEqual((a.kv_layers, a.recurrent_bytes), (16, QWEN35_STATE))

    def test_full_attention_interval(self):
        text = dict(QWEN35_CONFIG["text_config"])
        del text["layer_types"]
        text["full_attention_interval"] = 4
        self.assertEqual(arch_from_config({"text_config": text}).kv_layers, 16)

    def test_nemotron_h_pattern(self):
        c = {"num_hidden_layers": 8, "num_attention_heads": 32, "num_key_value_heads": 8,
             "hidden_size": 4096, "hybrid_override_pattern": "M-M*-M-*",
             "mamba_d_state": 128, "mamba_n_heads": 128, "mamba_head_dim": 64,
             "n_groups": 8, "conv_kernel": 4}
        a = arch_from_config(c)
        self.assertEqual(a.kv_layers, 2)                     # „*” = uwaga, „-” = MLP bez stanu
        self.assertEqual(a.recurrent_bytes, 3 * (3 * (8192 + 2 * 8 * 128) + 128 * 8192) * 4)

    def test_sliding_window_layer_types_are_not_hybrid(self):
        c = {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512,
             "layer_types": ["sliding_attention", "full_attention"] * 2}
        self.assertIsNone(arch_from_config(c).kv_layers)


class SwiftRegressionTest(unittest.TestCase):
    """Przypadek ze zgłoszenia, składnik po składniku z logu llama-server
    (tests/calibration/swift27b_rx7900xt_2026-10-30.json)."""

    def setUp(self):
        self.log = CALIBRATION["log_mib"]
        self.spec = replace(swift_spec(SWIFT_MD), weights_bytes=int(self.log["model_gpu"] * MIB))
        self.hw = pc(ram_gb=32, platform="windows",
                     gpus=[gpu(name="AMD Radeon RX 7900 XT", vendor=GpuVendor.AMD, vram_gb=20)])
        self.need = memory_need(self.spec, SWIFT_SETTINGS, QUANTS)

    def test_file_with_mtp_layer(self):
        a = header(SWIFT_MD)
        self.assertEqual((a.n_layers, a.kv_layers, a.mtp_layers), (64, 16, 1))
        self.assertEqual(a.recurrent_bytes, QWEN35_STATE)    # warstwa MTP nie jest liniowa

    def test_kv_matches_log(self):
        n = self.need
        self.assertEqual((n.kv_layers, n.n_layers), (self.log["kv_layers"], 64))
        kv = (n.kv_cache - n.recurrent) / MIB
        # llama.cpp zaokrągla kontekst do 256 komórek (90 112) – stąd 4 MiB różnicy
        self.assertAlmostEqual(kv * 90112 / 90000, self.log["kv"], delta=1)
        self.assertLess(n.kv_cache, 3.2 * GIB)               # v0.3.0: 11,9 GB (64 warstwy)

    def test_recurrent_state_matches_log(self):
        copies = self.log["recurrent_copies"]                # 1 + 3 dla draft-mtp
        self.assertAlmostEqual(self.need.recurrent * copies / MIB, self.log["recurrent_total"], delta=0.1)

    def test_compute_buffer_fits_overhead(self):
        self.assertLess(abs(self.need.overhead / MIB - self.log["compute"]), 64)

    def test_mtp_extra_matches_llama_cpp_projection(self):
        extra = mtp_extra_bytes(self.spec, SWIFT_SETTINGS) / MIB
        self.assertAlmostEqual(extra, self.log["mtp_kv_f16"] + self.log["mtp_compute"]
                               + 3 * self.need.recurrent / MIB, delta=2)
        total = self.need.vram_part / MIB + extra
        self.assertLess(abs(total - self.log["fit_projection"]), 64)
        # Menedżer zadań pokazuje ~1,1 GB więcej – poza rachunkiem llama.cpp (ROCm, Windows)
        self.assertLess(CALIBRATION["task_manager_mib"] - self.log["fit_projection"], 1200)

    def test_verdict_green(self):
        v = evaluate(self.spec, self.hw, SWIFT_SETTINGS, QUANTS)
        self.assertEqual(v.level, FitLevel.GREEN)            # v0.3.0: żółty, „76% w karcie”

    def test_explanation_mentions_layers_and_mtp_estimate(self):
        from aibobr.sources.base import LoadedModel, Variant
        loaded = LoadedModel(name=self.spec.name, source=self.spec.source, arch=header(SWIFT_MD),
                             variants=(Variant("IQ4_XS", self.spec.weights_bytes),))
        spec = loaded.spec()
        _, e = Advisor(QUANTS).explain(loaded, spec, self.hw, SWIFT_SETTINGS)
        kv_line = next(line for line in e.need if line.key == "expl.need.kv_hybrid")
        self.assertIn("tylko 16 z 64 warstw", render(kv_line, "pl"))
        self.assertIn("only 16 of 64 layers", render(kv_line, "en"))
        mtp = next(line for line in e.notes if line.key == "note.mtp")
        self.assertIn("ok. 0,9 GB więcej", render(mtp, "pl"))
        self.assertIn("szacunkowe", render(mtp, "pl"))
        self.assertIn("estimate", render(mtp, "en"))

    def test_no_mtp_note_without_mtp_layers(self):
        self.assertEqual(mtp_extra_bytes(swift_spec(), SWIFT_SETTINGS), 0)


if __name__ == "__main__":
    unittest.main()
