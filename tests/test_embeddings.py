"""Tablica słownika (token_embd) w RAM – SPEC §5, pomiary autora z etapu 8.

llama.cpp (Ollama, LM Studio) nie trzyma tablicy wejściowej w karcie. Model z osobnym
wyjściem (Llama 3.1) zwalnia więc miejsce w VRAM, a model ze wspólnym wejściem i wyjściem
(Gemma) ma w RAM dodatkową kopię. Bez karty nic się nie zmienia.
"""
import unittest
from dataclasses import replace

from aibobr.calculator.memory import memory_need
from aibobr.calculator.verdict import evaluate
from aibobr.domain.models import RunSettings
from aibobr.domain.units import gib
from aibobr.sources.gguf_reader import GgufHeader, TensorInfo, arch_from_headers
from aibobr.sources.huggingface import arch_from_config
from tests.fixtures import DESKTOP_3060, LLAMA_31_8B, QUANTS, pc

LLAMA_EMBD = 128_256 * 4096


class SplitTest(unittest.TestCase):
    def test_untied_table_leaves_the_card(self):
        n = memory_need(LLAMA_31_8B, RunSettings(), QUANTS)
        self.assertEqual(n.embd_copy, 0)
        self.assertEqual(n.embd_in_ram, int(LLAMA_EMBD * 4.5 / 8))     # dolny szacunek bitów
        self.assertEqual(n.vram_part, n.total - n.embd_in_ram)
        self.assertEqual(n.total, n.weights + n.kv_cache + n.overhead)

    def test_tied_table_adds_a_copy_in_ram(self):
        gemma = replace(LLAMA_31_8B, embd_params=262_144 * 2560, tied_embeddings=True)
        n = memory_need(gemma, RunSettings(), QUANTS)
        self.assertEqual(n.embd_copy, n.embd_in_ram)
        self.assertEqual(n.vram_part, n.weights + n.kv_cache + n.overhead)  # karta bez zmian
        # Pomiar autora: „CPU_Mapped model buffer size = 525.00 MiB” (Q6_K) – trafiamy w MiB.
        self.assertAlmostEqual(n.embd_in_ram / 2**20, 525, delta=1)

    def test_unknown_table_changes_nothing(self):
        spec = replace(LLAMA_31_8B, embd_params=None, tied_embeddings=None)
        n = memory_need(spec, RunSettings(), QUANTS)
        self.assertEqual((n.embd_in_ram, n.embd_copy), (0, 0))

    def test_cpu_only_has_no_split(self):
        v = evaluate(LLAMA_31_8B, pc(ram_gb=32), RunSettings(), QUANTS)
        self.assertEqual(v.need.embd_in_ram, 0)
        self.assertEqual(v.in_ram, v.need.total)

    def test_green_keeps_only_the_table_in_ram(self):
        v = evaluate(LLAMA_31_8B, DESKTOP_3060, RunSettings(), QUANTS)
        self.assertEqual(v.in_ram, v.need.embd_in_ram)
        self.assertEqual(v.spill, 0)
        self.assertEqual(v.vram_fraction, 1.0)

    def test_table_needs_ram_too(self):
        # Karta duża, RAM zerowy po rezerwie – tablica nie ma się gdzie zmieścić.
        v = evaluate(LLAMA_31_8B, pc(ram_gb=2, gpus=DESKTOP_3060.gpus), RunSettings(), QUANTS)
        self.assertEqual(v.capacity.ram_usable, 0)
        self.assertNotEqual(v.level.name, "GREEN")


class SourcesTest(unittest.TestCase):
    def header(self, with_output: bool) -> GgufHeader:
        md = {"general.architecture": "llama", "llama.block_count": 2,
              "llama.embedding_length": 64, "llama.attention.head_count": 4}
        tensors = [TensorInfo("token_embd.weight", (64, 1000)),
                   TensorInfo("blk.0.attn_q.weight", (64, 64))]
        if with_output:
            tensors.append(TensorInfo("output.weight", (64, 1000)))
        return GgufHeader(3, md, tuple(tensors))

    def test_gguf_untied_and_tied(self):
        a = arch_from_headers([self.header(True)])
        self.assertEqual((a.embd_params, a.tied_embeddings), (64_000, False))
        b = arch_from_headers([self.header(False)])
        self.assertEqual((b.embd_params, b.tied_embeddings), (64_000, True))

    def test_hf_config(self):
        base = {"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 64,
                "vocab_size": 1000}
        self.assertFalse(arch_from_config({**base, "tie_word_embeddings": False}).tied_embeddings)
        self.assertTrue(arch_from_config({**base, "tie_word_embeddings": True}).tied_embeddings)
        # Brak pola = domyślne transformers (wspólne) – ostrożnie: nic nie odejmujemy od karty.
        self.assertTrue(arch_from_config(base).tied_embeddings)
        self.assertEqual(arch_from_config(base).embd_params, 64_000)


if __name__ == "__main__":
    unittest.main()
