import io
import tempfile
import unittest
from pathlib import Path

from aibobr.domain.enums import ModelSource
from aibobr.sources.base import SourceError, quant_from_filename
from aibobr.sources.gguf_reader import (
    LocalFile, arch_from_headers, load_gguf_file, quant_from_header, read_header,
)
from tests.gguf_writer import gguf_bytes, llama_like


def header(data: bytes):
    return read_header(LocalFile(io.BytesIO(data)))


class HeaderTest(unittest.TestCase):
    def test_llama_8b_like_architecture(self):
        h = header(llama_like())
        a = arch_from_headers([h])
        # Ten sam układ tensorów co Llama 3.1 8B → dokładnie jej liczba parametrów.
        self.assertEqual(a.params_total, 8_030_261_248)
        self.assertEqual((a.n_layers, a.n_kv_heads, a.head_dim, a.max_context),
                         (32, 8, 128, 131072))
        self.assertIsNone(a.n_experts)
        self.assertEqual(quant_from_header(h), "Q4_K_M")

    def test_big_tokenizer_arrays_are_skipped_not_kept(self):
        h = header(llama_like())
        self.assertIsNone(h.metadata["tokenizer.ggml.tokens"])
        self.assertIsNone(h.metadata["tokenizer.ggml.scores"])

    def test_explicit_key_and_value_length(self):
        md = [("general.architecture", "x", "str"), ("x.block_count", 2, "u32"),
              ("x.attention.head_count", 8, "u32"), ("x.attention.head_count_kv", 4, "u32"),
              ("x.embedding_length", 1024, "u32"),
              ("x.attention.key_length", 192, "u32"), ("x.attention.value_length", 128, "u32")]
        a = arch_from_headers([header(gguf_bytes(md))])
        self.assertEqual(a.head_dim, 160)   # (192 + 128) / 2 → K+V dokładnie jak w cache

    def test_per_layer_kv_heads_array_uses_maximum(self):
        md = [("general.architecture", "x", "str"), ("x.block_count", 3, "u32"),
              ("x.attention.head_count", 8, "u32"),
              ("x.attention.head_count_kv", [2, 4, 0], "arr:i32"),
              ("x.embedding_length", 512, "u32")]
        self.assertEqual(arch_from_headers([header(gguf_bytes(md))]).n_kv_heads, 4)

    def test_mla_cache_is_compressed(self):
        md = [("general.architecture", "deepseek2", "str"), ("deepseek2.block_count", 61, "u32"),
              ("deepseek2.attention.head_count", 128, "u32"),
              ("deepseek2.embedding_length", 7168, "u32"),
              ("deepseek2.attention.kv_lora_rank", 512, "u32"),
              ("deepseek2.rope.dimension_count", 64, "u32")]
        a = arch_from_headers([header(gguf_bytes(md))])
        self.assertEqual((a.n_kv_heads, a.head_dim), (1, 288))  # 2 × 288 = 512 + 64

    def test_moe_active_parameters(self):
        md = [("general.architecture", "m", "str"), ("m.block_count", 1, "u32"),
              ("m.expert_count", 8, "u32"), ("m.expert_used_count", 2, "u32")]
        tensors = [("token_embd.weight", (100, 10)), ("blk.0.ffn_up_exps.weight", (10, 20, 8))]
        a = arch_from_headers([header(gguf_bytes(md, tensors))])
        self.assertEqual(a.params_total, 1000 + 1600)
        self.assertEqual(a.params_active, 1000 + 400)
        self.assertEqual((a.n_experts, a.n_experts_used), (8, 2))

    def test_quant_falls_back_to_filename(self):
        h = header(gguf_bytes([("general.architecture", "x", "str")]))
        self.assertEqual(quant_from_header(h, "model-IQ4_XS.gguf"), "IQ4_XS")
        self.assertEqual(quant_from_header(h, "model.gguf"), "?")


class BrokenFileTest(unittest.TestCase):
    def assertCode(self, data: bytes, code: str):
        with self.assertRaises(SourceError) as ctx:
            header(data)
        self.assertEqual(ctx.exception.code, code)

    def test_not_gguf(self):
        self.assertCode(b"PK\x03\x04" + b"\0" * 40, "source.not_gguf")

    def test_old_version(self):
        self.assertCode(gguf_bytes([], version=1), "source.gguf_version")

    def test_truncated(self):
        self.assertCode(llama_like()[:5000], "source.bad_file")


class QuantFromFilenameTest(unittest.TestCase):
    def test_names(self):
        cases = {
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": "Q4_K_M",
            "Qwen3-30B-A3B-UD-Q4_K_XL.gguf": "Q4_K_XL",
            "gemma-3-12b-it-IQ4_XS.gguf": "IQ4_XS",
            "phi-4-bf16.gguf": "BF16",
            "Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf": "Q8_0",
            "gpt-oss-20b-MXFP4.gguf": "MXFP4",
            "model.gguf": None,
        }
        for name, quant in cases.items():
            self.assertEqual(quant_from_filename(name), quant, name)


class LocalFileTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_single_file(self):
        f = self.path / "Test-8B-Q4_K_M.gguf"
        f.write_bytes(llama_like() + b"\0" * 1000)       # „wagi” po nagłówku
        m = load_gguf_file(f)
        self.assertEqual(m.name, "Test Model 8B")
        self.assertEqual(m.source, ModelSource.GGUF_FILE)
        self.assertEqual(m.variants[0].size, f.stat().st_size)
        spec = m.spec()
        self.assertEqual((spec.quant, spec.weights_bytes, spec.n_layers),
                         ("Q4_K_M", f.stat().st_size, 32))

    def _split(self, count=3):
        md = [("general.architecture", "x", "str"), ("x.block_count", 3, "u32"),
              ("split.count", count, "u16")]
        for i in range(1, count + 1):
            (self.path / f"Big-Q8_0-{i:05d}-of-{count:05d}.gguf").write_bytes(
                gguf_bytes(md, [(f"blk.{i}.w", (10, 10))]) + b"\0" * 100)

    def test_split_file_sums_all_parts_from_any_part(self):
        self._split()
        m = load_gguf_file(self.path / "Big-Q8_0-00002-of-00003.gguf")
        self.assertEqual(len(m.variants[0].files), 3)
        self.assertEqual(m.arch.params_total, 300)       # tensory ze wszystkich części
        self.assertEqual(m.variants[0].size,
                         sum(p.stat().st_size for p in self.path.iterdir()))
        self.assertEqual(m.variants[0].quant, "Q8_0")    # brak file_type → z nazwy

    def test_split_file_with_missing_part(self):
        self._split()
        (self.path / "Big-Q8_0-00003-of-00003.gguf").unlink()
        with self.assertRaises(SourceError) as ctx:
            load_gguf_file(self.path / "Big-Q8_0-00001-of-00003.gguf")
        self.assertEqual(ctx.exception.code, "source.split_incomplete")

    def test_missing_file(self):
        with self.assertRaises(SourceError) as ctx:
            load_gguf_file(self.path / "nope.gguf")
        self.assertEqual(ctx.exception.code, "source.file_missing")


if __name__ == "__main__":
    unittest.main()
