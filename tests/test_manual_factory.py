import tempfile
import unittest
from pathlib import Path

from aibobr.calculator.verdict import evaluate
from aibobr.domain.enums import FitLevel, ModelSource
from aibobr.domain.models import RunSettings
from aibobr.sources.base import SourceError
from aibobr.sources.factory import load_model
from aibobr.sources.manual import ManualInput, load_manual
from tests.fake_http import FakeHttp
from tests.fixtures import DESKTOP_3060, QUANTS
from tests.gguf_writer import llama_like
from tests.test_huggingface import gguf_fake


class ManualTest(unittest.TestCase):
    def code(self, data):
        with self.assertRaises(SourceError) as ctx:
            load_manual(data, QUANTS)
        return ctx.exception.code

    def test_basic_level_is_an_estimate(self):
        m = load_manual(ManualInput(params_billion=8, quant="q4_k_m"), QUANTS)
        self.assertEqual(m.name, "8B")
        self.assertEqual(m.source, ModelSource.MANUAL)
        v = evaluate(m.spec(), DESKTOP_3060, RunSettings(), QUANTS)
        self.assertTrue(v.need.weights_estimated and v.need.kv_estimated)
        self.assertEqual(v.level, FitLevel.GREEN)

    def test_advanced_level_is_exact_for_kv(self):
        m = load_manual(ManualInput(8.03, "Q4_K_M", "Llama", 32, 8, 128, 131072), QUANTS)
        v = evaluate(m.spec(), DESKTOP_3060, RunSettings(), QUANTS)
        self.assertFalse(v.need.kv_estimated)
        self.assertEqual(v.need.kv_cache, 1024**3)

    def test_validation(self):
        self.assertEqual(self.code(ManualInput(0, "Q4_K_M")), "manual.params_required")
        self.assertEqual(self.code(ManualInput(8, "Q9")), "manual.unknown_quant")
        self.assertEqual(self.code(ManualInput(8, "Q4_K_M", n_layers=32)),
                         "manual.incomplete_architecture")
        self.assertEqual(self.code(ManualInput(8, "Q4_K_M", n_layers=-1, n_kv_heads=8, head_dim=1)),
                         "manual.not_positive")
        self.assertEqual(self.code(ManualInput(30, "Q4_K_M", active_params_billion=31)),
                         "manual.active_params_range")


class FactoryTest(unittest.TestCase):
    def code(self, text, http=None):
        with self.assertRaises(SourceError) as ctx:
            load_model(text, http or FakeHttp())
        return ctx.exception.code

    def test_local_file_path_even_with_quotes(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m-Q4_K_M.gguf"
            f.write_bytes(llama_like())
            http = FakeHttp()
            m = load_model(f'  "{f}" ', http)
            self.assertEqual(m.source, ModelSource.GGUF_FILE)
            self.assertEqual(http.calls, [])                     # bez internetu
            self.assertEqual(self.code(d), "source.is_directory")

    def test_missing_local_gguf(self):
        self.assertEqual(self.code("/nie/ma/takiego/pliku.gguf"), "source.file_missing")

    def test_huggingface_and_pasted_ollama_command(self):
        m = load_model("ollama run hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q8_0", gguf_fake())
        self.assertEqual(m.source, ModelSource.HUGGINGFACE)
        self.assertEqual(m.spec().quant, "Q8_0")

    def test_ollama_prefix(self):
        from tests.test_ollama_calibrate import FakeOllama
        from aibobr.sources.factory import source_kind
        fake = FakeOllama()
        m = load_model("ollama:llama3.1:8b", fake.http)
        self.assertEqual((m.name, m.origin), ("llama3.1:8b", "ollama:llama3.1:8b"))
        self.assertEqual(source_kind(" ollama:qwen3:8b "), "ollama")
        self.assertEqual(source_kind("hf.co/a/b"), "remote")
        self.assertEqual(self.code("ollama:"), "source.empty_input")

    def test_errors(self):
        self.assertEqual(self.code("   "), "source.empty_input")
        self.assertEqual(self.code("https://ollama.com/library/llama3.1"), "source.unsupported_site")
        self.assertEqual(self.code("lmstudio.ai/models/x"), "source.unsupported_site")
        self.assertEqual(self.code("zrób mi model"), "source.unrecognized_input")


if __name__ == "__main__":
    unittest.main()
