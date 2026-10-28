"""Źródło Ollama i narzędzie kalibracyjne – na fałszywej Ollamie (bez serwera)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from aibobr.sources.base import SourceError
from aibobr.sources.ollama import DEFAULT_HOST, OllamaSource
from tests.fake_http import FakeHttp
from tests.fixtures import QUANTS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import calibrate  # noqa: E402

GIB, MIB = 1024**3, 1024**2
H = DEFAULT_HOST


def llama_info():
    return {"general.architecture": "llama", "general.file_type": 15,
            "general.parameter_count": 8_030_261_248, "llama.block_count": 32,
            "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
            "llama.embedding_length": 4096, "llama.context_length": 131072,
            "tokenizer.ggml.tokens": ["a", "b"]}


def gemma_info():
    return {"general.architecture": "gemma3", "general.parameter_count": 3_880_000_000,
            "gemma3.block_count": 34, "gemma3.attention.head_count": 8,
            "gemma3.attention.head_count_kv": 4, "gemma3.attention.key_length": 256,
            "gemma3.attention.value_length": 256, "gemma3.embedding_length": 2560,
            "gemma3.context_length": 131072, "gemma3.attention.sliding_window": 1024}


class FakeOllama:
    """Udaje serwer Ollama: modele, ładowanie z kontekstem, /api/ps i zajęty VRAM."""

    def __init__(self, vram_gib=8.0, overhead=0.6 * GIB):
        self.loaded = {}
        self.vram = vram_gib * GIB
        self.overhead = overhead
        self.models = {
            "llama3.1:8b": {"size": 4_920_000_000, "info": llama_info(), "quant": "Q4_K_M",
                            "kv_per_token": 128 * 1024},
            "gemma3:4b": {"size": 3_340_000_000, "info": gemma_info(), "quant": "Q4_K_M",
                          "kv_per_token": 0},
        }
        self.http = FakeHttp(json={
            f"{H}/api/version": {"version": "0.12.6"},
            f"{H}/api/tags": {"models": [
                {"name": n, "model": n, "size": m["size"],
                 "details": {"quantization_level": m["quant"]}} for n, m in self.models.items()]},
            f"{H}/api/ps": lambda: {"models": [dict(v) for v in self.loaded.values()]},
            f"{H}/api/show": self.show,
            f"{H}/api/generate": self.generate,
        })

    def show(self, payload):
        m = self.models[payload["model"]]
        assert payload["verbose"] is True
        return {"details": {"quantization_level": m["quant"]}, "model_info": m["info"],
                "tensors": [{"name": "token_embd.weight", "type": "Q4_K", "shape": [4096, 128256]}]}

    def generate(self, payload):
        name = payload["model"]
        if payload.get("keep_alive") == 0:
            self.loaded.pop(name, None)
            return {"done_reason": "unload"}
        m, ctx = self.models[name], payload["options"]["num_ctx"]
        size = int(m["size"] + m["kv_per_token"] * ctx + self.overhead)
        self.loaded[name] = {"name": name, "model": name, "size": size,
                             "size_vram": int(min(size, self.vram - 0.5 * GIB)), "context_length": ctx}
        return {"done": True}

    def nvidia_mib(self):
        return int((512 * MIB + sum(v["size_vram"] for v in self.loaded.values())) // MIB)


class OllamaSourceTest(unittest.TestCase):
    def test_load_model_from_show(self):
        fake = FakeOllama()
        m = OllamaSource(fake.http).load("llama3.1:8b")
        self.assertEqual((m.name, m.variants[0].quant, m.variants[0].size),
                         ("llama3.1:8b", "Q4_K_M", 4_920_000_000))
        a = m.arch
        self.assertEqual((a.n_layers, a.n_kv_heads, a.head_dim), (32, 8, 128))
        self.assertEqual(a.params_total, 4096 * 128256)       # z listy tensorów (tu tylko jeden)
        self.assertEqual(m.origin, "ollama:llama3.1:8b")

    def test_name_without_tag_and_sliding_window(self):
        fake = FakeOllama()
        fake.models["gemma3:latest"] = fake.models.pop("gemma3:4b")
        fake.http.json[f"{H}/api/tags"]["models"][1]["name"] = "gemma3:latest"
        m = OllamaSource(fake.http).load("gemma3")
        self.assertEqual((m.arch.sliding_window, m.arch.swa_layers), (1024, 29))   # 5×5 + 4

    def test_errors(self):
        with self.assertRaises(SourceError) as ctx:
            OllamaSource(FakeOllama().http).load("nie-ma:1b")
        self.assertEqual(ctx.exception.code, "ollama.not_found")

        class Down:
            def get_json(self, url):
                raise SourceError("source.network", url)
        with self.assertRaises(SourceError) as ctx:
            OllamaSource(Down()).installed()
        self.assertEqual(ctx.exception.code, "ollama.not_running")


class CalibratorTest(unittest.TestCase):
    def calibrator(self, fake):
        lines = []
        cal = calibrate.Calibrator(OllamaSource(fake.http), QUANTS, nvidia=fake.nvidia_mib,
                                   sleep=lambda s: None, log=lines.append)
        return cal, lines

    def test_measures_overhead_and_unloads(self):
        fake = FakeOllama(overhead=0.6 * GIB)
        cal, lines = self.calibrator(fake)
        results = cal.run(["llama3.1:8b"], [2048, 8192])
        self.assertEqual([m.context for m in results], [2048, 8192])
        for m in results:
            self.assertTrue(m.fully_in_vram)
            self.assertAlmostEqual(m.overhead / GIB, 0.6, delta=0.01)   # dokładnie tyle, ile udaje
        self.assertEqual(fake.loaded, {})                                # posprzątane
        self.assertTrue(any("8192" in line for line in lines))
        s = calibrate.summary(results)
        self.assertAlmostEqual(s["overhead_median_gb"], 0.6, delta=0.01)
        self.assertTrue(s["fully_in_vram"])

    def test_spill_uses_ollama_size_and_is_marked(self):
        fake = FakeOllama(vram_gib=4)
        cal, _ = self.calibrator(fake)
        [m] = cal.run(["llama3.1:8b"], [16384])
        self.assertFalse(m.fully_in_vram)
        self.assertEqual(m.measured, m.ollama_size)
        self.assertIn("część w RAM", calibrate.row(m))

    def test_sliding_window_model_is_marked(self):
        cal, _ = self.calibrator(FakeOllama())
        [m] = cal.run(["gemma3:4b"], [8192])
        self.assertTrue(m.kv_sliding)
        self.assertIn("okno przesuwne", calibrate.row(m))

    def test_main_writes_report(self):
        fake = FakeOllama()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "c.json"
            orig_client, orig_nv = calibrate.HttpClient, calibrate.nvidia_used_mib
            calibrate.HttpClient = lambda timeout=30: fake.http
            calibrate.nvidia_used_mib = fake.nvidia_mib
            orig_cal = calibrate.Calibrator.__init__

            def init(self, ollama, quants, **kw):
                orig_cal(self, ollama, quants, nvidia=fake.nvidia_mib, sleep=lambda s: None,
                         log=lambda *a: None)
            calibrate.Calibrator.__init__ = init
            try:
                code = calibrate.main(["--contexts", "4096", "--output", str(out)])
            finally:
                calibrate.HttpClient, calibrate.nvidia_used_mib = orig_client, orig_nv
                calibrate.Calibrator.__init__ = orig_cal
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["ollama"], "0.12.6")
            self.assertEqual(len(report["results"]), 2)
            self.assertIn("overhead_median_gb", report["summary"])


if __name__ == "__main__":
    unittest.main()
