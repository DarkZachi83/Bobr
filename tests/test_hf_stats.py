"""Dane z Hugging Face dla zapisanych modeli (etap 11c) – bez sieci, z udawanego API."""
import datetime
import tempfile
import unittest
from pathlib import Path

from aibobr.app.saved import SavedModel, SavedStore
from aibobr.i18n import set_language
from aibobr.sources.huggingface import HF, hf_stats
from tests.fake_http import FakeHttp

LLAMA_INFO = {"id": "bartowski/Llama-3.1-8B-Instruct-GGUF", "downloads": 123456, "likes": 245,
              "lastModified": "2026-10-28T12:00:00.000Z", "gated": False,
              "cardData": {"license": "llama3.1", "base_model": ["meta-llama/Llama-3.1-8B-Instruct"]}}
FLUX_INFO = {"id": "black-forest-labs/FLUX.1-dev", "downloads": 1_543_210, "likes": 11000,
             "lastModified": "2025-08-01T00:00:00Z", "gated": "auto",
             "tags": ["diffusers", "license:other"], "cardData": {"license": "flux-1-dev-non-commercial-license"}}


def saved(name, target, kind="text"):
    return SavedModel(kind, name, target, "Q4_K_M", "2026-11-03 20:00", "green", 5.0, 8192, "f16")


class ParseTest(unittest.TestCase):
    def test_public_fields(self):
        http = FakeHttp(json={f"{HF}/api/models/bartowski/Llama-3.1-8B-Instruct-GGUF": LLAMA_INFO})
        s = hf_stats(http, "bartowski/Llama-3.1-8B-Instruct-GGUF")
        self.assertEqual((s["downloads"], s["likes"], s["updated"], s["license"]),
                         (123456, 245, "2026-10-28", "llama3.1"))
        self.assertFalse(s["noncommercial"] or s["gated"])
        self.assertEqual(s["base_model"], "meta-llama/Llama-3.1-8B-Instruct")

    def test_noncommercial_and_gated(self):
        http = FakeHttp(json={f"{HF}/api/models/black-forest-labs/FLUX.1-dev": FLUX_INFO})
        s = hf_stats(http, "black-forest-labs/FLUX.1-dev")
        self.assertTrue(s["noncommercial"])
        self.assertTrue(s["gated"])


class FetchTest(unittest.TestCase):
    def test_only_hf_models_are_fetched_and_saved(self):
        import main
        with tempfile.TemporaryDirectory() as d:
            c = main.build_app(user_dir=Path(d) / "cfg", cache_dir=Path(d) / "cache")
            http = FakeHttp(json={f"{HF}/api/models/bartowski/Llama-3.1-8B-Instruct-GGUF": LLAMA_INFO})
            c.http = lambda: http
            hf = saved("llama", "https://huggingface.co/bartowski/Llama-3.1-8B-Instruct-GGUF")
            via_ollama = saved("llama2", "ollama:hf.co/bartowski/Llama-3.1-8B-Instruct-GGUF:Q8_0")
            local = saved("plik", str(Path(d) / "model.gguf"))
            gone = saved("brak", "https://huggingface.co/nikt/nie-ma")
            ollama = saved("ollama", "ollama:llama3.1:8b")
            for item in (hf, via_ollama, local, gone, ollama):
                c.saved.add(item)
            out = c.fetch_hf_stats(c.saved.items, today=datetime.date(2026, 11, 3))
            self.assertEqual(set(out), {hf.key, via_ollama.key, gone.key})     # plik i Ollama pominięte
            self.assertEqual(out[gone.key]["error"], "source.not_found")
            again = SavedStore(Path(d) / "cfg" / "saved_models.json")            # zostaje w pliku
            by_name = {i.name: i for i in again.items}
            self.assertEqual(by_name["llama"].hf["downloads"], 123456)
            self.assertEqual(by_name["llama"].hf["fetched"], "2026-11-03")
            self.assertIsNone(by_name["brak"].hf)
            c.saved.add(hf)                                                     # ponowny zapis nie gubi danych
            self.assertEqual(next(i for i in c.saved.items if i.name == "llama").hf["likes"], 245)


class FormatTest(unittest.TestCase):
    def tearDown(self):
        set_language("en")

    def test_counts(self):
        from aibobr.i18n import fmt_count
        set_language("pl")
        self.assertEqual([fmt_count(n) for n in (999, 12_345, 1_543_210, None)],
                         ["999", "12,3 tys.", "1,5 mln", ""])
        set_language("en")
        self.assertEqual(fmt_count(12_000), "12k")


if __name__ == "__main__":
    unittest.main()
