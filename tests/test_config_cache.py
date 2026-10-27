import json
import tempfile
import unittest
from pathlib import Path

from aibobr.app.config import ConfigStore, run_settings
from aibobr.domain.enums import KvCacheType
from aibobr.domain.units import gib
from aibobr.sources.cache import ModelCache
from aibobr.sources.huggingface import HfRef, HuggingFaceSource
from tests.test_huggingface import REPO, gguf_fake

ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.user = Path(self.tmp.name)
        self.store = ConfigStore(ROOT / "config", self.user)

    def tearDown(self):
        self.tmp.cleanup()

    def test_repo_defaults(self):
        s = self.store.load()
        self.assertEqual(s["default_context"], 8192)
        self.assertNotIn("_opis", s)
        self.assertEqual(s["hf_token"], "")

    def test_user_changes_go_to_user_dir_only(self):
        before = (ROOT / "config" / "settings.json").read_text(encoding="utf-8")
        self.store.save_user({"hf_token": "hf_secret", "margin_ratio": 0.15})
        self.assertEqual(self.store.load()["margin_ratio"], 0.15)
        self.assertEqual((ROOT / "config" / "settings.json").read_text(encoding="utf-8"), before)
        self.assertEqual(json.loads((self.user / "settings.json").read_text())["hf_token"], "hf_secret")

    def test_broken_user_file_is_ignored(self):
        (self.user / "settings.json").write_text("{nie json")
        self.assertEqual(self.store.load()["default_context"], 8192)

    def test_ram_reserve_per_system(self):
        s = self.store.load()
        self.assertEqual(run_settings(s, "win32").ram_reserve, gib(4))
        self.assertEqual(run_settings(s, "linux").ram_reserve, gib(2))
        self.assertEqual(run_settings(s, "darwin").ram_reserve, gib(2))
        self.assertEqual(run_settings({**s, "kv_cache_type": "bzdura"}, "linux").kv_cache_type,
                         KvCacheType.F16)


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = [1000.0]
        self.cache = ModelCache(Path(self.tmp.name), max_age=60, clock=lambda: self.now[0])

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_and_preferred_quant_from_each_link(self):
        http = gguf_fake()
        src = HuggingFaceSource(http, self.cache)
        first = src.load(HfRef(REPO, quant="Q8_0"))
        calls = len(http.calls)
        second = src.load(HfRef(REPO, path="Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"))
        self.assertEqual(len(http.calls), calls)
        self.assertEqual(second.arch, first.arch)
        self.assertEqual(second.variants, first.variants)
        self.assertEqual((first.spec().quant, second.spec().quant), ("Q8_0", "IQ4_XS"))

    def test_expiry_and_damage(self):
        http = gguf_fake()
        src = HuggingFaceSource(http, self.cache)
        src.load(HfRef(REPO))
        self.now[0] += 61
        calls = len(http.calls)
        src.load(HfRef(REPO))
        self.assertGreater(len(http.calls), calls)                 # przeterminowane
        for f in Path(self.tmp.name).glob("*.json"):
            f.write_text("uszkodzone")
        self.assertIsNone(self.cache.get(f"{REPO}@main"))


if __name__ == "__main__":
    unittest.main()
