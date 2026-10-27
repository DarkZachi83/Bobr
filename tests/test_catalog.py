import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import main
from aibobr.cli.bar import PLAIN
from aibobr.cli.cli import Cli
from aibobr.i18n import set_language
from aibobr.knowledge import catalog
from aibobr.sources.base import SourceError
from aibobr.sources.huggingface import HuggingFaceSource
from tests.fake_http import FakeHttp
from tests.test_huggingface import REPO, gguf_fake

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_catalog  # noqa: E402

GIB = 1024**3


def entry(name, params_b, q4_gib, layers=32, kv=8, extra=None):
    return {"name": name, "repo": f"test/{name.replace(' ', '-')}-GGUF",
            "arch": {"params_total": int(params_b * 1e9), "n_layers": layers, "n_kv_heads": kv,
                     "head_dim": 128, "max_context": 32768, **(extra or {})},
            "variants": {"Q8_0": int(q4_gib * 1.7 * GIB), "Q4_K_M": int(q4_gib * GIB)}}


def catalog_data(updated="2026-10-27", models=None):
    return {"format": 1, "updated": updated, "source": "test",
            "models": models if models is not None else [
                entry("Small 3B", 3.2, 1.9, layers=28), entry("Mid 8B", 8.0, 4.6),
                entry("Big 14B", 14.8, 8.4, layers=48)]}


class CatalogFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_bundled_empty_catalog_is_valid(self):
        cat = catalog.load(ROOT / "data" / "models" / "catalog.json", None)
        self.assertEqual(len(cat), 0)
        self.assertEqual(cat.suggestion_specs(), [])

    def test_parse_and_suggestions_use_q4_k_m(self):
        cat = catalog.load(self.write("c.json", catalog_data()), None)
        self.assertEqual(len(cat), 3)
        specs = cat.suggestion_specs()
        self.assertEqual({s.quant for s in specs}, {"Q4_K_M"})
        self.assertEqual(specs[1].weights_bytes, int(4.6 * GIB))
        self.assertEqual(specs[1].origin, "https://huggingface.co/test/Mid-8B-GGUF")

    def test_newer_copy_wins_and_broken_entries_are_skipped(self):
        bundled = self.write("b.json", catalog_data("2026-10-01"))
        models = catalog_data()["models"] + [{"name": "broken"}, {"variants": {"Q4_K_M": 1}}]
        downloaded = self.write("d.json", catalog_data("2026-11-15", models))
        cat = catalog.load(bundled, downloaded)
        self.assertEqual((cat.updated, len(cat)), ("2026-11-15", 3))
        self.assertEqual(catalog.load(bundled, self.write("x.json", {"nie": "katalog"})).updated,
                         "2026-10-01")

    def test_update(self):
        current = catalog.load(self.write("b.json", catalog_data("2026-10-01")), None)
        target = self.dir / "user" / "catalog.json"
        http = FakeHttp(json={catalog.CATALOG_URL: catalog_data("2026-11-15")})
        fresh, newer = catalog.update(http, target, current)
        self.assertTrue(newer)
        self.assertEqual(fresh.updated, "2026-11-15")
        self.assertTrue(target.exists())
        _, again = catalog.update(http, target, fresh)               # już aktualny
        self.assertFalse(again)

    def test_damaged_download_keeps_current(self):
        current = catalog.load(self.write("b.json", catalog_data("2026-10-01")), None)
        target = self.dir / "user" / "catalog.json"
        with self.assertRaises(SourceError) as ctx:
            catalog.update(FakeHttp(json={catalog.CATALOG_URL: {"format": 99}}), target, current)
        self.assertEqual(ctx.exception.code, "catalog.bad")
        empty = FakeHttp(json={catalog.CATALOG_URL: catalog_data("2027-01-01", [])})
        self.assertFalse(catalog.update(empty, target, current)[1])  # pusty nie zastępuje
        self.assertFalse(target.exists())


class BuildCatalogTest(unittest.TestCase):
    def test_build_from_repositories(self):
        lines = []
        data, failed = build_catalog.build([REPO, "nobody/missing-GGUF"],
                                           HuggingFaceSource(gguf_fake()), "2026-10-27", lines.append)
        self.assertEqual(failed, [("nobody/missing-GGUF", "source.not_found")])
        [m] = data["models"]
        self.assertEqual(m["repo"], REPO)
        self.assertEqual(m["arch"]["params_total"], 8_030_261_248)
        self.assertEqual(m["variants"]["Q4_K_M"], 4_920_000_000)
        self.assertNotIn("n_experts", m["arch"])                     # puste pola pomijane
        cat = catalog.parse(json.loads(json.dumps(data)))           # wynik czyta się z powrotem
        self.assertEqual(cat.suggestion_specs()[0].quant, "Q4_K_M")

    def test_source_list_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.txt"
            p.write_text("# komentarz\n\na/b-GGUF   # opis\nc/d-GGUF\n", encoding="utf-8")
            self.assertEqual(build_catalog.read_sources(p), ["a/b-GGUF", "c/d-GGUF"])
        self.assertGreater(len(build_catalog.read_sources(build_catalog.SOURCES)), 10)


class CatalogInAppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        set_language("pl")

    def tearDown(self):
        self.tmp.cleanup()
        set_language("en")

    def test_cli_update_then_suggestion(self):
        from tests.gguf_writer import llama_like
        user = self.dir / "cfg"
        c = main.build_app(ROOT / "tests" / "fake_hardware" / "real_laptop_rtx5050_linux.json",
                           user_dir=user, cache_dir=self.dir / "cache")
        http = FakeHttp(json={catalog.CATALOG_URL: catalog_data()})
        c.update_catalog = (lambda orig: lambda client=None: orig(http))(c.update_catalog)

        big = self.dir / "Big-70B-Q4_K_M.gguf"
        with open(big, "wb") as f:
            f.write(llama_like(name="Llama 3.3 70B", layers=80, heads=64, embed=8192))
            f.truncate(42_520_000_000)
        out = io.StringIO()
        args = main.build_parser().parse_args(["--update-catalog", str(big), "--explain"])
        Cli(c, out=out, palette=PLAIN).run(args)
        text = out.getvalue()
        self.assertIn("Lista modeli zaktualizowana: 3 modeli", text)
        # Laptop 8 GB: Mid 8B (6,1 GB) mieści się na styk → podpowiedź to Small 3B.
        self.assertIn("model Small 3B w Q4_K_M zmieści się w całości.", text)
        self.assertTrue((user / "catalog.json").exists())
        again = main.build_app(None, user_dir=user, cache_dir=self.dir / "cache")
        self.assertEqual(len(again.catalog), 3)                      # pobrana kopia po restarcie


if __name__ == "__main__":
    unittest.main()
