"""Testy warstwy tłumaczeń (wzór z NIUCHACZA): te same klucze i pola w en/pl,
brak pustych tekstów i każdy klucz użyty w kodzie istnieje w słowniku."""
import json
import re
import string
import unittest
from pathlib import Path

from aibobr import i18n
from aibobr.i18n import LANGUAGES, fmt_context, fmt_gb, fmt_params, has_key

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "aibobr" / "i18n"
POLISH = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")
KEY_IN_CODE = re.compile(
    r"""["']((?:app|args|cli|detection|source|manual|warn|verdict|expl|note|catalog|gui|ollama|scan|image|audio)\.[a-z0-9_.]+)["']""")


def _dict(lang):
    return json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def _fields(text):
    return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}


class DictionariesTest(unittest.TestCase):
    def setUp(self):
        self.dicts = {lang: _dict(lang) for lang in LANGUAGES}
        self.base = self.dicts[i18n.DEFAULT_LANGUAGE]

    def test_same_keys(self):
        for lang, d in self.dicts.items():
            with self.subTest(lang=lang):
                self.assertEqual(set(), set(self.base) - set(d), "brakujące klucze")
                self.assertEqual(set(), set(d) - set(self.base), "nadmiarowe klucze")

    def test_same_format_fields(self):
        for lang, d in self.dicts.items():
            for key, text in self.base.items():
                with self.subTest(lang=lang, key=key):
                    self.assertEqual(_fields(text), _fields(d.get(key, "")))

    def test_no_empty_texts(self):
        for lang, d in self.dicts.items():
            for key, text in d.items():
                self.assertTrue(text.strip(), f"{lang}:{key}")

    def test_system_language_from_environment(self):
        sl = i18n.system_language
        self.assertEqual(sl({"LANG": "pl_PL.UTF-8"}), "pl")
        self.assertEqual(sl({"LC_ALL": "C.UTF-8", "LANG": "pl_PL.UTF-8"}), "pl")   # „C” pomijamy
        self.assertEqual(sl({"LC_MESSAGES": "en_US.UTF-8", "LANG": "pl_PL.UTF-8"}), "en")

    def test_every_language_has_a_name(self):
        for lang in LANGUAGES:
            self.assertIn(f"lang.{lang}", self.base)

    def test_english_has_no_polish_letters(self):
        names = {"gui.about_text"}                   # imię i nazwisko autora zostają z polskimi literami
        bad = [k for k, v in self.base.items() if POLISH.search(v) and k not in names]
        self.assertEqual([], bad)

    def test_every_key_used_in_code_exists(self):
        missing = set()
        for path in (ROOT / "aibobr").rglob("*.py"):
            for key in KEY_IN_CODE.findall(path.read_text(encoding="utf-8")):
                if not key.endswith(".") and not has_key(key):
                    missing.add(f"{path.name}: {key}")
        self.assertEqual(set(), missing)


class FormattingTest(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(fmt_gb(int(6.1 * 1024**3), "pl"), "6,1 GB")
        self.assertEqual(fmt_gb(int(6.1 * 1024**3), "en"), "6.1 GB")
        self.assertEqual(fmt_gb(None), "?")
        self.assertEqual(fmt_context(8192), "8k")
        self.assertEqual(fmt_context(12000), "12000")
        self.assertEqual(fmt_params(3_350_000_000, "pl"), "3,4B")
        self.assertEqual(fmt_params(500_000_000, "en"), "500M")


if __name__ == "__main__":
    unittest.main()
