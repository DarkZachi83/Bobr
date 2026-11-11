import json
import tempfile
import unittest
from pathlib import Path

from aibobr.app.bench_db import BenchDb
from aibobr.quality import custom, suites
from aibobr.quality.checks import FORMAT, NOT_RUN, OK, WRONG
from aibobr.quality.custom import CustomSuiteError, CustomSuites, Problem, problems
from aibobr.quality.sandbox import Status


def sample(**info) -> dict:
    data = custom.new_suite("moj_test", "Mój test", ["pl", "en"])
    data["suite_info"].update(info)
    data["tasks"] = [
        {"id": "T1", "category": "liczby", "level": "easy", "max_tokens": 256,
         "prompt": {"pl": "Ile to 2+2? Podaj tylko liczbę.", "en": "What is 2+2? Number only."},
         "check": {"type": "number", "expected": 4}},
        {"id": "T2", "category": "wybór", "level": "medium", "max_tokens": 256,
         "prompt": {"pl": "Czy 7 jest pierwsza? tak/nie", "en": "Is 7 prime? yes/no"},
         "check": {"type": "choice", "expected": {"pl": "tak", "en": "yes"},
                   "options": {"pl": ["tak", "nie"], "en": ["yes", "no"]}}},
        {"id": "T3", "category": "kod", "level": "hard", "max_tokens": 512,
         "prompt": {"pl": "Napisz add(a, b).", "en": "Write add(a, b)."},
         "check": {"type": "python", "function": "add",
                   "tests": [{"args": [1, 2], "expected": 3}, {"args": [0, 0], "expected": 0}]}},
    ]
    return data


def codes(data) -> set[tuple]:
    return {(p.task, p.field, p.code) for p in problems(data)}


class Validation(unittest.TestCase):
    def test_sample_is_valid(self):
        self.assertEqual(problems(sample()), [])

    def test_every_check_type_valid_example(self):
        data = sample()
        extra = [{"type": "text", "expected": ["kot"], "sentence": ["Ala ma kota."]},
                 {"type": "fraction", "expected": "3/4"}, {"type": "json", "expected": {"a": [1, None]}},
                 {"type": "manual"}, {"type": "number", "expected": 0.5, "tolerance": 0.01}]
        for n, spec in enumerate(extra, 10):
            task = custom.new_task(data)
            task.update(category="x", prompt={"pl": "p", "en": "e"}, check=spec)
            data["tasks"].append(task)
        self.assertEqual(problems(data), [])

    def test_suite_info_errors(self):
        self.assertIn((None, "suite_info.id", "bad_id"), codes(sample(id="Mój test!")))
        self.assertIn((None, "suite_info.id", "standard_id"), codes(sample(id="scisly_v1")))
        self.assertIn((None, "suite_info.kind", "kind"), codes(sample(kind="science")))
        for bad in (["en", "pl"], ["pl", "de"], ["PL"], [], ["pl", "en", "de"], ["polski"], "pl"):
            self.assertIn((None, "suite_info.languages", "languages"), codes(sample(languages=bad)), bad)
        self.assertIn((None, "suite_info.name", "name"), codes(sample(name={"pl": "x"})))
        self.assertEqual(codes([1]), {(None, "file", "not_object")})

    def test_any_main_language_with_optional_english(self):
        for langs in (["lt"], ["ko", "en"], ["en"], ["zh", "en"], ["fil"]):
            data = sample(languages=langs, name={l: "Test" for l in langs})
            for task in data["tasks"]:
                task["prompt"] = {l: "p" for l in langs}
                for key in ("expected", "options"):
                    if isinstance(task["check"].get(key), dict):
                        task["check"][key] = {l: task["check"][key]["en"] for l in langs}
            self.assertEqual(problems(data), [], langs)

    def test_task_errors_point_at_task_and_field(self):
        data = sample()
        data["tasks"][0].update(level="trudne", max_tokens=5, category=" ")
        data["tasks"][0]["prompt"]["en"] = ""
        data["tasks"][1]["id"] = "T1"
        found = codes(data)
        for item in [("T1", "level", "level"), ("T1", "max_tokens", "max_tokens"), ("T1", "category", "missing"),
                     ("T1", "prompt.en", "prompt"), ("T1", "id", "duplicate_id")]:
            self.assertIn(item, found)

    def test_check_errors(self):
        cases = [
            ({"type": "regex", "expected": "x"}, "check.type", "check_type"),
            ({"type": "number", "expected": "4"}, "check.expected", "number"),
            ({"type": "number", "expected": True}, "check.expected", "number"),
            ({"type": "number", "expected": 4, "tolerance": -1}, "check.tolerance", "tolerance"),
            ({"type": "fraction", "expected": "3/0"}, "check.expected", "fraction"),
            ({"type": "fraction", "expected": "0.75"}, "check.expected", "fraction"),
            ({"type": "text", "expected": []}, "check.expected", "text_expected"),
            ({"type": "text", "expected": {"pl": ["a"]}}, "check.expected", "text_expected"),
            ({"type": "choice", "expected": "c", "options": ["a", "b"]}, "check.expected", "choice_expected"),
            ({"type": "choice", "expected": "a", "options": ["a"]}, "check.options", "choice_options"),
            ({"type": "python", "function": "class", "tests": [{"args": [], "expected": 1}]},
             "check.function", "python_function"),
            ({"type": "python", "function": "f", "tests": []}, "check.tests", "python_tests"),
            ({"type": "python", "function": "f", "tests": [{"args": "1", "expected": 1}]},
             "check.tests", "python_tests"),
            ({"type": "number"}, "check.expected", "missing"),
        ]
        for spec, field, code in cases:
            data = sample()
            data["tasks"][0]["check"] = spec
            self.assertIn(("T1", field, code), codes(data), spec)

    def test_no_code_in_test_file(self):
        """Plik testu nie ma miejsca na kod: każde dodatkowe pole w sprawdzeniu jest błędem."""
        data = sample()
        data["tasks"][2]["check"]["setup"] = "import os; os.system('rm -rf ~')"
        self.assertIn(("T3", "check.setup", "unknown_field"), codes(data))
        data = sample()
        data["tasks"][2]["check"]["tests"][0]["code"] = "x"
        self.assertIn(("T3", "check.tests", "python_tests"), codes(data))

    def test_limits(self):
        data = sample()
        data["tasks"] = []
        self.assertIn((None, "tasks", "no_tasks"), codes(data))
        data = sample()
        data["tasks"] = [dict(data["tasks"][0], id=f"T{n}") for n in range(custom.MAX_TASKS + 1)]
        self.assertIn((None, "tasks", "too_many_tasks"), codes(data))

    def test_copies_of_standard_suites_are_valid(self):
        for sid in suites.STANDARD:
            data = custom.copy_of_standard(sid, "kopia", "Kopia")
            self.assertEqual(problems(data), [], sid)
            self.assertEqual(data["suite_info"]["kind"], "custom")
            self.assertTrue(data["suite_info"]["based_on"].startswith(sid))
        self.assertEqual(suites.load("scisly_v1")["suite_info"]["kind"], "science")  # oryginał nietknięty

    def test_level_counts(self):
        self.assertEqual(custom.level_counts(sample()), {"easy": 1, "medium": 1, "hard": 1})


class Helpers(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(custom.slug("Mój Żółty test!"), "moj_zolty_test")
        self.assertEqual(custom.slug("!!!"), "test")
        self.assertEqual(custom.unique_id("Mój test", ["moj_test", "moj_test_2"]), "moj_test_3")
        self.assertEqual(custom.unique_id("scisly v1", []), "scisly_v1_2")
        data = sample()
        self.assertEqual(custom.next_task_id(data), "T4")
        task = custom.new_task(data, "choice")
        self.assertEqual((task["id"], task["check"]["type"], set(task["prompt"])), ("T4", "choice", {"pl", "en"}))

    def test_bump(self):
        self.assertEqual(custom.bump("1"), "2")
        self.assertEqual(custom.bump("1.1"), "1.2")
        self.assertEqual(custom.bump("beta"), "beta.1")


class Storage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / "quality"
        self.used = set()
        self.store = CustomSuites(self.base, used=lambda s, v: (s, v) in self.used)

    def test_save_load_and_listing(self):
        self.store.save(sample())
        self.assertEqual(self.store.load("moj_test"), sample())
        (self.base / "zepsuty.json").write_text("{nie json", encoding="utf-8")
        entries = {e.suite_id: e for e in self.store.entries()}
        self.assertIsNone(entries["moj_test"].error)
        self.assertEqual(entries["zepsuty"].error.problems, [Problem(None, "file", "not_json")])
        self.assertEqual([d["suite_info"]["id"] for d in self.store.valid()], ["moj_test"])
        self.assertTrue(suites.load("moj_test", self.base))  # ten sam format co standardowe

    def test_invalid_save_writes_nothing(self):
        data = sample()
        data["tasks"][0]["level"] = "x"
        with self.assertRaises(CustomSuiteError) as e:
            self.store.save(data)
        self.assertEqual(e.exception.problems, [Problem("T1", "level", "level")])
        self.assertFalse(self.base.exists() and any(self.base.iterdir()))

    def test_path_traversal_rejected(self):
        with self.assertRaises(CustomSuiteError):
            self.store.path("../../etc/passwd")

    def test_version_bumps_only_when_used(self):
        self.store.save(sample())
        changed = sample()
        changed["tasks"][0]["check"]["expected"] = 5
        self.assertEqual(self.store.save(changed)["suite_info"]["version"], "1")
        self.used.add(("moj_test", "1"))
        self.assertEqual(self.store.save(changed)["suite_info"]["version"], "1")  # bez zmian – bez nowej wersji
        changed["tasks"][0]["check"]["expected"] = 6
        self.assertEqual(self.store.save(changed)["suite_info"]["version"], "2")
        self.assertEqual(self.store.load("moj_test")["suite_info"]["version"], "2")

    def test_delete(self):
        self.store.save(sample())
        self.store.delete("moj_test")
        self.assertEqual(self.store.entries(), [])

    def test_export_and_import(self):
        self.store.save(sample())
        out = self.store.export("moj_test", Path(self.tmp.name) / "wyjscie")
        self.assertEqual(out.suffix, ".json")
        data, conflict = self.store.read_import(out)
        self.assertTrue(conflict)
        copy_ = self.store.import_as_copy(data)
        self.assertEqual((copy_["suite_info"]["id"], copy_["suite_info"]["version"]), ("moj_test_2", "1"))
        self.assertEqual(sorted(self.store.ids()), ["moj_test", "moj_test_2"])

    def test_import_replace_never_reuses_used_version(self):
        self.store.save(sample())
        self.used.add(("moj_test", "1"))
        foreign = sample(version="1")
        foreign["tasks"][0]["check"]["expected"] = 9
        self.assertEqual(self.store.import_replace(foreign)["suite_info"]["version"], "2")

    def test_import_rejects_bad_files(self):
        bad = Path(self.tmp.name) / "zly.json"
        bad.write_text(json.dumps({"suite_info": {"id": "x"}}), encoding="utf-8")
        with self.assertRaises(CustomSuiteError):
            self.store.read_import(bad)
        big = Path(self.tmp.name) / "duzy.json"
        big.write_bytes(b" " * (custom.MAX_FILE_BYTES + 1))
        with self.assertRaises(CustomSuiteError) as e:
            self.store.read_import(big)
        self.assertEqual(e.exception.problems[0].code, "too_large")

    def test_db_knows_used_versions(self):
        db = BenchDb(Path(self.tmp.name) / "b.sqlite")
        self.assertFalse(db.quality_suite_used("moj_test", "1"))
        qs = db.start_quality_session(started_at="2026-11-11T10:00:00+01:00", attempts=1)
        db.start_quality_run(qs, model="m", suite="moj_test", suite_version="1", lang="pl", started_at="x")
        self.assertTrue(db.quality_suite_used("moj_test", "1"))
        self.assertFalse(db.quality_suite_used("moj_test", "2"))


class FakeSandbox:
    def __init__(self, ready):
        self.ready, self.calls = ready, []

    def status(self):
        return Status(self.ready, "ok" if self.ready else "unsupported")

    def run_tests(self, code, function, tests):
        self.calls.append(function)
        return True, ""


class Preview(unittest.TestCase):
    def test_same_verdicts_as_in_test(self):
        spec = {"type": "number", "expected": 204}
        self.assertEqual(custom.preview(spec, "204").reason, OK)
        self.assertEqual(custom.preview(spec, "240 × 0,85 = 204").reason, FORMAT)
        self.assertEqual(custom.preview(spec, "200").reason, WRONG)
        choice = sample()["tasks"][1]["check"]
        self.assertTrue(custom.preview(choice, "yes", "en").correct)

    def test_python_only_in_ready_sandbox(self):
        spec = sample()["tasks"][2]["check"]
        answer = "def add(a, b):\n    return a + b\n"
        self.assertEqual(custom.preview(spec, answer).reason, NOT_RUN)
        self.assertEqual(custom.preview(spec, answer, sandbox=FakeSandbox(False)).reason, NOT_RUN)
        box = FakeSandbox(True)
        self.assertTrue(custom.preview(spec, answer, sandbox=box).correct)
        self.assertEqual(box.calls, ["add"])


if __name__ == "__main__":
    unittest.main()
