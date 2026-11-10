"""Testy jakości 0.10: pliki zadań (45 zadań, poziomy, kategorie) i sprawdzanie odpowiedzi
(correct + format_compliant), w tym prawdziwe odpowiedzi Minitrona z pomiarów autora."""
import unittest
from collections import Counter

from aibobr.quality import checks as c
from aibobr.quality import suites


class SuitesTest(unittest.TestCase):
    def setUp(self):
        self.h, self.s = suites.standard()

    def test_shape_agreed_with_author(self):
        for data in (self.h, self.s):
            self.assertEqual(len(data["tasks"]), 15)
            self.assertEqual(Counter(t["level"] for t in data["tasks"]), {"easy": 5, "medium": 5, "hard": 5})
        self.assertEqual(Counter(t["category"] for t in self.h["tasks"]),
                         {"ortografia": 3, "fleksja": 4, "liczebniki": 2, "aspekt": 1, "interpunkcja": 2,
                          "przeksztalcanie": 2, "wiedza": 1})
        self.assertEqual(Counter(t["category"] for t in self.s["tasks"]),
                         {"arytmetyka": 3, "procenty": 3, "logika": 3, "json": 3, "python": 3})
        self.assertEqual(list(suites.variants(self.h)), [("humanistyczny_v1", "pl")])
        self.assertEqual(list(suites.variants(self.s)), [("scisly_v1", "pl"), ("scisly_v1", "en")])

    def test_every_expected_answer_passes_in_every_language(self):
        """Wzorcowa odpowiedź musi przejść własne sprawdzenie – inaczej zadanie jest zepsute."""
        for data in (self.h, self.s):
            for task in data["tasks"]:
                spec = task["check"]
                for lang in data["suite_info"]["languages"]:
                    if spec["type"] == "python":
                        continue                                   # kod – w testach piaskownicy
                    answer = c.expected_text(spec, lang).split(" | ")[0]
                    r = c.check(answer, spec, lang)
                    self.assertEqual((r.correct, r.format_compliant, r.reason), (True, True, "ok"),
                                     (task["id"], lang, answer, r))

    def test_validation_messages(self):
        bad = {"suite_info": {"id": "x", "languages": ["pl", "en"]},
               "tasks": [{"id": "A", "level": "easy", "prompt": {"pl": "?"}, "check": {"type": "text"}}]}
        with self.assertRaisesRegex(suites.SuiteError, "brak treści w języku \\['en'\\]"):
            suites.validate(bad)
        bad["tasks"][0]["prompt"]["en"] = "?"
        bad["tasks"][0]["check"]["type"] = "judge"
        with self.assertRaisesRegex(suites.SuiteError, "nieznany rodzaj"):
            suites.validate(bad)


class TextTest(unittest.TestCase):
    def test_normalization_keeps_what_is_graded(self):
        exp = ["Wiem, że przyjdziesz, chociaż pada deszcz."]
        for answer in ("Wiem, że przyjdziesz, chociaż pada deszcz.", "  „Wiem, że przyjdziesz, chociaż pada deszcz”  ",
                       "**Wiem, że przyjdziesz, chociaż pada deszcz.**", "wiem, że przyjdziesz, chociaż pada deszcz"):
            self.assertEqual(c.check_text(answer, exp).reason, "ok", answer)
        self.assertEqual(c.check_text("Wiem, że przyjdziesz chociaż pada deszcz.", exp).reason, "wrong")  # przecinek
        self.assertEqual(c.check_text("Zolw wszedl do zrodla.", ["Żółw wszedł do źródła."]).reason, "wrong")

    def test_minitron_answers(self):
        self.assertEqual(c.check_text("Tomaszu.", ["Tomaszu"]).reason, "ok")         # kropka – normalizacja
        self.assertEqual(c.check_text("jabłka.", ["jabłka"]).reason, "ok")
        r = c.check_text("Odpowiedź:\nkoni", ["koni"])                               # osobna linia + komentarz
        self.assertEqual((r.correct, r.format_compliant, r.reason), (True, False, "format"))
        r = c.check_text("Wołacz od Grzegorz to Grzegorzu.", ["Grzegorzu"])          # zdanie zamiast słowa
        self.assertEqual((r.correct, r.reason), (False, "wrong"))
        self.assertEqual(c.check_text("", ["x"]).reason, "empty")
        self.assertEqual(c.check_text("<think>hmm</think>\nkoni", ["koni"]).reason, "ok")
        self.assertEqual(c.check_text("Mysz zjedzona została przez kota.",
                                      ["Mysz została zjedzona przez kota.", "Mysz zjedzona została przez kota."]).reason,
                         "ok")


class NumberTest(unittest.TestCase):
    def test_number_forms(self):
        for answer, reason in (("391", "ok"), ("391.0", "ok"), ("391,0", "ok"), ("**391**", "ok"), ("2 500", "ok"),
                               ("Wynik: 391", "format"), ("4%", "format"), ("2500 m", "format")):
            exp = 2500 if "2" in answer[:1] else 4 if "%" in answer else 391
            r = c.check_number(answer, exp)
            self.assertEqual((r.correct, r.reason), (True, reason), answer)
        self.assertEqual(c.check_number("0,185", 0.185).reason, "ok")
        self.assertEqual(c.check_number("17 × 23 = 391", 391).reason, "format")    # po ostatnim „=” jedna liczba
        self.assertEqual(c.check_number("Może 391, może 392", 391).reason, "ambiguous")  # nie zgadujemy
        self.assertEqual(c.check_number("390", 391).reason, "wrong")
        self.assertEqual(c.check_number("nie wiem", 391).reason, "wrong")

    def test_final_answer_patterns_from_author_run(self):
        """Prawdziwe odpowiedzi Minitrona (10.11.2026) – wynik końcowy tylko z dwóch wzorców (propozycja B)."""
        r = c.check_number("240 * (1 - 15 / 100) = 240 * 0.85 = 204", 204)
        self.assertEqual((r.correct, r.format_compliant, r.value), (True, False, 204.0))
        worked = "Aby obliczyć 17 × 23, rozbijmy:\n17 × 20 = 340\n17 × 3 = 51\n340 + 51 = 391\n\nWynik: 391"
        self.assertEqual(c.check_number(worked, 391).reason, "format")                 # ostatnia linia – jedna liczba
        latex = "First:\n\\[ 0.15 \\times 0.4 = 0.06 \\]\nThen 0.06 + 0.125 = 0.185\n\\]"
        self.assertEqual(c.check_number(latex, 0.185).reason, "format")                # linia „\\]” pominięta
        self.assertEqual(c.check_number("17 × 23 = 391\nAle może też 392 albo 393.", 391).reason, "ambiguous")
        self.assertEqual(c.check_number("Krok 1: 17 × 20 = 340\nKrok 2: 17 × 3 = 51", 391).reason, "wrong")

    def test_minitron_latex_and_worked_choice(self):
        """Prawdziwe odpowiedzi z drugiego pomiaru autora (propozycje D i E)."""
        s2 = ("To add fractions, we need to find a common denominator.\n"
              "\\[ \\frac{3}{4} = \\frac{9}{12} \\]\n\\[ \\frac{5}{6} = \\frac{10}{12} \\]\n"
              "\\[ \\frac{9}{12} + \\frac{10}{12} = \\frac{19}{12} \\]\n"
              "The result is already in its simplest form, so the answer is:\n\\[ \\boxed{\\frac{19}{12}} \\]")
        r = c.check_fraction(s2, "19/12")
        self.assertEqual((r.correct, r.format_compliant, r.value), (True, False, "19/12"))
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        s9 = ("1. The day before yesterday was Tuesday.\n3. Today would be Thursday.\n"
              "4. The day after tomorrow would be Friday.\n\nSo, the answer is:\n\\[ \\boxed{Friday} \\]")
        r = c.check_choice(s9, "saturday", days)
        self.assertEqual((r.correct, r.reason, r.value), (False, "wrong", "friday"))      # uczciwie błędne
        self.assertEqual(c.check_choice(s9.replace("Friday} ", "Saturday} "), "saturday", days).reason, "format")
        self.assertEqual(c.check_choice("Tuesday or Saturday?", "saturday", days).reason, "ambiguous")

    def test_fill_in_whole_sentence(self):
        spec = {"type": "text", "expected": ["koni"], "sentence": ["Na łące pasło się sześć koni."]}
        r = c.check("Na łące pasło się sześć koni.", spec)
        self.assertEqual((r.correct, r.format_compliant, r.reason), (True, False, "format"))
        self.assertEqual(c.check("Na łące pasło się sześć koniów.", spec).reason, "wrong")
        self.assertEqual(c.check("koni", spec).reason, "ok")

    def test_fraction_must_be_irreducible(self):
        self.assertEqual(c.check_fraction("19/12", "19/12").reason, "ok")
        self.assertEqual(c.check_fraction("Wynik: 19 / 12", "19/12").reason, "format")
        self.assertEqual(c.check_fraction("38/24", "19/12").reason, "wrong")
        self.assertEqual(c.check_fraction("3/4 + 5/6 = 19/12", "19/12").reason, "format")   # wynik po „=”
        self.assertEqual(c.check_fraction("3/4 albo 5/6, nie wiem", "19/12").reason, "ambiguous")

    def test_choice(self):
        r = c.check_choice("Tak, Mruczek jest ssakiem.", "tak", ["tak", "nie"])
        self.assertEqual((r.correct, r.format_compliant), (True, False))
        self.assertEqual(c.check_choice("Saturday", "saturday", ["monday", "saturday"]).reason, "ok")
        days = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]
        self.assertEqual(c.check_choice("Dziś czwartek, więc pojutrze sobota.", "sobota", days).reason, "ambiguous")
        self.assertEqual(c.check_choice("Piątek", "sobota", days).reason, "wrong")


class JsonTest(unittest.TestCase):
    EXP = {"name": "Ola", "age": 28, "city": "Gdynia"}

    def test_table_agreed_with_author(self):
        rows = [
            ('{"name": "Ola", "age": 28, "city": "Gdynia"}', True, True, "ok"),
            ('```json\n{"name": "Ola", "age": 28, "city": "Gdynia"}\n```', True, False, "format"),
            ('Oto dane:\n{"name": "Ola", "age": 28, "city": "Gdynia"}\nGotowe.', True, False, "format"),
            ('{"name": "Ola", "age": "28", "city": "Gdynia"}', False, None, "wrong"),      # wiek jako tekst
            ('{"name": "Ola", "age": 28, "city": "Gdynia"', False, None, "syntax"),
            ('{"name": "Ola"} albo {"name": "Ola", "age": 28, "city": "Gdynia"}', False, None, "ambiguous"),
        ]
        for answer, correct, fmt, reason in rows:
            r = c.check_json(answer, self.EXP)
            self.assertEqual((r.correct, r.format_compliant, r.reason), (correct, fmt, reason), answer)

    def test_escaping_and_types(self):
        self.assertEqual(c.check_json('{"path": "C:\\\\Users\\\\Ola"}', {"path": "C:\\Users\\Ola"}).reason, "ok")
        self.assertEqual(c.check_json('{"path": "C:\\Users\\Ola"}', {"path": "C:\\Users\\Ola"}).reason, "syntax")
        self.assertFalse(c.same(True, 1))
        self.assertTrue(c.same(28, 28.0))
        self.assertFalse(c.same([2, 4], [4, 2]))


class PythonTest(unittest.TestCase):
    SPEC = {"type": "python", "function": "f", "tests": [{"args": [1], "expected": 2}]}

    def test_without_sandbox_not_evaluated(self):
        r = c.check("def f(x):\n    return x + 1\n", self.SPEC)
        self.assertEqual((r.correct, r.format_compliant, r.reason), (None, True, "not_run"))

    def test_minitron_code_prefix_and_fences(self):
        r = c.check("Code:\n\ndef f(x):\n    return x + 1", self.SPEC)
        self.assertEqual((r.correct, r.format_compliant, r.value.startswith("def f")), (None, False, True))
        r = c.check("```python\ndef f(x):\n    return x + 1\n```", self.SPEC)
        self.assertEqual(r.format_compliant, False)
        self.assertEqual(c.check("def f(x) return", self.SPEC).reason, "syntax")
        self.assertEqual(c.check("```\na\n```\n```\nb\n```", self.SPEC).reason, "ambiguous")

    def test_runner_decides(self):
        calls = []

        def run(code, fn, tests):
            calls.append((fn, tests))
            return "x + 1" in code, "f(1) == 3"
        self.assertEqual(c.check("def f(x):\n    return x + 1", self.SPEC, run_python=run).reason, "ok")
        r = c.check("def f(x):\n    return x + 2", self.SPEC, run_python=run)
        self.assertEqual((r.correct, r.reason, r.value), (False, "wrong", "f(1) == 3"))
        self.assertEqual(calls[0], ("f", [{"args": [1], "expected": 2}]))
        self.assertEqual(c.check("def f(x):\n    return 1", self.SPEC, run_python=lambda *a: (None, "")).reason,
                         "not_run")                                    # piaskownica odmówiła


if __name__ == "__main__":
    unittest.main()
