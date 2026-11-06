"""Eksport CSV: wydajność, każda próba jakości osobno i podsumowanie (aibobr.bench.export)."""
import csv
import tempfile
import unittest
from pathlib import Path

from aibobr.bench import export
from aibobr.bench.quality import Attempt, QualityResult
from aibobr.bench.runner import PerfResult
from aibobr.i18n import set_language


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter=";"))


def attempt(n, text, passed, cached=0, valid=True, released=True):
    return Attempt(n, text, passed, "" if passed else "quality.reason.pattern", 64, cached, 2.4, released, valid)


HENRYK = "Daj, abym ja pomielił, a ty odpoczywaj"
CONTROLLED = QualityResult(
    "minitron", "pl_jezyk_v1", "pl_henrykowska", "Księga henrykowska", "knowledge", True, "", HENRYK, "Wyjaśnij…",
    runs=3, identical=3, passes=3, prompt_tokens=64, cached=0, mode="controlled", invalid=0,
    attempts=(attempt(1, HENRYK, True), attempt(2, HENRYK, True), attempt(3, HENRYK, True)))
# Standardowy z 3 próbami bez zwalniania (jak w 0.8): pierwsza od zera, dwie z pamięci promptu.
CACHED = QualityResult(
    "minitron", "pl_jezyk_v1", "pl_henrykowska", "Księga henrykowska", "knowledge", False,
    "quality.reason.missing:odpocz|poczyw", HENRYK, "Wyjaśnij…", runs=3, identical=1, passes=1, prompt_tokens=64,
    cached=63, cache_warn=True, mode="standard",
    attempts=(attempt(1, HENRYK, True), attempt(2, "Inaczej", False, 63), attempt(3, "Inaczej", False, 63)))
NO_FIELD = QualityResult(
    "gemma", "pl_jezyk_v1", "pl_vocative", "Wołacz", "inflection", False, "quality.reason.pattern", "Tomku", "Podaj…",
    runs=1, identical=1, passes=0, mode="controlled", invalid=1,
    attempts=(attempt(1, "Tomku", False, None, valid=False, released=False),))


class ExportTest(unittest.TestCase):
    def setUp(self):
        set_language("pl")
        self.dir = Path(tempfile.mkdtemp())

    def test_three_files_only_with_data(self):
        perf = [PerfResult("minitron", 1024, pp_tps=1914, ttft_ms=344, tg_tps=57.7, gpu_share=1.0, predicted="green")]
        files = export.export_all(self.dir / "bobr_x.csv", perf, [CONTROLLED])
        self.assertEqual([p.name for p in files],
                         ["bobr_x_wydajnosc.csv", "bobr_x_jakosc_proby.csv", "bobr_x_jakosc_podsumowanie.csv"])
        self.assertEqual(read(files[0])[1][:2], ["minitron", "1024"])
        only_q = export.export_all(self.dir / "q.csv", [], [CONTROLLED])          # zgłoszenie autora: sama jakość
        self.assertEqual([p.name for p in only_q], ["q_jakosc_proby.csv", "q_jakosc_podsumowanie.csv"])
        self.assertEqual(export.export_all(self.dir / "n.csv", [], []), [])

    def test_every_attempt_with_full_answer(self):
        rows = read(export.export_all(self.dir / "a.csv", [], [CONTROLLED, CACHED])[0])
        head = rows[0]
        self.assertEqual(len(rows), 1 + 6)
        col = {name: head.index(name) for name in head}
        self.assertEqual([r[col["Próba"]] for r in rows[1:4]], ["1/3", "2/3", "3/3"])
        self.assertEqual(rows[1][col["Odpowiedź"]], HENRYK)
        self.assertEqual(rows[1][col["Tryb"]], "Kontrolowany")
        self.assertEqual(rows[5][col["Tokeny promptu z pamięci"]], "63")
        self.assertEqual(rows[5][col["Zaliczona"]], "nie")

    def test_summary_separates_first_attempt_from_verdict(self):
        rows = read(export.export_all(self.dir / "s.csv", [], [CONTROLLED, CACHED, NO_FIELD])[1])
        col = {name: rows[0].index(name) for name in rows[0]}
        ok, cached, missing = rows[1], rows[2], rows[3]
        self.assertEqual((ok[col["Zaliczone"]], ok[col["Werdykt większościowy"]], ok[col["Zgodne z próbą 1"]]),
                         ("3/3", "zaliczone", "3/3"))
        # Minitron z pamięcią promptu: próba 1 zaliczona, werdykt – niezaliczone; najczęstsza 2/3.
        self.assertEqual((cached[col["Próba 1 zaliczona"]], cached[col["Werdykt większościowy"]],
                          cached[col["Zgodne z próbą 1"]], cached[col["Najczęstsza odpowiedź"]],
                          cached[col["Różnych odpowiedzi"]], cached[col["Ostrzeżenie (pamięć promptu)"]]),
                         ("tak", "niezaliczone", "1/3", "2/3", "2", "⚠"))
        self.assertEqual((missing[col["Pamięć promptu min"]], missing[col["Prób pominiętych"]]), ("null", "1"))


if __name__ == "__main__":
    unittest.main()
