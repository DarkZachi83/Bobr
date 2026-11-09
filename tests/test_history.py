"""Historia 0.9: wiersze (model w sesji), werdykt według obecnych progów, filtr, archiwum."""
import csv
import tempfile
import unittest
from pathlib import Path

from aibobr.app.bench_db import BenchDb
from aibobr.bench import export, history, speed
from aibobr.i18n import set_language

TH = speed.DEFAULT_THRESHOLDS


def add_run(db, sid, model, n, gen, ttft=56.0, status="finished", warnings=()):
    verdict = speed.verdict(gen, ttft / 1000, TH)[0] if status == "finished" else None
    rid = db.start_run(sid, model=model, run_no=n, started_at="2026-11-09T14:00:00+01:00", status=status,
                       gen_tps=gen, gen_tokens_total=578, gen_ms_total=578 / gen * 1000, ttft_wall_median_ms=ttft, verdict=verdict, thresholds_json=TH,
                       warnings_json=list(warnings))
    db.add_task(rid, task_no=1, task_id="t", gen_tokens=100, eval_ms=1000.0, response_text="odp")
    db.add_long_context(rid, num_ctx=1024, prompt_tokens=770, pp_tps=1900.0, tg_tps=55.0, ttft_wall_ms=400.0,
                        valid=1)
    return rid


class HistoryTest(unittest.TestCase):
    def setUp(self):
        set_language("pl")
        self.db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        old = self.db.start_session(started_at="2026-11-08T10:00:00+01:00", runs_per_model=1, status="finished")
        add_run(self.db, old, "gemma", 1, 31.0)
        self.sid = self.db.start_session(started_at="2026-11-09T14:00:00+01:00", runs_per_model=3,
                                         models_json=["minitron", "gemma"], status="finished")
        for n, gen in enumerate((61.2, 59.8, 61.5), 1):
            add_run(self.db, self.sid, "minitron", n, gen, warnings=("limited_basis",) if n == 2 else ())
        add_run(self.db, self.sid, "gemma", 1, 12.0, status="unreliable")

    def test_rows_newest_first_with_spread(self):
        rows = history.rows(self.db)
        self.assertEqual([(r["model"], r["started_at"][:10]) for r in rows],
                         [("minitron", "2026-11-09"), ("gemma", "2026-11-09"), ("gemma", "2026-11-08")])
        m = rows[0]
        self.assertEqual((m["runs"], m["spread"], m["verdicts"]), (3, (59.8, 61.5), ["fast"] * 3))
        self.assertEqual(m["warnings"], ["limited_basis"])
        self.assertEqual(rows[1]["verdicts"], ["unreliable"])
        self.assertIsNone(rows[1]["spread"])                       # niewiarygodny – bez zakresu

    def test_filter_and_models(self):
        self.assertEqual(history.models(self.db), ["gemma", "minitron"])
        rows = history.rows(self.db, model="gemma")
        self.assertEqual([r["started_at"][:10] for r in rows], ["2026-11-09", "2026-11-08"])
        self.assertTrue(all(r["model"] == "gemma" for r in rows))

    def test_current_thresholds_change_only_some(self):
        stricter = {"gen_tps": [40.0, 20.0, 8.0], "ttft_s": [1.0, 3.0, 8.0]}
        rows = history.rows(self.db, current=stricter)
        m, unreliable, old = rows
        self.assertEqual((m["verdicts_now"], m["changed"]), (["fast"] * 3, False))
        self.assertEqual((old["verdicts"], old["verdicts_now"], old["changed"]), (["fast"], ["useful"], True))
        self.assertEqual(unreliable["verdicts_now"], ["unreliable"])          # bez przeliczania

    def test_run_details_and_archive_export(self):
        rid = history.rows(self.db)[0]["run_ids"][0]
        run = history.run_details(self.db, rid)
        self.assertEqual((run["model"], len(run["tasks"]), run["long_context"][0]["num_ctx"]), ("minitron", 1, 1024))
        self.assertIsNone(history.run_details(self.db, "brak"))
        from aibobr.bench.runner import PerfResult
        self.db.add_perf(PerfResult("old", 8192, pp_tps=1000.5, tg_tps=41.25), measured_at="2026-11-04 10:00",
                         runtime="ollama", runtime_version="0.40.14", hardware="hw", power="ac", suite="perf_v1")
        folder = export.export_archive(self.db, tempfile.mkdtemp(), "2026-11-09")
        self.assertEqual(folder.name, "archiwum_przed_0.9_2026-11-09")
        with open(folder / "perf.csv", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        self.assertEqual((rows[0]["model"], rows[0]["tg_tps"], rows[0]["power_draw_w"]), ("old", "41,25", ""))
        self.assertTrue((folder / "quality.csv").exists())


if __name__ == "__main__":
    unittest.main()
