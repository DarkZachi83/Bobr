"""Baza benchmarków 0.9: migracja z kopią, nietknięte archiwum, NULL zamiast 0, retencja telemetrii."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from aibobr.app import bench_db
from aibobr.app.bench_db import BenchDb

# Baza z 0.8.1 (user_version 0): pełne tabele perf i quality z rekordami.
LEGACY_081 = """
CREATE TABLE perf (id INTEGER PRIMARY KEY, measured_at TEXT NOT NULL, model TEXT NOT NULL, runtime TEXT NOT NULL,
  runtime_version TEXT, hardware TEXT, power TEXT, context INTEGER NOT NULL, pp_tps REAL, ttft_ms REAL, tg_tps REAL,
  prompt_tokens INTEGER, gen_tokens INTEGER, gpu_share REAL, predicted TEXT, suite TEXT,
  power_draw_w REAL, power_limit_w REAL, power_max_w REAL, vram_busy_mb REAL);
CREATE TABLE quality (id INTEGER PRIMARY KEY, measured_at TEXT NOT NULL, model TEXT NOT NULL, runtime TEXT NOT NULL,
  runtime_version TEXT, hardware TEXT, suite TEXT NOT NULL, suite_version TEXT, test_id TEXT NOT NULL, category TEXT,
  passed INTEGER NOT NULL, strict INTEGER, reason TEXT, answer TEXT, subcategory TEXT, runs INTEGER, identical INTEGER,
  passes INTEGER, prompt_tokens INTEGER, prompt_cached INTEGER, cache_warn INTEGER, mode TEXT, invalid INTEGER,
  attempts TEXT);
INSERT INTO perf (measured_at, model, runtime, runtime_version, hardware, context, pp_tps, ttft_ms, tg_tps)
  VALUES ('2026-11-04 10:00', 'gemma2:9b', 'ollama', '0.40.14', 'RTX 4060', 8192, 1000.5, 344, 41.2);
INSERT INTO quality (measured_at, model, runtime, suite, test_id, passed, answer, prompt_cached, attempts)
  VALUES ('2026-11-06 21:00', 'minitron', 'ollama', 'pl_jezyk_v1', 'pl_henrykowska', 1, 'Daj, abym ja pomielił',
          NULL, '[{"n": 1}]');
"""
# Starsza baza (przed etapem 12c-poprawki): bez późniejszych kolumn.
LEGACY_OLD = """
CREATE TABLE perf (id INTEGER PRIMARY KEY, measured_at TEXT NOT NULL, model TEXT NOT NULL, runtime TEXT NOT NULL,
  runtime_version TEXT, hardware TEXT, power TEXT, context INTEGER NOT NULL, pp_tps REAL, ttft_ms REAL, tg_tps REAL,
  prompt_tokens INTEGER, gen_tokens INTEGER, gpu_share REAL, predicted TEXT, suite TEXT);
INSERT INTO perf (measured_at, model, runtime, context, tg_tps) VALUES ('2026-11-03 10:00', 'm', 'ollama', 1024, 30);
"""


def make_db(path, script):
    with closing(sqlite3.connect(path)) as db:
        db.executescript(script)
    return path


def dump(path, table):
    with closing(sqlite3.connect(path)) as db:
        return db.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "benchmarks.sqlite"

    def test_legacy_backup_and_untouched_archive(self):
        make_db(self.path, LEGACY_081)
        before = {t: dump(self.path, t) for t in ("perf", "quality")}
        db = BenchDb(self.path, today="2026-11-09")
        self.assertEqual(db.schema_version(), bench_db.SCHEMA_VERSION)
        self.assertEqual(db.backup, self.dir / "benchmarks.sqlite.bak-2026-11-09")
        for t in ("perf", "quality"):
            self.assertEqual(dump(self.path, t), before[t])          # archiwum bez zmian
            self.assertEqual(dump(db.backup, t), before[t])          # kopia ma te same rekordy
        with closing(sqlite3.connect(db.backup)) as b:
            self.assertEqual(b.execute("PRAGMA user_version").fetchone()[0], 0)
        self.assertEqual(db.archive("quality")[0]["answer"], "Daj, abym ja pomielił")
        self.assertIsNone(db.archive("quality")[0]["prompt_cached"])
        again = BenchDb(self.path, today="2026-11-09")               # drugie otwarcie: bez nowej kopii
        self.assertIsNone(again.backup)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         ["benchmarks.sqlite", "benchmarks.sqlite.bak-2026-11-09"])

    def test_old_database_gets_missing_columns_and_second_backup_name(self):
        make_db(self.path, LEGACY_OLD)
        (self.dir / "benchmarks.sqlite.bak-2026-11-09").write_bytes(b"inna kopia")   # nie nadpisujemy
        db = BenchDb(self.path, today="2026-11-09")
        self.assertEqual(db.backup.name, "benchmarks.sqlite.bak-2026-11-09-2")
        row = db.archive("perf")[0]
        self.assertEqual((row["model"], row["tg_tps"], row["vram_busy_mb"]), ("m", 30, None))
        self.assertEqual(db.quality_count(), 0)

    def test_new_database_without_backup(self):
        db = BenchDb(self.path)
        self.assertEqual(db.schema_version(), bench_db.SCHEMA_VERSION)
        self.assertIsNone(db.backup)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["benchmarks.sqlite"])

    def test_failed_migration_changes_nothing(self):
        make_db(self.path, LEGACY_081)
        before = dump(self.path, "perf")
        broken = bench_db.SCHEMA_V1 + "; CREATE TABLE błąd ("
        with mock.patch.dict(bench_db.MIGRATIONS, {1: broken}):
            with self.assertRaises(sqlite3.Error):
                BenchDb(self.path, today="2026-11-09")
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 0)
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {"perf", "quality"})                 # żadnej nowej tabeli
        self.assertEqual(dump(self.path, "perf"), before)
        self.assertTrue((self.dir / "benchmarks.sqlite.bak-2026-11-09").exists())
        self.assertEqual(BenchDb(self.path, today="2026-11-10").schema_version(), bench_db.SCHEMA_VERSION)   # kolejna próba się udaje


class QualitySchemaTest(unittest.TestCase):
    def test_upgrade_from_09_keeps_speed_history(self):
        path = Path(tempfile.mkdtemp()) / "benchmarks.sqlite"
        with mock.patch.object(bench_db, "SCHEMA_VERSION", 1):              # baza z 0.9
            old = BenchDb(path, today="2026-11-09")
            sid = old.start_session(started_at="2026-11-09T10:00:00+01:00", status="finished")
            old.start_run(sid, model="m", run_no=1, started_at="x", gen_tps=61.2, verdict="fast")
        before = dump(path, "runs")
        db = BenchDb(path, today="2026-11-10")
        self.assertEqual(db.schema_version(), bench_db.SCHEMA_VERSION)
        self.assertEqual(db.backup.name, "benchmarks.sqlite.bak-2026-11-10")
        after = [row[:-1] for row in dump(path, "runs")]                    # bez nowej kolumny bos_status (3)
        self.assertEqual(after, before)                                     # historia prędkości nietknięta
        self.assertEqual([row[-1] for row in dump(path, "runs")], [None])
        self.assertEqual(dump(db.backup, "runs"), before)

    def test_quality_session_cascade(self):
        db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        qs = db.start_quality_session(started_at="2026-11-10T10:00:00+01:00", attempts=1)
        qr = db.start_quality_run(qs, model="m", suite="scisly_v1", lang="en", started_at="x")
        row = db.add_quality_task(qr, task_no=1, task_id="S1", correct=None, reason="not_run")
        db.update_quality_task(row, correct=1, manual=1)
        t = db.quality_tasks(qr)[0]
        self.assertEqual((t["correct"], t["manual"], t["format_compliant"]), (1, 1, None))
        self.assertEqual(db.quality_sessions(model="m")[0]["id"], qs)
        db.delete_quality_session(qs)
        self.assertEqual((db.quality_runs(qs), db.quality_tasks(qr)), ([], []))


class DiagnosticsSchemaTest(unittest.TestCase):
    def test_upgrade_from_010_keeps_quality_history(self):
        path = Path(tempfile.mkdtemp()) / "benchmarks.sqlite"
        with mock.patch.object(bench_db, "SCHEMA_VERSION", 2):              # baza z 0.10 / 0.11
            old = BenchDb(path, today="2026-11-11")
            qs = old.start_quality_session(started_at="2026-11-11T10:00:00+01:00", attempts=1)
            old.start_quality_run(qs, model="m", suite="scisly_v1", lang="pl", started_at="x", correct=12)
        before = dump(path, "quality_runs")
        db = BenchDb(path, today="2026-11-12")
        self.assertEqual(db.schema_version(), bench_db.SCHEMA_VERSION)
        self.assertEqual(db.backup.name, "benchmarks.sqlite.bak-2026-11-12")
        self.assertEqual([row[:-1] for row in dump(path, "quality_runs")], before)
        self.assertEqual(dump(db.backup, "quality_runs"), before)
        self.assertEqual(db.bos_checks(), [])

    def test_upgrade_from_012_adds_steps(self):
        from aibobr.diagnostics.fragility import TaskFragility
        path = Path(tempfile.mkdtemp()) / "benchmarks.sqlite"
        with mock.patch.object(bench_db, "SCHEMA_VERSION", 3):              # baza z 0.12.0
            old = BenchDb(path, today="2026-11-12")
            ds = old.start_diag_session(started_at="x", model="m", suite="s", lang="pl")
            old._insert("diag_tasks", {"session_id": ds, "task_no": 1, "task_id": "T1", "label": "stable"})
        db = BenchDb(path, today="2026-11-13")
        self.assertEqual(db.schema_version(), 4)
        self.assertEqual(db.backup.name, "benchmarks.sqlite.bak-2026-11-13")
        self.assertIsNone(db.diag_tasks(ds)[0]["steps_json"])                 # stare wiersze – bez kroków
        db.add_diag_task(ds, 2, TaskFragility("T2", "stable", step_data=[["a", 0.9, "b", 0.1]]))
        db.finish_diag_session(ds, finished_at="y")
        self.assertEqual(db.last_fragility_tasks("m", "s", "pl")["T2"]["steps_json"], [["a", 0.9, "b", 0.1]])

    def test_bos_checks_latest_per_model_and_digest(self):
        from aibobr.diagnostics.bos import BosCheck
        db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        self.assertIsNone(db.bos_check("m"))
        first = BosCheck("m", "d1", "missing", "measured", True, "<s>", False, 0, 1, 0, False, True, "file")
        second = BosCheck("m", "d2", "ok", "measured", True, "<s>", True, 0, 1, 1, False, True, "file")
        db.add_bos_check(first, checked_at="2026-11-12T10:00:00+01:00", runtime_version="0.40.14")
        db.add_bos_check(second, checked_at="2026-11-12T11:00:00+01:00")
        self.assertEqual(db.bos_check("m", "d1"), first)                    # ta sama wersja pliku
        self.assertEqual(db.bos_check("m"), second)                         # ostatnie
        self.assertIsNone(db.bos_check("m", "d3"))
        self.assertEqual([r["status"] for r in db.bos_checks()], ["ok"])

    def test_diag_session_tasks_and_last_fragility(self):
        from aibobr.diagnostics.fragility import TaskFragility
        db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        ds = db.start_diag_session(started_at="2026-11-12T10:00:00+01:00", model="m", suite="scisly_v1",
                                   suite_version="1.1", lang="pl", fragile_threshold=0.1, tie_threshold=0.02)
        self.assertEqual(db.last_fragility("m", "scisly_v1", "pl"), {})      # niezakończona – nie liczy się
        db.add_diag_task(ds, 1, TaskFragility("S1", "stable", 4, 0.9, 0, (("3", 0.95), ("4", 0.05))))
        db.add_diag_task(ds, 2, TaskFragility("S2", "fragile", 10, 0.09, 3, (("a", 0.5), ("b", 0.41)),
                                              1, 0, 3, (("a", 0.5), ("b", 0.41)), "przed", "odp"))
        db.finish_diag_session(ds, finished_at="x", stable=1, fragile=1, tie=0, not_run=0)
        tasks = db.diag_tasks(ds)
        self.assertEqual(tasks[1]["first_pair_json"], [["a", 0.5], ["b", 0.41]])
        self.assertEqual(db.last_fragility("m", "scisly_v1", "pl"), {"S1": "stable", "S2": "fragile"})
        self.assertEqual(db.last_fragility("m", "scisly_v1", "pl", "1.0"), {})
        self.assertEqual(db.diag_sessions("m")[0]["fragile"], 1)
        db.delete_diag_session(ds)
        self.assertEqual((db.diag_sessions(), db.diag_tasks(ds)), ([], []))


class SessionDataTest(unittest.TestCase):
    def setUp(self):
        self.db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        self.thresholds = {"gen_tps": [30, 15, 6], "ttft_s": [1, 3, 8]}
        self.sid = self.db.start_session(started_at="2026-11-09T10:00:00+01:00", app_version="0.9.0",
                                         runs_per_model=3, models_json=["a", "b"],
                                         thresholds_version="1", thresholds_json=self.thresholds)

    def test_session_runs_tasks_and_verdict_with_thresholds(self):
        rid = self.db.start_run(self.sid, model="a", run_no=1, started_at="2026-11-09T10:01:00+01:00")
        self.db.add_task(rid, task_no=1, task_id="json_1", gen_tokens=120, eval_ms=2000.0, response_text="{}",
                         done_reason="stop", counted_gen=1, counted_pp=1)
        self.db.update_run(rid, status="finished", gen_tokens_total=1214, gen_ms_total=22400.0, gen_tps=54.2,
                           verdict="szybko", verdict_measure="gen", thresholds_json=self.thresholds,
                           warnings_json=["stabilizacja_niepelna"])
        self.db.finish_session(self.sid, finished_at="2026-11-09T10:20:00+01:00")
        s = self.db.sessions()[0]
        self.assertEqual((s["status"], s["run_count"], s["models_json"]), ("finished", 1, '["a", "b"]'))
        run = self.db.runs(self.sid)[0]
        self.assertEqual((run["verdict"], run["gen_tokens_total"]), ("szybko", 1214))
        import json
        self.assertEqual(json.loads(run["thresholds_json"]), self.thresholds)   # rzeczywiste wartości progów
        self.assertEqual(self.db.tasks(rid)[0]["response_text"], "{}")
        self.assertEqual(self.db.sessions(model="a")[0]["id"], self.sid)
        self.assertEqual(self.db.sessions(model="zzz"), [])

    def test_missing_values_are_null_not_zero(self):
        rid = self.db.start_run(self.sid, model="a", run_no=1, started_at="2026-11-09T10:01:00+01:00")
        self.db.add_task(rid, task_no=1, task_id="t", prompt_tokens=40)            # brak prompt_cached
        self.db.add_telemetry([{"session_id": self.sid, "run_id": rid, "t": "2026-11-09T10:01:01+01:00",
                                "phase": "baseline", "cpu_pct": 3.5, "cpu_cores_json": [1.0, 6.0],
                                "gpu_temp_c": 41.0}])
        task = self.db.tasks(rid)[0]
        self.assertIsNone(task["prompt_cached"])
        sample = self.db.telemetry(run_id=rid)[0]
        self.assertIsNone(sample["cpu_power_w"])
        self.assertIsNone(sample["gpu_throttle"])
        self.assertEqual(sample["cpu_cores_json"], "[1.0, 6.0]")
        self.assertEqual(sample["cpu_pct"], 3.5)

    def test_unknown_column_is_an_error(self):
        with self.assertRaises(KeyError):
            self.db.start_run(self.sid, model="a", run_no=1, started_at="x", verdikt="szybko")

    def test_delete_session_cascades_but_keeps_archive(self):
        from aibobr.bench.runner import PerfResult
        self.db.add_perf(PerfResult("m", 1024, tg_tps=30.0), measured_at="2026-11-03", runtime="ollama",
                         runtime_version="1", hardware="hw", power="ac", suite="perf_v1")
        rid = self.db.start_run(self.sid, model="a", run_no=1, started_at="x")
        self.db.add_task(rid, task_no=1, task_id="t")
        self.db.add_long_context(rid, num_ctx=8192, prompt_tokens=6912)
        self.db.add_telemetry([{"session_id": self.sid, "run_id": None, "t": "2026-11-09T10:00:00+01:00",
                                "phase": "cooling"}])
        self.db.delete_session(self.sid)
        with closing(sqlite3.connect(self.db.path)) as c:
            counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in bench_db.NEW_TABLES}
        self.assertEqual(set(counts.values()), {0})
        self.assertEqual(self.db.count(), 1)


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.db = BenchDb(Path(tempfile.mkdtemp()) / "benchmarks.sqlite")
        self.sid = self.db.start_session(started_at="2025-01-01T10:00:00+01:00")
        self.rid = self.db.start_run(self.sid, model="a", run_no=1, started_at="2025-01-01T10:00:00+01:00",
                                     gen_tps=54.2, verdict="szybko", warnings_json=["x"])
        self.db.add_task(self.rid, task_no=1, task_id="t", response_text="odpowiedź")
        self.now = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
        old = (self.now - timedelta(days=100)).astimezone(timezone(timedelta(hours=1))).isoformat()
        fresh = (self.now - timedelta(days=10)).isoformat()
        self.db.add_telemetry([{"session_id": self.sid, "run_id": self.rid, "t": old, "phase": "task"},
                               {"session_id": self.sid, "run_id": self.rid, "t": fresh, "phase": "task"}])

    def test_choices(self):
        self.assertEqual([bench_db.retention_days(v) for v in (30, "90", None, "", 7, "abc", 365)],
                         [30, 90, None, None, None, None, 365])

    def test_forever_keeps_everything(self):
        self.assertEqual(self.db.prune_telemetry(None, now=self.now), 0)
        self.assertEqual(len(self.db.telemetry(run_id=self.rid)), 2)

    def test_prune_removes_only_old_samples(self):
        self.assertEqual(self.db.prune_telemetry(90, now=self.now), 1)
        self.assertEqual(len(self.db.telemetry(run_id=self.rid)), 1)
        run = self.db.runs(self.sid)[0]
        self.assertEqual((run["verdict"], run["gen_tps"], run["warnings_json"]), ("szybko", 54.2, '["x"]'))
        self.assertEqual(self.db.tasks(self.rid)[0]["response_text"], "odpowiedź")
        self.assertIsNotNone(self.db.session(self.sid))
        self.assertEqual(self.db.prune_telemetry(180, now=self.now), 0)


if __name__ == "__main__":
    unittest.main()
