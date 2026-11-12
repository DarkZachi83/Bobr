"""Wyniki benchmarków w SQLite (SPEC §5c) – plik benchmarks.sqlite obok ustawień.

Wersja schematu w `PRAGMA user_version`:
  0 – do 0.8.1: tabele `perf` (1k/8k) i `quality` (oceniane zadania). Od 0.9 to ARCHIWUM: nowy kod
      ich nie zapisuje i nie zmienia ich zawartości (jedynie brakujące kolumny ze starszych wersji).
  1 – od 0.9: benchmark prędkości. `sessions` (jeden pomiar: ustawienia wspólne), `runs` (jeden
      przebieg jednego modelu: surowe sumy + werdykt z dnia testu + kopia progów), `tasks` (15 zadań
      z odpowiedziami, bez oceny treści), `telemetry` (próbka co 1 s), `long_context` (pomiar 1k/8k).
  2 – od 0.10: moduł jakości (`quality_sessions`, `quality_runs`, `quality_tasks`).
  3 – od 0.12: diagnostyka – `bos_checks` (token początku, jeden wynik na model i wersję pliku),
      `diag_sessions` + `diag_tasks` (kruchość promptu); kolumna `bos_status` w `runs` i `quality_runs`
      (z jakim tokenem początku liczono wynik).

Brak odczytu (czujnik niedostępny, pole nieobecne w odpowiedzi silnika) = NULL, nigdy 0.
Przed migracją robimy kopię pliku (API kopii SQLite), migracja idzie w jednej transakcji.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS perf (
    id INTEGER PRIMARY KEY,
    measured_at TEXT NOT NULL,
    model TEXT NOT NULL,
    runtime TEXT NOT NULL,
    runtime_version TEXT,
    hardware TEXT,
    power TEXT,
    context INTEGER NOT NULL,
    pp_tps REAL, ttft_ms REAL, tg_tps REAL,
    prompt_tokens INTEGER, gen_tokens INTEGER,
    gpu_share REAL,
    predicted TEXT,
    suite TEXT
);
CREATE INDEX IF NOT EXISTS perf_model ON perf(model, context);
CREATE TABLE IF NOT EXISTS quality (
    id INTEGER PRIMARY KEY,
    measured_at TEXT NOT NULL,
    model TEXT NOT NULL,
    runtime TEXT NOT NULL,
    runtime_version TEXT,
    hardware TEXT,
    suite TEXT NOT NULL,
    suite_version TEXT,
    test_id TEXT NOT NULL,
    category TEXT,
    passed INTEGER NOT NULL,
    strict INTEGER,
    reason TEXT,
    answer TEXT,
    subcategory TEXT,
    runs INTEGER,
    identical INTEGER,
    passes INTEGER,
    prompt_tokens INTEGER,
    prompt_cached INTEGER,
    cache_warn INTEGER,
    mode TEXT,
    invalid INTEGER,
    attempts TEXT
);
CREATE INDEX IF NOT EXISTS quality_model ON quality(model, suite);
"""
# Kolumny dodane później – starsze bazy dostają je przy otwarciu (ALTER TABLE).
EXTRA_COLUMNS = {"power_draw_w": "REAL", "power_limit_w": "REAL", "power_max_w": "REAL",
                 "vram_busy_mb": "REAL"}       # zajęte przez system przed testem (np. pulpit)

QUALITY_EXTRA = {"subcategory": "TEXT", "runs": "INTEGER", "identical": "INTEGER", "passes": "INTEGER",
                 "prompt_tokens": "INTEGER", "prompt_cached": "INTEGER", "cache_warn": "INTEGER",
                 "mode": "TEXT", "invalid": "INTEGER", "attempts": "TEXT"}   # attempts: JSON, każda próba

# Schemat 1 (0.9). Czasy silnika (Ollama) w ms, czasy rzeczywiste mierzone przez aplikację – *_wall_*.
# Pola *_json: JSON (listy, słowniki); thresholds_json = rzeczywiste wartości progów w chwili testu.
SCHEMA_V1 = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    app_version TEXT,
    runtime TEXT,
    runtime_version TEXT,
    hardware TEXT,
    hardware_json TEXT,
    power TEXT,
    suite TEXT,
    suite_version TEXT,
    runs_per_model INTEGER,
    models_json TEXT,
    settings_json TEXT,
    thresholds_version TEXT,
    thresholds_json TEXT,
    note TEXT
);
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    run_no INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    model_json TEXT,
    num_ctx INTEGER,
    gpu_share REAL,
    stabilization TEXT,
    cooling_seconds REAL,
    baseline_json TEXT,
    load_ms REAL,
    load_wall_ms REAL,
    gen_tokens_total INTEGER,
    gen_ms_total REAL,
    gen_tps REAL,
    pp_tokens_total INTEGER,
    pp_ms_total REAL,
    pp_tps REAL,
    ttft_wall_median_ms REAL,
    ttft_wall_max_ms REAL,
    engine_ms_total REAL,
    wall_ms_total REAL,
    overhead_ratio REAL,
    tasks_counted INTEGER,
    tasks_error INTEGER,
    gen_level TEXT,
    ttft_level TEXT,
    verdict TEXT,
    verdict_measure TEXT,
    thresholds_version TEXT,
    thresholds_json TEXT,
    warnings_json TEXT
);
CREATE INDEX runs_session ON runs(session_id, run_no);
CREATE INDEX runs_model ON runs(model, started_at);
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_no INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    category TEXT,
    started_at TEXT,
    prompt_text TEXT,
    num_ctx INTEGER,
    num_predict INTEGER,
    prompt_tokens INTEGER,
    prompt_cached INTEGER,
    gen_tokens INTEGER,
    load_ms REAL,
    prompt_eval_ms REAL,
    eval_ms REAL,
    total_ms REAL,
    ttft_wall_ms REAL,
    wall_ms REAL,
    done_reason TEXT,
    response_text TEXT,
    error TEXT,
    counted_gen INTEGER,
    counted_pp INTEGER,
    flags_json TEXT
);
CREATE INDEX tasks_run ON tasks(run_id, task_no);
CREATE TABLE telemetry (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
    t TEXT NOT NULL,
    phase TEXT NOT NULL,
    task_no INTEGER,
    cpu_pct REAL,
    cpu_cores_json TEXT,
    cpu_temp_c REAL,
    cpu_power_w REAL,
    ram_used_mb REAL,
    ram_available_mb REAL,
    swap_used_mb REAL,
    swap_in_bps REAL,
    swap_out_bps REAL,
    gpu_util_pct REAL,
    gpu_temp_c REAL,
    gpu_power_w REAL,
    gpu_clock_mhz REAL,
    vram_used_mb REAL,
    vram_total_mb REAL,
    vram_temp_c REAL,
    gpu_throttle TEXT,
    extra_json TEXT
);
CREATE INDEX telemetry_run ON telemetry(run_id, t);
CREATE INDEX telemetry_time ON telemetry(t);
CREATE TABLE long_context (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    num_ctx INTEGER NOT NULL,
    target_tokens INTEGER,
    reserve_tokens INTEGER,
    prompt_tokens INTEGER,
    prompt_cached INTEGER,
    gen_tokens INTEGER,
    free_tokens INTEGER,
    prompt_eval_ms REAL,
    eval_ms REAL,
    pp_tps REAL,
    tg_tps REAL,
    ttft_wall_ms REAL,
    gpu_share REAL,
    done_reason TEXT,
    valid INTEGER,
    invalid_reason TEXT,
    calibration_json TEXT,
    error TEXT
);
CREATE INDEX long_context_run ON long_context(run_id, num_ctx);
"""
# Schemat 2 (0.10): moduł jakości – własne tabele, bez związku z werdyktem prędkości.
SCHEMA_V2 = """
CREATE TABLE quality_sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    app_version TEXT,
    runtime_version TEXT,
    hardware TEXT,
    attempts INTEGER,
    tests_json TEXT,
    models_json TEXT,
    sandbox TEXT,
    note TEXT
);
CREATE TABLE quality_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES quality_sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    suite TEXT NOT NULL,
    suite_version TEXT,
    lang TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    total INTEGER,
    evaluated INTEGER,
    correct INTEGER,
    format_noncompliant INTEGER,
    not_evaluated INTEGER,
    manual INTEGER,
    by_category_json TEXT,
    by_level_json TEXT
);
CREATE INDEX quality_runs_session ON quality_runs(session_id, model);
CREATE TABLE quality_tasks (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES quality_runs(id) ON DELETE CASCADE,
    task_no INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    category TEXT,
    level TEXT,
    prompt_text TEXT,
    expected TEXT,
    correct INTEGER,
    format_compliant INTEGER,
    reason TEXT,
    value TEXT,
    answer TEXT,
    attempts_json TEXT,
    differing INTEGER,
    manual INTEGER DEFAULT 0
);
CREATE INDEX quality_tasks_run ON quality_tasks(run_id, task_no);
"""
# Schemat 3 (0.12): diagnostyka – token początku i kruchość promptu.
SCHEMA_V3 = """
CREATE TABLE bos_checks (
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    digest TEXT,
    checked_at TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    expects INTEGER,
    add_bos INTEGER,
    gguf_template_has INTEGER,
    gguf_template_source TEXT,
    bos_text TEXT,
    template_has INTEGER,
    without_count INTEGER,
    with_text_count INTEGER,
    effective INTEGER,
    runtime_version TEXT,
    app_version TEXT
);
CREATE INDEX bos_checks_model ON bos_checks(model, digest);
CREATE TABLE diag_sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    model TEXT NOT NULL,
    digest TEXT,
    suite TEXT,
    suite_version TEXT,
    lang TEXT,
    fragile_threshold REAL,
    tie_threshold REAL,
    bos_status TEXT,
    app_version TEXT,
    runtime_version TEXT,
    stable INTEGER,
    fragile INTEGER,
    tie INTEGER,
    not_run INTEGER
);
CREATE INDEX diag_sessions_model ON diag_sessions(model, suite, lang);
CREATE TABLE diag_tasks (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES diag_sessions(id) ON DELETE CASCADE,
    task_no INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    label TEXT NOT NULL,
    steps INTEGER,
    min_margin REAL,
    min_pos INTEGER,
    min_pair_json TEXT,
    fragile_steps INTEGER,
    tie_steps INTEGER,
    first_pos INTEGER,
    first_pair_json TEXT,
    first_context TEXT,
    scope TEXT,
    answer_steps INTEGER,
    after_fragile INTEGER,
    after_tie INTEGER,
    answer TEXT,
    reason TEXT
);
CREATE INDEX diag_tasks_session ON diag_tasks(session_id, task_no);
ALTER TABLE runs ADD COLUMN bos_status TEXT;
ALTER TABLE quality_runs ADD COLUMN bos_status TEXT
"""
MIGRATIONS = {1: SCHEMA_V1, 2: SCHEMA_V2, 3: SCHEMA_V3}
QUALITY_TABLES = ("quality_sessions", "quality_runs", "quality_tasks")
DIAG_TABLES = ("bos_checks", "diag_sessions", "diag_tasks")

NEW_TABLES = ("sessions", "runs", "tasks", "telemetry", "long_context")
ARCHIVE_TABLES = ("perf", "quality")

# Ustawienie „telemetry_retention_days”: None = bezterminowo (domyślnie).
RETENTION_CHOICES = (30, 90, 180, 365, None)


def retention_days(value) -> Optional[int]:
    """Wartość z ustawień → liczba dni albo None (bezterminowo). Nieznana wartość = bezterminowo,
    bo przy błędzie w ustawieniach lepiej niczego nie usuwać."""
    try:
        days = None if value in (None, "", "forever") else int(value)
    except (TypeError, ValueError):
        return None
    return days if days in RETENTION_CHOICES else None


def new_id() -> str:
    return uuid.uuid4().hex


def _json(value) -> Optional[str]:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _attempts_json(r) -> Optional[str]:
    """Wszystkie próby zadania (pełne odpowiedzi, pamięć promptu, warunki) – do eksportu i analizy."""
    from dataclasses import asdict
    attempts = getattr(r, "attempts", ())
    return json.dumps([asdict(a) for a in attempts], ensure_ascii=False) if attempts else None


def _backup_path(path: Path, day: str) -> Path:
    candidate = path.with_name(f"{path.name}.bak-{day}")
    n = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{day}-{n}")
        n += 1
    return candidate


class BenchDb:
    def __init__(self, path: Path, today: Optional[str] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backup: Optional[Path] = None              # kopia zrobiona przy tym otwarciu (do komunikatu)
        existed = self.path.exists() and self.path.stat().st_size > 0
        with closing(self._connect()) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
        if existed and version < SCHEMA_VERSION:
            self.backup = self._make_backup(today or datetime.now().strftime("%Y-%m-%d"))
        with closing(self._connect()) as db, db:
            db.executescript(SCHEMA)
            have = {row[1] for row in db.execute("PRAGMA table_info(perf)")}
            for name, kind in EXTRA_COLUMNS.items():
                if name not in have:
                    db.execute(f"ALTER TABLE perf ADD COLUMN {name} {kind}")
            have = {row[1] for row in db.execute("PRAGMA table_info(quality)")}
            for name, kind in QUALITY_EXTRA.items():          # tabela z wersji roboczej 12c
                if name not in have:
                    db.execute(f"ALTER TABLE quality ADD COLUMN {name} {kind}")
        self._migrate(version)
        self._columns = self._read_columns()

    # ------------------------------------------------------------------ połączenie i migracje
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def _make_backup(self, day: str) -> Path:
        target = _backup_path(self.path, day)
        with closing(sqlite3.connect(self.path)) as src, closing(sqlite3.connect(target)) as dst:
            src.backup(dst)
        return target

    def _migrate(self, version: int) -> None:
        """Każdy krok w osobnej transakcji razem z podniesieniem user_version: błąd = nic się nie zmienia."""
        for target in range(version + 1, SCHEMA_VERSION + 1):
            db = self._connect()
            db.isolation_level = None                     # sami sterujemy transakcją (DDL też w niej)
            try:
                db.execute("BEGIN")
                for statement in MIGRATIONS[target].split(";"):
                    if statement.strip():
                        db.execute(statement)
                db.execute(f"PRAGMA user_version = {target}")
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
            finally:
                db.close()

    def _read_columns(self) -> dict[str, set[str]]:
        with closing(self._connect()) as db:
            return {t: {row[1] for row in db.execute(f"PRAGMA table_info({t})")}
                    for t in NEW_TABLES + QUALITY_TABLES + DIAG_TABLES}

    def schema_version(self) -> int:
        with closing(self._connect()) as db:
            return db.execute("PRAGMA user_version").fetchone()[0]

    def _row(self, table: str, values: dict) -> tuple[str, list]:
        """Kolumny spoza schematu to błąd programu (literówka), nie cicha utrata danych."""
        unknown = set(values) - self._columns[table]
        if unknown:
            raise KeyError(f"{table}: nieznane kolumny {sorted(unknown)}")
        cols = list(values)
        vals = [_json(v) if c.endswith("_json") and not isinstance(v, str) else v
                for c, v in values.items()]
        return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals

    def _insert(self, table: str, values: dict) -> int:
        sql, vals = self._row(table, values)
        with closing(self._connect()) as db, db:
            return db.execute(sql, vals).lastrowid

    def _update(self, table: str, key: str, values: dict) -> None:
        unknown = set(values) - self._columns[table]
        if unknown:
            raise KeyError(f"{table}: nieznane kolumny {sorted(unknown)}")
        sets = ", ".join(f"{c} = ?" for c in values)
        vals = [_json(v) if c.endswith("_json") and not isinstance(v, str) else v for c, v in values.items()]
        with closing(self._connect()) as db, db:
            db.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*vals, key])

    def _select(self, sql: str, args: Iterable = ()) -> list[dict]:
        with closing(self._connect()) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute(sql, tuple(args)).fetchall()]

    # ------------------------------------------------------------------ benchmark prędkości (0.9)
    def start_session(self, **values) -> str:
        sid = values.pop("id", None) or new_id()
        self._insert("sessions", {"id": sid, **values})
        return sid

    def finish_session(self, session_id: str, status: str = "finished", **values) -> None:
        self._update("sessions", session_id, {"status": status, **values})

    def start_run(self, session_id: str, **values) -> str:
        rid = values.pop("id", None) or new_id()
        self._insert("runs", {"id": rid, "session_id": session_id, **values})
        return rid

    def update_run(self, run_id: str, **values) -> None:
        self._update("runs", run_id, values)

    def add_task(self, run_id: str, **values) -> int:
        return self._insert("tasks", {"run_id": run_id, **values})

    def add_long_context(self, run_id: str, **values) -> int:
        return self._insert("long_context", {"run_id": run_id, **values})

    def add_telemetry(self, samples: Iterable[dict]) -> int:
        """Wiele próbek naraz (próbnik zapisuje paczkami). Brakujące pola zostają NULL."""
        n = 0
        with closing(self._connect()) as db, db:
            for s in samples:
                sql, vals = self._row("telemetry", s)
                db.execute(sql, vals)
                n += 1
        return n

    def sessions(self, model: Optional[str] = None) -> list[dict]:
        """Lista do zakładki „Historia”: najnowsze najpierw, z liczbą przebiegów."""
        where, args = "", []
        if model:
            where, args = "WHERE s.id IN (SELECT session_id FROM runs WHERE model = ?)", [model]
        return self._select(
            "SELECT s.*, (SELECT COUNT(*) FROM runs r WHERE r.session_id = s.id) AS run_count"
            f" FROM sessions s {where} ORDER BY s.started_at DESC", args)

    def session(self, session_id: str) -> Optional[dict]:
        rows = self._select("SELECT * FROM sessions WHERE id = ?", [session_id])
        return rows[0] if rows else None

    def runs(self, session_id: str) -> list[dict]:
        return self._select("SELECT * FROM runs WHERE session_id = ? ORDER BY model, run_no", [session_id])

    def run(self, run_id: str) -> Optional[dict]:
        rows = self._select("SELECT * FROM runs WHERE id = ?", [run_id])
        return rows[0] if rows else None

    def last_speed(self, model: str) -> Optional[dict]:
        """Ostatni ukończony przebieg benchmarku prędkości modelu – do prognozy czasu testów jakości."""
        rows = self._select("SELECT * FROM runs WHERE model = ? AND status = 'finished' ORDER BY started_at DESC "
                            "LIMIT 1", [model])
        return rows[0] if rows else None

    def run_models(self) -> list[str]:
        """Modele z przebiegów 0.9 – do filtra w zakładce Historia."""
        return [r["model"] for r in self._select("SELECT DISTINCT model FROM runs ORDER BY model")]

    def tasks(self, run_id: str) -> list[dict]:
        return self._select("SELECT * FROM tasks WHERE run_id = ? ORDER BY task_no", [run_id])

    def long_context(self, run_id: str) -> list[dict]:
        return self._select("SELECT * FROM long_context WHERE run_id = ? ORDER BY num_ctx, id", [run_id])

    def telemetry(self, *, run_id: Optional[str] = None, session_id: Optional[str] = None) -> list[dict]:
        if run_id is not None:
            return self._select("SELECT * FROM telemetry WHERE run_id = ? ORDER BY t, id", [run_id])
        return self._select("SELECT * FROM telemetry WHERE session_id = ? ORDER BY t, id", [session_id])

    def delete_session(self, session_id: str) -> None:
        """Ręczne usunięcie sesji (z potwierdzeniem w interfejsie): przebiegi, zadania, telemetria.
        Tabel archiwum (`perf`, `quality`) to nie dotyczy."""
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM sessions WHERE id = ?", [session_id])

    def prune_telemetry(self, days, now: Optional[datetime] = None, vacuum: bool = True) -> int:
        """Usuwa WYŁĄCZNIE szczegółowe próbki telemetrii starsze niż `days` dni. Wyniki zbiorcze,
        werdykty, ustawienia, ostrzeżenia, zadania i odpowiedzi zostają. None = bezterminowo."""
        days = retention_days(days)
        if days is None:
            return 0
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).astimezone(timezone.utc)
        with closing(self._connect()) as db, db:
            # t zapisujemy w ISO 8601 z przesunięciem; porównanie po UTC przez julianday
            removed = db.execute("DELETE FROM telemetry WHERE julianday(t) < julianday(?)",
                                 [cutoff.isoformat()]).rowcount
        if removed and vacuum:
            with closing(self._connect()) as db:
                db.execute("VACUUM")                       # plik rzeczywiście się zmniejsza
        return removed

    # ------------------------------------------------------------------ moduł jakości (0.10)
    def start_quality_session(self, **values) -> str:
        sid = values.pop("id", None) or new_id()
        self._insert("quality_sessions", {"id": sid, **values})
        return sid

    def finish_quality_session(self, session_id: str, status: str = "finished", **values) -> None:
        self._update("quality_sessions", session_id, {"status": status, **values})

    def start_quality_run(self, session_id: str, **values) -> str:
        rid = values.pop("id", None) or new_id()
        self._insert("quality_runs", {"id": rid, "session_id": session_id, **values})
        return rid

    def update_quality_run(self, run_id: str, **values) -> None:
        self._update("quality_runs", run_id, values)

    def add_quality_task(self, run_id: str, **values) -> int:
        return self._insert("quality_tasks", {"run_id": run_id, **values})

    def update_quality_task(self, task_row: int, **values) -> None:
        self._update("quality_tasks", task_row, values)

    def quality_sessions(self, model: Optional[str] = None) -> list[dict]:
        where, args = "", []
        if model:
            where, args = "WHERE s.id IN (SELECT session_id FROM quality_runs WHERE model = ?)", [model]
        return self._select(f"SELECT s.* FROM quality_sessions s {where} ORDER BY s.started_at DESC", args)

    def quality_suite_used(self, suite: str, version: str) -> bool:
        """Czy ta wersja testu ma już wyniki – edytor podnosi wtedy wersję przy zapisie."""
        if not self.path.exists():
            return False
        return bool(self._select("SELECT 1 FROM quality_runs WHERE suite = ? AND suite_version = ? LIMIT 1",
                                 [suite, version]))

    def quality_runs(self, session_id: str) -> list[dict]:
        return self._select("SELECT * FROM quality_runs WHERE session_id = ? ORDER BY started_at, model, suite, lang",
                            [session_id])

    def quality_tasks(self, run_id: str) -> list[dict]:
        return self._select("SELECT * FROM quality_tasks WHERE run_id = ? ORDER BY task_no", [run_id])

    def delete_quality_session(self, session_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM quality_sessions WHERE id = ?", [session_id])

    # ------------------------------------------------------------------ archiwum do 0.8.1
    # ------------------------------------------------------------------ diagnostyka (schemat 3)
    BOS_FIELDS = {"status": "status", "reason": "reason", "expects": "expects", "add_bos": "add_bos",
                  "gguf_template_has": "gguf_template_has", "gguf_template_source": "gguf_template_source",
                  "bos_text": "bos_text", "template_has": "template_has", "without": "without_count",
                  "with_text": "with_text_count", "effective": "effective"}

    def add_bos_check(self, check, *, checked_at: str, runtime_version: Optional[str] = None,
                      app_version: Optional[str] = None) -> int:
        """Wynik sprawdzenia tokenu początku (aibobr.diagnostics.bos.BosCheck); każde sprawdzenie osobno."""
        values = {"model": check.model, "digest": check.digest, "checked_at": checked_at,
                  "runtime_version": runtime_version, "app_version": app_version}
        for attr, col in self.BOS_FIELDS.items():
            v = getattr(check, attr)
            values[col] = int(v) if isinstance(v, bool) else v
        return self._insert("bos_checks", values)

    def bos_check(self, model: str, digest: Optional[str] = None):
        """Ostatnie sprawdzenie dla modelu (i tej wersji pliku, gdy podany skrót) → BosCheck albo None."""
        from aibobr.diagnostics.bos import BosCheck
        if not self.path.exists():
            return None
        sql, args = "SELECT * FROM bos_checks WHERE model = ?", [model]
        if digest is not None:
            sql += " AND digest = ?"
            args.append(digest)
        rows = self._select(sql + " ORDER BY id DESC LIMIT 1", args)
        if not rows:
            return None
        row = rows[0]
        data = {"model": row["model"], "digest": row["digest"]}
        for attr, col in self.BOS_FIELDS.items():
            v = row[col]
            data[attr] = bool(v) if attr in ("expects", "add_bos", "gguf_template_has", "template_has") \
                and v is not None else v
        return BosCheck.from_dict(data)

    def bos_checks(self) -> list[dict]:
        """Ostatnie sprawdzenie każdego modelu (do tabeli w zakładce Diagnostyka)."""
        return self._select("SELECT * FROM bos_checks WHERE id IN (SELECT MAX(id) FROM bos_checks GROUP BY model) "
                            "ORDER BY model")

    def start_diag_session(self, **values) -> str:
        sid = values.pop("id", None) or new_id()
        self._insert("diag_sessions", {"id": sid, **values})
        return sid

    def finish_diag_session(self, session_id: str, status: str = "finished", **values) -> None:
        self._update("diag_sessions", session_id, {"status": status, **values})

    def add_diag_task(self, session_id: str, task_no: int, item) -> int:
        """Wynik kruchości jednego zadania (aibobr.diagnostics.fragility.TaskFragility)."""
        d = item.as_dict() if hasattr(item, "as_dict") else dict(item)
        return self._insert("diag_tasks", {
            "session_id": session_id, "task_no": task_no, "task_id": d["task_id"], "label": d["label"],
            "steps": d.get("steps"), "min_margin": d.get("min_margin"), "min_pos": d.get("min_pos"),
            "min_pair_json": d.get("min_pair"), "fragile_steps": d.get("fragile_steps"),
            "tie_steps": d.get("tie_steps"), "first_pos": d.get("first_pos"), "first_pair_json": d.get("first_pair"),
            "first_context": d.get("first_context"), "scope": d.get("scope"), "answer_steps": d.get("answer_steps"),
            "after_fragile": d.get("after_fragile"), "after_tie": d.get("after_tie"), "answer": d.get("answer"),
            "reason": d.get("reason")})

    def diag_sessions(self, model: Optional[str] = None) -> list[dict]:
        if not self.path.exists():
            return []
        if model:
            return self._select("SELECT * FROM diag_sessions WHERE model = ? ORDER BY started_at DESC", [model])
        return self._select("SELECT * FROM diag_sessions ORDER BY started_at DESC")

    def diag_tasks(self, session_id: str) -> list[dict]:
        rows = self._select("SELECT * FROM diag_tasks WHERE session_id = ? ORDER BY task_no", [session_id])
        for row in rows:
            for key in ("min_pair_json", "first_pair_json"):
                row[key] = json.loads(row[key]) if row[key] else None
        return rows

    def last_fragility(self, model: str, suite: str, lang: str, suite_version: Optional[str] = None) -> dict:
        """Ostatnia zakończona diagnostyka kruchości tego testu dla modelu: {zadanie: etykieta} – do znacznika
        w zakładce Jakość. Pusty słownik – nie było."""
        if not self.path.exists():
            return {}
        sql = ("SELECT id FROM diag_sessions WHERE model = ? AND suite = ? AND lang = ? AND status = 'finished'")
        args = [model, suite, lang]
        if suite_version is not None:
            sql += " AND suite_version = ?"
            args.append(suite_version)
        rows = self._select(sql + " ORDER BY started_at DESC LIMIT 1", args)
        if not rows:
            return {}
        return {t["task_id"]: t["label"] for t in self.diag_tasks(rows[0]["id"])}

    def delete_diag_session(self, session_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM diag_sessions WHERE id = ?", [session_id])

    def add_perf(self, r, *, measured_at: str, runtime: str, runtime_version: Optional[str],
                 hardware: str, power: Optional[str], suite: str,
                 power_limit_w: Optional[float] = None, power_max_w: Optional[float] = None,
                 vram_busy_mb: Optional[float] = None) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO perf (measured_at, model, runtime, runtime_version, hardware, power, context,"
                " pp_tps, ttft_ms, tg_tps, prompt_tokens, gen_tokens, gpu_share, predicted, suite,"
                " power_draw_w, power_limit_w, power_max_w, vram_busy_mb)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (measured_at, r.model, runtime, runtime_version, hardware, power, r.context, r.pp_tps,
                 r.ttft_ms, r.tg_tps, r.prompt_tokens, r.gen_tokens, r.gpu_share, r.predicted, suite,
                 getattr(r, "power_draw_w", None), power_limit_w, power_max_w, vram_busy_mb))

    def add_quality(self, r, *, measured_at: str, runtime: str, runtime_version: Optional[str],
                    hardware: str, suite_version: Optional[str] = None) -> None:
        """Etap 12c: jeden wiersz na zadanie – z pełną odpowiedzią modelu (do porównań w czasie)."""
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO quality (measured_at, model, runtime, runtime_version, hardware, suite,"
                " suite_version, test_id, category, passed, strict, reason, answer, subcategory, runs,"
                " identical, passes, prompt_tokens, prompt_cached, cache_warn, mode, invalid, attempts)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (measured_at, r.model, runtime, runtime_version, hardware, r.suite, suite_version, r.test_id,
                 r.category, int(r.passed), None if r.strict is None else int(r.strict), r.reason, r.answer,
                 r.subcategory, r.runs, r.identical, r.passes, getattr(r, "prompt_tokens", None),
                 getattr(r, "cached", None), int(bool(getattr(r, "cache_warn", False))),
                 getattr(r, "mode", None), getattr(r, "invalid", 0), _attempts_json(r)))

    def quality_count(self) -> int:
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM quality").fetchone()[0]

    def archive(self, table: str) -> list[dict]:
        """Archiwum (przed 0.9) do podglądu i eksportu – tylko odczyt."""
        if table not in ARCHIVE_TABLES:
            raise KeyError(table)
        return self._select(f"SELECT * FROM {table} ORDER BY measured_at, id")

    def latest(self, model: str, hardware: str) -> list[dict]:
        """Ostatni pomiar każdej długości kontekstu dla modelu na tym sprzęcie."""
        with closing(self._connect()) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM perf WHERE model=? AND hardware=? AND id IN ("
                " SELECT MAX(id) FROM perf WHERE model=? AND hardware=? GROUP BY context)"
                " ORDER BY context", (model, hardware, model, hardware)).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM perf").fetchone()[0]
