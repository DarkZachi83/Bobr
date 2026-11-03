"""Wyniki benchmarków w SQLite (etap 12a, SPEC §5c) – plik benchmarks.sqlite obok ustawień.

Każdy wiersz = jeden model przy jednej długości kontekstu na tym sprzęcie, z ustawieniami,
które wpływają na wynik (program i jego wersja, zasilanie), i przewidywaniem AI Bóbra.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional

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
"""


class BenchDb:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(SCHEMA)

    def _connect(self):
        return sqlite3.connect(self.path)

    def add_perf(self, r, *, measured_at: str, runtime: str, runtime_version: Optional[str],
                 hardware: str, power: Optional[str], suite: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO perf (measured_at, model, runtime, runtime_version, hardware, power, context,"
                " pp_tps, ttft_ms, tg_tps, prompt_tokens, gen_tokens, gpu_share, predicted, suite)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (measured_at, r.model, runtime, runtime_version, hardware, power, r.context, r.pp_tps,
                 r.ttft_ms, r.tg_tps, r.prompt_tokens, r.gen_tokens, r.gpu_share, r.predicted, suite))

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
