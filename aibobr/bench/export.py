"""Eksport wyników benchmarku do CSV (średnik i BOM – Excel z polskimi ustawieniami).

Jeden przycisk „Eksport CSV…”, do trzech plików – tylko te, dla których są dane:
  <nazwa>_wydajnosc.csv             pomiar wydajności (jak dotąd),
  <nazwa>_jakosc_proby.csv          KAŻDA próba osobno, z pełną odpowiedzią i warunkami
                                    (pierwsza zawsze jako „Próba 1/N” – nie jako werdykt),
  <nazwa>_jakosc_podsumowanie.csv   zadanie: tryb, ważne / pominięte próby, werdykt większościowy,
                                    zgodność z próbą 1, najczęstsza odpowiedź, pamięć promptu.
Brak danych o pamięci promptu to „null”, nigdy 0.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from aibobr.i18n import t

PERF = "_wydajnosc.csv"
ATTEMPTS = "_jakosc_proby.csv"
SUMMARY = "_jakosc_podsumowanie.csv"


def base_of(path: str | Path) -> Path:
    """„…/bobr_2026-11-06.csv” → „…/bobr_2026-11-06” (pliki dostają przyrostki)."""
    p = Path(path)
    for suffix in (PERF, ATTEMPTS, SUMMARY, ".csv"):
        if p.name.endswith(suffix):
            return p.with_name(p.name[: -len(suffix)])
    return p


def _yes(value: Optional[bool]) -> str:
    return "—" if value is None else t("export.yes") if value else t("export.no")


def _null(value) -> str:
    return "null" if value is None else str(value)


def _num(x: Optional[float], digits: int = 1) -> str:
    return "" if x is None else f"{x:.{digits}f}".replace(".", t("bench.decimal"))


def _write(path: Path, header: list[str], rows: Iterable[list]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


def perf_rows(results) -> list[list]:
    return [[r.model, r.context, r.predicted or "", _num(r.pp_tps, 0), _num(r.ttft_ms, 0), _num(r.tg_tps),
             f"{r.gpu_share * 100:.0f}%" if r.gpu_share is not None else "",
             f"{r.power_draw_w:.0f}" if getattr(r, "power_draw_w", None) else ""]
            for r in results if r.tg_tps]


def _reason(reason: str) -> str:
    if not reason:
        return ""
    key, _, arg = reason.partition(":")
    return t(key, arg=arg) if key.startswith("quality.") else reason


def attempt_rows(quality) -> list[list]:
    rows = []
    for r in quality:
        attempts = r.attempts or ()
        for a in attempts:
            rows.append([r.model, t(f"bench.suite.{r.suite}"), r.test_id, r.title, t(f"bench.mode.{r.mode}"),
                         f"{a.n}/{len(attempts)}", _yes(a.valid), _yes(a.passed),
                         "" if a.passed else _reason(a.reason), _null(a.cached), a.prompt_tokens,
                         _yes(a.released), _num(a.load_seconds), a.text])
    return rows


def summary_rows(quality) -> list[list]:
    rows = []
    for r in quality:
        attempts = r.attempts or ()
        valid = [a for a in attempts if a.valid]
        texts = [a.text for a in valid] or ([r.answer] if not attempts else [])
        common = Counter(texts).most_common(1)[0][1] if texts else 0
        cached = [a.cached for a in attempts if a.cached is not None]
        first = attempts[0] if attempts else None
        rows.append([r.model, t(f"bench.suite.{r.suite}"), r.test_id, r.title, t(f"bench.mode.{r.mode}"),
                     r.runs, r.invalid, f"{r.passes}/{r.runs}",
                     "—" if not r.runs else t("export.passed") if r.passed else t("export.failed"),
                     _yes(first.passed if first else r.passed),
                     f"{sum(x == texts[0] for x in texts)}/{len(texts)}" if texts else "—",
                     f"{common}/{len(texts)}" if texts else "—", len(set(texts)),
                     min(cached) if cached else "null", max(cached) if cached else "null",
                     r.prompt_tokens or "", "⚠" if r.cache_warn else "",
                     "" if r.passed else _reason(r.reason)])
    return rows


def export_all(path: str | Path, perf=(), quality=()) -> list[Path]:
    """Zapisuje pliki, dla których są dane; zwraca ich ścieżki (pusta lista – nic do zapisania)."""
    base = base_of(path)
    written = []
    rows = perf_rows(perf)
    if rows:
        out = base.with_name(base.name + PERF)
        _write(out, [t(f"bench.col.{c}") for c in ("model", "context", "predicted", "pp", "ttft", "tg", "gpu")]
               + ["W"], rows)
        written.append(out)
    quality = [r for r in quality if not r.skipped]
    if quality:
        out = base.with_name(base.name + ATTEMPTS)
        _write(out, [t(f"export.q.{c}") for c in ("model", "suite", "test", "title", "mode", "attempt", "valid",
                                                   "passed", "reason", "cached", "prompt_tokens", "released",
                                                   "load_s", "answer")], attempt_rows(quality))
        written.append(out)
        out = base.with_name(base.name + SUMMARY)
        _write(out, [t(f"export.q.{c}") for c in ("model", "suite", "test", "title", "mode", "valid_n",
                                                   "skipped_n", "passes", "verdict", "first", "same_first",
                                                   "most_common", "distinct", "cached_min", "cached_max",
                                                   "prompt_tokens", "warn", "reason")], summary_rows(quality))
        written.append(out)
    return written


# ------------------------------------------------------------------ benchmark prędkości 0.9
EXPORT_SCHEMA_VERSION = 1
RUN_COLS = ("id", "model", "run_no", "started_at", "finished_at", "status", "verdict", "verdict_measure",
            "gen_level", "ttft_level", "gen_tokens_total", "gen_ms_total", "gen_tps", "ttft_wall_median_ms",
            "ttft_wall_max_ms", "pp_tokens_total", "pp_ms_total", "pp_tps", "load_ms", "load_wall_ms",
            "engine_ms_total", "wall_ms_total", "overhead_ratio", "tasks_counted", "tasks_error", "num_ctx",
            "gpu_share", "stabilization", "cooling_seconds", "thresholds_version", "thresholds_json",
            "warnings_json", "baseline_json")
TASK_COLS = ("task_no", "task_id", "category", "started_at", "num_ctx", "num_predict", "prompt_tokens",
             "prompt_cached", "gen_tokens", "load_ms", "prompt_eval_ms", "eval_ms", "total_ms", "ttft_wall_ms",
             "wall_ms", "done_reason", "error", "counted_gen", "counted_pp", "flags_json", "prompt_text",
             "response_text")
LONG_COLS = ("num_ctx", "target_tokens", "reserve_tokens", "prompt_tokens", "prompt_cached", "gen_tokens",
             "free_tokens", "prompt_eval_ms", "eval_ms", "pp_tps", "tg_tps", "ttft_wall_ms", "gpu_share",
             "done_reason", "valid", "invalid_reason", "calibration_json", "error")
TELEMETRY_COLS = ("run_id", "t", "phase", "task_no", "cpu_pct", "cpu_cores_json", "cpu_temp_c", "cpu_power_w",
                  "ram_used_mb", "ram_available_mb", "swap_used_mb", "swap_in_bps", "swap_out_bps",
                  "gpu_util_pct", "gpu_temp_c", "gpu_power_w", "gpu_clock_mhz", "vram_used_mb", "vram_total_mb",
                  "vram_temp_c", "gpu_throttle", "extra_json")


def exact(value) -> str:
    """Wartość bez zaokrąglania (decyzja autora: eksport zachowuje dokładną liczbę); przecinek
    dziesiętny dla Excela z polskimi ustawieniami; brak = pusta komórka (NULL), nie 0."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value).replace(".", t("bench.decimal"))
    return str(value)


def session_folder(session: dict) -> str:
    """benchmark_2026-11-09_1432_a3f9c2e1 – data, godzina i 8 znaków identyfikatora (bez nadpisywania)."""
    stamp = (session.get("started_at") or "")[:16].replace("T", "_").replace(":", "")
    return f"benchmark_{stamp}_{session['id'][:8]}"


def export_session(db, session_id: str, parent: str | Path) -> Path:
    """Folder z werdykt.csv, zadania.csv, dlugi_kontekst.csv, telemetria.csv i manifest.json."""
    import json
    session = db.session(session_id)
    if session is None:
        raise KeyError(session_id)
    folder = Path(parent) / session_folder(session)
    folder.mkdir(parents=True, exist_ok=True)
    runs = db.runs(session_id)
    _write(folder / "werdykt.csv", list(RUN_COLS), ([exact(r.get(c)) for c in RUN_COLS] for r in runs))
    _write(folder / "zadania.csv", ["run_id", "model", "run_no", *TASK_COLS],
           ([r["id"], r["model"], r["run_no"], *(exact(x.get(c)) for c in TASK_COLS)]
            for r in runs for x in db.tasks(r["id"])))
    _write(folder / "dlugi_kontekst.csv", ["run_id", "model", "run_no", *LONG_COLS],
           ([r["id"], r["model"], r["run_no"], *(exact(x.get(c)) for c in LONG_COLS)]
            for r in runs for x in db.long_context(r["id"])))
    _write(folder / "telemetria.csv", list(TELEMETRY_COLS),
           ([exact(s.get(c)) for c in TELEMETRY_COLS] for s in db.telemetry(session_id=session_id)))

    def parsed(value):
        try:
            return json.loads(value) if isinstance(value, str) else value
        except ValueError:
            return value
    manifest = {"export_schema_version": EXPORT_SCHEMA_VERSION, "db_schema_version": db.schema_version(),
                "session": {k: parsed(v) if k.endswith("_json") else v for k, v in session.items()},
                "runs": [{"id": r["id"], "model": r["model"], "run_no": r["run_no"], "verdict": r["verdict"],
                          "thresholds_version": r["thresholds_version"],
                          "thresholds": parsed(r["thresholds_json"])} for r in runs],
                "files": ["werdykt.csv", "zadania.csv", "dlugi_kontekst.csv", "telemetria.csv"],
                "note": t("speed.disclaimer")}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def export_archive(db, parent: str | Path, day: str) -> Path:
    """Archiwum sprzed 0.9 (tabele perf i quality, tylko odczyt) do folderu, bez zaokrągleń."""
    folder = Path(parent) / f"archiwum_przed_0.9_{day}"
    folder.mkdir(parents=True, exist_ok=True)
    for table in ("perf", "quality"):
        rows = db.archive(table)
        cols = list(rows[0]) if rows else []
        _write(folder / f"{table}.csv", cols, ([exact(r[c]) for c in cols] for r in rows))
    return folder


# ------------------------------------------------------------------ moduł jakości 0.10
Q_RUN_COLS = ("id", "model", "suite", "suite_version", "lang", "started_at", "finished_at", "status", "total",
              "evaluated", "correct", "format_noncompliant", "not_evaluated", "manual", "by_category_json",
              "by_level_json")
Q_TASK_COLS = ("task_no", "task_id", "category", "level", "correct", "format_compliant", "reason", "manual",
               "differing", "expected", "value", "prompt_text", "answer", "attempts_json")


def export_quality_session(db, session_id: str, parent: str | Path) -> Path:
    """Folder jakosc_RRRR-MM-DD_GGMM_<id8>: wyniki.csv, zadania.csv (z odpowiedziami), manifest.json."""
    import json
    session = next((s for s in db.quality_sessions() if s["id"] == session_id), None)
    if session is None:
        raise KeyError(session_id)
    stamp = (session.get("started_at") or "")[:16].replace("T", "_").replace(":", "")
    folder = Path(parent) / f"jakosc_{stamp}_{session_id[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    runs = db.quality_runs(session_id)
    _write(folder / "wyniki.csv", list(Q_RUN_COLS), ([exact(r.get(c)) for c in Q_RUN_COLS] for r in runs))
    _write(folder / "zadania.csv", ["run_id", "model", "suite", "lang", *Q_TASK_COLS],
           ([r["id"], r["model"], r["suite"], r["lang"], *(exact(x.get(c)) for c in Q_TASK_COLS)]
            for r in runs for x in db.quality_tasks(r["id"])))

    def parsed(value):
        try:
            return json.loads(value) if isinstance(value, str) else value
        except ValueError:
            return value
    manifest = {"export_schema_version": EXPORT_SCHEMA_VERSION, "kind": "quality", "db_schema_version": db.schema_version(),
                "session": {k: parsed(v) if k.endswith("_json") else v for k, v in session.items()},
                "files": ["wyniki.csv", "zadania.csv"]}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


D_TASK_COLS = ("task_no", "task_id", "label", "steps", "min_margin", "min_pos", "min_pair_json", "fragile_steps",
               "tie_steps", "first_pos", "first_pair_json", "first_context", "scope", "answer_steps", "after_fragile",
               "after_tie", "reason", "answer", "steps_json")
BOS_COLS = ("model", "digest", "checked_at", "status", "reason", "expects", "add_bos", "gguf_template_has",
            "gguf_template_source", "bos_text", "template_has", "without_count", "with_text_count", "effective",
            "runtime_version", "app_version")


def export_diag_session(db, session_id: str, parent: str | Path) -> Path:
    """Folder diagnostyka_RRRR-MM-DD_GGMM_<id8>: kruchosc.csv (zadania z odstępami i odpowiedziami),
    token_poczatku.csv (sprawdzenia tego modelu), manifest.json (ustawienia przebiegu, progi)."""
    import json
    session = next((s for s in db.diag_sessions() if s["id"] == session_id), None)
    if session is None:
        raise KeyError(session_id)
    stamp = (session.get("started_at") or "")[:16].replace("T", "_").replace(":", "")
    folder = Path(parent) / f"diagnostyka_{stamp}_{session_id[:8]}"
    folder.mkdir(parents=True, exist_ok=True)

    def cell(task, col):
        value = task.get(col)
        return json.dumps(value, ensure_ascii=False) if col.endswith("_json") and value is not None else exact(value)
    _write(folder / "kruchosc.csv", ["model", "suite", "suite_version", "lang", *D_TASK_COLS],
           ([session["model"], session["suite"], session["suite_version"], session["lang"],
             *(cell(x, c) for c in D_TASK_COLS)] for x in db.diag_tasks(session_id)))
    checks = db._select("SELECT * FROM bos_checks WHERE model = ? ORDER BY id", [session["model"]])
    _write(folder / "token_poczatku.csv", list(BOS_COLS), ([exact(r.get(c)) for c in BOS_COLS] for r in checks))
    manifest = {"export_schema_version": EXPORT_SCHEMA_VERSION, "kind": "diagnostics",
                "db_schema_version": db.schema_version(), "session": session,
                "files": ["kruchosc.csv", "token_poczatku.csv"]}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def export_bos_checks(db, path: str | Path) -> Path:
    """Wszystkie sprawdzenia tokenu początku (każde osobno, z datą) do jednego pliku CSV."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    rows = db._select("SELECT * FROM bos_checks ORDER BY model, id")
    _write(path, list(BOS_COLS), ([exact(r.get(c)) for c in BOS_COLS] for r in rows))
    return path
