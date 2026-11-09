"""Historia benchmarku 0.9 (decyzja autora 9.11.2026, pytanie 9a) – dane bez tkintera.

Jeden wiersz = jeden model w jednej sesji: przebiegi, werdykt z dnia testu (i według obecnych
progów, gdy inny), zakres generowania (pomiar rozrzutu), TTFT, uwagi. Werdykt według obecnych progów
liczymy z zapisanych surowych sum, nie z zaokrąglonych liczb z ekranu.
"""
from __future__ import annotations

import json
from typing import Optional

from aibobr.bench import speed


def _json(value, default=None):
    if not isinstance(value, str):
        return value if value is not None else default
    try:
        return json.loads(value)
    except ValueError:
        return default


def rows(db, model: Optional[str] = None, current: Optional[dict] = None) -> list[dict]:
    """Sesje od najnowszej; current – obecne progi (do „według obecnych progów”)."""
    out = []
    for session in db.sessions(model):
        runs = db.runs(session["id"])
        order = _json(session.get("models_json"), []) or []        # kolejność jak w teście
        models = []
        for r in sorted(runs, key=lambda r: (order.index(r["model"]) if r["model"] in order else len(order),
                                             r["started_at"] or "", r["run_no"])):
            if r["model"] not in models and (model is None or r["model"] == model):
                models.append(r["model"])
        for name in models:
            mine = [r for r in runs if r["model"] == name]
            then = [r["verdict"] if r["status"] == "finished" else r["status"] for r in mine]
            now = [speed.recompute(r, current) if r["status"] == "finished" else r["status"] for r in mine] \
                if current else then
            warnings = sorted({w for r in mine for w in (_json(r["warnings_json"], []) or [])})
            ttft = [r["ttft_wall_median_ms"] for r in mine if r["ttft_wall_median_ms"] is not None]
            out.append({"session_id": session["id"], "started_at": session["started_at"],
                        "status": session["status"], "model": name, "runs": len(mine),
                        "runs_per_model": session["runs_per_model"], "verdicts": then, "verdicts_now": now,
                        "changed": now != then, "spread": speed.spread(mine),
                        "ttft": (min(ttft), max(ttft)) if ttft else None, "warnings": warnings,
                        "hardware": session["hardware"], "run_ids": [r["id"] for r in mine]})
    return out


def run_details(db, run_id: str) -> Optional[dict]:
    """Przebieg z zadaniami i długim kontekstem – do okna szczegółów (jak tuż po teście)."""
    run = db.run(run_id)
    if run is None:
        return None
    return {**run, "tasks": db.tasks(run_id), "long_context": db.long_context(run_id)}


def models(db) -> list[str]:
    return db.run_models()
