"""Pliki testów jakości (data/quality/*.json) – format gotowy pod przyszły edytor testów.

{
  "suite_info": {"id", "version", "kind": "humanities" | "science" | "custom",
                 "name": {"pl": …, "en": …}, "languages": ["pl", "en"]},
  "tasks": [{"id", "category", "level": "easy" | "medium" | "hard", "max_tokens",
             "prompt": {"pl": …, "en": …}, "check": {"type": …, …}}]
}
Rodzaje sprawdzeń – aibobr/quality/checks.py (CHECK_TYPES).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from aibobr import resources
from aibobr.quality.checks import CHECK_TYPES

STANDARD = ("humanistyczny_v1", "scisly_v1")
LEVELS = ("easy", "medium", "hard")


class SuiteError(ValueError):
    pass


def load(suite_id: str, folder: Path | None = None) -> dict:
    base = folder or resources.path("data", "quality")
    data = json.loads((Path(base) / f"{suite_id}.json").read_text(encoding="utf-8"))
    validate(data)
    return data


def standard() -> list[dict]:
    return [load(s) for s in STANDARD]


def validate(data: dict) -> None:
    """Błąd w pliku testu = jasny komunikat, a nie wyjątek w połowie pomiaru."""
    info = data.get("suite_info") or {}
    langs = info.get("languages") or []
    if not info.get("id") or not langs:
        raise SuiteError("suite_info: brak id albo languages")
    seen = set()
    for task in data.get("tasks") or []:
        tid = task.get("id")
        if not tid or tid in seen:
            raise SuiteError(f"zadanie bez id albo powtórzone: {tid!r}")
        seen.add(tid)
        if task.get("level") not in LEVELS:
            raise SuiteError(f"{tid}: poziom {task.get('level')!r} spoza {LEVELS}")
        missing = [lang for lang in langs if not (task.get("prompt") or {}).get(lang)]
        if missing:
            raise SuiteError(f"{tid}: brak treści w języku {missing}")
        kind = (task.get("check") or {}).get("type")
        if kind not in CHECK_TYPES:
            raise SuiteError(f"{tid}: nieznany rodzaj sprawdzenia {kind!r}")
    if not seen:
        raise SuiteError("test bez zadań")


def variants(data: dict) -> Iterable[tuple[str, str]]:
    """(id testu, język) do uruchomienia: humanistyczny_v1/pl, scisly_v1/pl, scisly_v1/en…"""
    for lang in data["suite_info"]["languages"]:
        yield data["suite_info"]["id"], lang
