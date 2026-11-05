"""Zapamiętana lista „Modele na dysku” (etap 13c, decyzja autora: skan po kliknięciu,
lista zostaje między uruchomieniami). Plik found_models.json obok zapisanych modeli;
uszkodzony albo z innej wersji = pusta lista (wystarczy kliknąć „Szukaj modeli”).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

from aibobr.scan.model import FoundModel, Kind, ScanReport

log = logging.getLogger(__name__)
FORMAT = 1
_FIELDS = {f.name for f in fields(FoundModel)}


def save(path: Path, report: ScanReport, when: str) -> None:
    rows = []
    for m in report.models:
        d = asdict(m)
        d["kind"], d["path"] = m.kind.value, str(m.path)
        rows.append(d)
    data = {"format": FORMAT, "scanned_at": when, "models": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load(path: Path) -> tuple[Optional[ScanReport], Optional[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as e:
        log.warning("cannot read %s: %s", path, e)
        return None, None
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        return None, None
    models = []
    for raw in data.get("models", []):
        try:
            raw = {k: v for k, v in raw.items() if k in _FIELDS}
            raw["kind"], raw["path"] = Kind(raw["kind"]), Path(raw["path"])
            models.append(FoundModel(**raw))
        except (TypeError, ValueError, KeyError):
            continue
    return ScanReport(models=models), data.get("scanned_at")
