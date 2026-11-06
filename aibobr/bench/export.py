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
