"""Wynik testów jakości 0.10: ułamki w kategoriach i poziomach, pokrycie, etykieta „w tych testach”.

Zasady autora (10.11.2026):
- w kategoriach ułamki („Fleksja 2/4”), procent tylko przy wyniku całego testu;
- nieocenione zadania (np. kod bez piaskownicy) NIE obniżają wyniku: „12/12 ocenionych · pokrycie
  12/15 (80%) · wynik częściowy”;
- wyniki humanistyczny i ścisły osobno, bez wspólnej oceny; etykieta tylko „w tych testach”:
  różnica ≥ 15 punktów procentowych między humanistycznym a ścisłym PO POLSKU → „lepiej wypadł
  w zadaniach tekstowych / ścisłych”, mniejsza → „podobnie w obu”, oba < 50% → „słabo w obu”;
- przy trzech próbach zadanie zaliczone przy 2 z 3 (pomiar stabilności, nie 45 zadań).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

LABEL_GAP = 15.0          # punkty procentowe
WEAK = 50.0


@dataclass
class Tally:
    total: int = 0
    evaluated: int = 0
    correct: int = 0

    def add(self, correct: Optional[bool]) -> None:
        self.total += 1
        if correct is not None:
            self.evaluated += 1
            self.correct += bool(correct)

    @property
    def percent(self) -> Optional[float]:
        return 100.0 * self.correct / self.evaluated if self.evaluated else None

    @property
    def coverage(self) -> float:
        return 100.0 * self.evaluated / self.total if self.total else 0.0

    @property
    def partial(self) -> bool:
        return self.evaluated < self.total

    def as_dict(self) -> dict:
        return {"total": self.total, "evaluated": self.evaluated, "correct": self.correct}


@dataclass
class Summary:
    overall: Tally = field(default_factory=Tally)
    by_category: dict = field(default_factory=dict)
    by_level: dict = field(default_factory=dict)
    format_noncompliant: int = 0
    manual: int = 0


def summarize(tasks: list[dict]) -> Summary:
    """tasks: {category, level, correct (True/False/None), format_compliant, manual}."""
    s = Summary()
    for t in tasks:
        s.overall.add(t.get("correct"))
        s.by_category.setdefault(t.get("category"), Tally()).add(t.get("correct"))
        s.by_level.setdefault(t.get("level"), Tally()).add(t.get("correct"))
        if t.get("correct") and t.get("format_compliant") is False:
            s.format_noncompliant += 1
        s.manual += bool(t.get("manual"))
    return s


def majority(attempts: list[Optional[bool]]) -> Optional[bool]:
    """Trzy próby: zaliczone przy 2 z 3 ocenionych; jedna próba – jej wynik; brak ocen – None."""
    rated = [a for a in attempts if a is not None]
    if not rated:
        return None
    if len(attempts) == 1:
        return rated[0]
    return sum(rated) >= 2


@dataclass
class Label:
    kind: str            # text / science / similar / weak / none
    humanities: Optional[float]
    science: Optional[float]
    partial: bool


def label(humanities: Optional[Tally], science_pl: Optional[Tally]) -> Label:
    """Porównanie humanistycznego ze ścisłym po polsku (ten sam język po obu stronach)."""
    if not humanities or not science_pl or humanities.percent is None or science_pl.percent is None:
        return Label("none", humanities.percent if humanities else None,
                     science_pl.percent if science_pl else None, True)
    h, s = humanities.percent, science_pl.percent
    partial = humanities.partial or science_pl.partial
    if h < WEAK and s < WEAK:
        kind = "weak"
    elif h - s >= LABEL_GAP:
        kind = "text"
    elif s - h >= LABEL_GAP:
        kind = "science"
    else:
        kind = "similar"
    return Label(kind, h, s, partial)


def estimate_seconds(tasks: int, attempts: int, load_s: float = 3.0, tps: float = 20.0,
                     avg_tokens: float = 60.0, python_tasks: int = 0) -> float:
    """Prognoza czasu: każde zadanie od zera (zwolnienie + wczytanie), odpowiedź, piaskownica."""
    return attempts * (tasks * (load_s + 0.5 + avg_tokens / max(tps, 1.0)) + python_tasks * 1.0)
