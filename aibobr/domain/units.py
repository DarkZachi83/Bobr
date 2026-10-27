"""Jednostki pamięci.

Wewnątrz programu wszystko liczymy w bajtach (int). Użytkownikowi pokazujemy „GB”
w znaczeniu binarnym (1 GB = 1024³ B), bo tak producenci opisują VRAM i RAM
(karta „12 GB” ma 12 × 1024³ B) i tak pokazuje to Windows. Rozmiary plików z Hugging
Face są podawane dziesiętnie, ale do nas trafiają jako bajty, więc nic nie mieszamy.
"""
from __future__ import annotations

GIB = 1024**3
MIB = 1024**2


def gib(value: float) -> int:
    """GB (binarne) → bajty."""
    return int(round(value * GIB))


def to_gib(value: int | float) -> float:
    """Bajty → GB (binarne)."""
    return value / GIB
