"""Gdzie leżą pliki programu (data/, config/, assets/).

Uruchomienie ze źródeł: katalog repozytorium. Spakowany .exe (PyInstaller): katalog,
do którego PyInstaller rozpakowuje pliki (sys._MEIPASS). Zapis zawsze idzie do katalogu
użytkownika (app/config.py), nigdy tutaj – w .exe ten katalog jest tymczasowy.
"""
from __future__ import annotations

import sys
from pathlib import Path


def root() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    return Path(bundle) if bundle else Path(__file__).resolve().parent.parent


def path(*parts: str) -> Path:
    return root().joinpath(*parts)


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))
