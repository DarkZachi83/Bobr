"""Warstwa tłumaczeń AI Bobra – przeniesiona z NIUCHACZA (niuchacz/i18n).

Zasady:
  * Teksty żyją w plikach en.json i pl.json obok tego modułu – nie w kodzie.
  * Napis powstaje w chwili wyświetlenia: wyjaśnienia i werdykty przechowują KLUCZE
    i pola (Line), a nie gotowe zdania – przełączenie języka działa od razu.
  * Cofnięcia: nieznany język → domyślny, brakujący klucz → domyślny → sam klucz.
    Błąd w polach formatowania daje surowy tekst i wpis w logu, a nie wyjątek.
  * Liczby formatujemy według języka: „6,1 GB” po polsku, „6.1 GB” po angielsku.

Dodanie języka: skopiuj en.json do np. de.json, przetłumacz wartości, dopisz kod do
LANGUAGES oraz klucz "lang.de" w każdym pliku. tests/test_i18n.py pilnuje kompletności.
"""
from __future__ import annotations

import json
import locale
import logging
from pathlib import Path
from typing import Optional

from aibobr.domain.units import to_gib

DIRECTORY = Path(__file__).resolve().parent
DEFAULT_LANGUAGE = "en"
LANGUAGES = ("en", "pl")

_log = logging.getLogger(__name__)
_dicts: dict[str, dict[str, str]] = {}
_current = DEFAULT_LANGUAGE


def _load(lang: str) -> dict[str, str]:
    if lang not in _dicts:
        try:
            _dicts[lang] = json.loads((DIRECTORY / f"{lang}.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            _dicts[lang] = {}
    return _dicts[lang]


def reload() -> None:
    _dicts.clear()


_NEUTRAL = ("c", "c.utf-8", "c.utf8", "posix", "")


def system_language(env=None) -> str:
    """Język systemu: najpierw zmienne użytkownika (LC_ALL, LC_MESSAGES, LANG – Linux),
    potem ustawienia regionalne (Windows: „Polish_Poland”).

    Neutralne „C” / „POSIX” pomijamy: Python z uv (python-build-standalone) i część
    środowisk wirtualnych zgłasza „C.UTF-8” mimo polskiego systemu (zgłoszenie autora)."""
    import os
    env = os.environ if env is None else env
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        code = (env.get(var) or "").lower()
        if code not in _NEUTRAL:
            break
    else:
        try:
            code = (locale.getlocale()[0] or "").lower()
        except (ValueError, TypeError):
            code = ""
    return "pl" if code.startswith(("pl", "polish")) else DEFAULT_LANGUAGE


def set_language(lang: Optional[str]) -> str:
    global _current
    if not lang:
        lang = system_language()
    _current = lang if lang in LANGUAGES else DEFAULT_LANGUAGE
    return _current


def current_language() -> str:
    return _current


def has_key(key: str) -> bool:
    return key in _load(DEFAULT_LANGUAGE)


def t(key: str, lang: Optional[str] = None, **fields) -> str:
    lang = lang or _current
    text = _load(lang).get(key) or _load(DEFAULT_LANGUAGE).get(key) or key
    if not fields:
        return text
    try:
        return text.format(**fields)
    except (KeyError, IndexError, ValueError) as exc:
        _log.warning("format error for key %r (%s): %s", key, lang, exc)
        return text


# --------------------------------------------------------------------------- liczby

def fmt_number(value: float, decimals: int = 1, lang: Optional[str] = None) -> str:
    text = f"{value:,.{decimals}f}".replace(",", " ")      # twarda spacja tysięcy
    return text.replace(".", ",") if (lang or _current) == "pl" else text


def fmt_gb(value: Optional[int], lang: Optional[str] = None) -> str:
    """Bajty → „6,1 GB” / „6.1 GB”; None → „?”."""
    if value is None:
        return "?"
    return f"{fmt_number(to_gib(value), 1, lang)} GB"


def fmt_context(tokens: int) -> str:
    """8192 → „8k”, 131072 → „128k”, 12000 → „12000”."""
    return f"{tokens // 1024}k" if tokens >= 1024 and tokens % 1024 == 0 else str(tokens)


def fmt_params(count: Optional[int], lang: Optional[str] = None) -> str:
    """8030261248 → „8,0B” / „8.0B”; poniżej miliarda w milionach."""
    if not count:
        return "?"
    if count >= 1e9:
        return f"{fmt_number(count / 1e9, 1, lang)}B"
    return f"{fmt_number(count / 1e6, 0, lang)}M"
