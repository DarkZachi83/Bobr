"""Ustawienia: wartości domyślne w repozytorium, zmiany użytkownika poza nim.

config/settings.json w repozytorium opisuje dostępne opcje i ich wartości domyślne.
Zmiany użytkownika (w tym token Hugging Face!) zapisujemy w katalogu użytkownika,
żeby nigdy nie trafiły do gita:
  Linux:   $XDG_CONFIG_HOME/ai-bobr/settings.json  (zwykle ~/.config/ai-bobr/)
  Windows: %APPDATA%\\AI-Bobr\\settings.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from aibobr.domain.enums import KvCacheType
from aibobr.domain.models import RunSettings
from aibobr.domain.units import gib

log = logging.getLogger(__name__)

DEFAULT_SETTINGS: dict[str, Any] = {
    "language": "",                 # "" = język systemu
    "default_context": 8192,        # tryb Automatyczny (SPEC §3)
    "kv_cache_type": "f16",
    "vram_reserve_gb": 1.0,         # na każdą kartę
    "ram_reserve_gb": {"windows": 4.0, "linux": 2.0, "other": 2.0},
    "margin_ratio": 0.10,           # ostrzeżenie o małym zapasie
    "margin_min_gb": 1.0,
    "overhead_gb": 0.3,             # zmierzone w etapie 8 (SPEC §5)
    "hf_token": "",
    "ollama_host": "http://127.0.0.1:11434",   # tryb Zaawansowany: modele z Ollamy
}


def _app_dir(env: str, windows_sub: str, xdg: str, fallback: str) -> Path:
    if sys.platform.startswith("win"):
        return Path(os.environ.get(env) or Path.home()) / "AI-Bobr" / windows_sub
    return Path(os.environ.get(xdg) or Path.home() / fallback) / "ai-bobr"


def user_config_dir() -> Path:
    return _app_dir("APPDATA", "", "XDG_CONFIG_HOME", ".config")


def user_cache_dir() -> Path:
    return _app_dir("LOCALAPPDATA", "cache", "XDG_CACHE_HOME", ".cache")


class ConfigStore:
    def __init__(self, defaults_dir: Path, user_dir: Path | None = None):
        self.defaults_file = Path(defaults_dir) / "settings.json"
        self.user_file = (Path(user_dir) if user_dir else user_config_dir()) / "settings.json"

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            log.warning("cannot read %s: %s", path, e)
            return {}

    def load(self) -> dict:
        settings = dict(DEFAULT_SETTINGS)
        for source in (self._read(self.defaults_file), self._read(self.user_file)):
            settings.update({k: v for k, v in source.items() if not k.startswith("_")})
        return settings

    def save_user(self, changes: dict) -> None:
        """Zapisuje tylko zmiany użytkownika (scalone z wcześniejszymi)."""
        data = self._read(self.user_file)
        data.update(changes)
        self.user_file.parent.mkdir(parents=True, exist_ok=True)
        self.user_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def platform_key(platform: str) -> str:
    return "windows" if platform.startswith("win") else "linux" if platform.startswith("linux") else "other"


def run_settings(settings: dict, platform: str) -> RunSettings:
    """Ustawienia z pliku → RunSettings dla danego systemu (rezerwa RAM zależy od systemu)."""
    reserve = settings.get("ram_reserve_gb", 2.0)
    if isinstance(reserve, dict):
        reserve = reserve.get(platform_key(platform), reserve.get("other", 2.0))
    try:
        kv = KvCacheType(str(settings.get("kv_cache_type", "f16")).lower())
    except ValueError:
        kv = KvCacheType.F16
    return RunSettings(
        context_tokens=int(settings.get("default_context", 8192)),
        kv_cache_type=kv,
        vram_reserve=gib(float(settings.get("vram_reserve_gb", 1.0))),
        ram_reserve=gib(float(reserve)),
        margin_ratio=float(settings.get("margin_ratio", 0.10)),
        margin_min=gib(float(settings.get("margin_min_gb", 1.0))),
        overhead=gib(float(settings.get("overhead_gb", 0.5))),
    )
