"""Profil sprzętu jako JSON: --fake-hw, testy i zgłaszanie błędów (SPEC §11).

Format (rozmiary w GB binarnych, null = „nie udało się odczytać”):

    {
      "platform": "windows",
      "cpu_name": "AMD Ryzen 7 7800X3D",
      "ram_total_gb": 32, "ram_used_gb": 9.5, "ram_description": "DDR5-6000",
      "gpus": [
        {"name": "AMD Radeon RX 7900 XTX", "vendor": "amd", "vram_total_gb": 24,
         "vram_used_gb": 1.1, "integrated": false,
         "compute_capability": null, "driver_version": null}
      ]
    }

Ten sam format zapisuje profile_to_dict(), więc użytkownik może wysłać swój skan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from aibobr.domain.enums import GpuVendor
from aibobr.domain.models import GpuInfo, HardwareProfile
from aibobr.domain.units import gib, to_gib


class ProfileError(ValueError):
    pass


def _gb(value: Any, field: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ProfileError(f"{field}: expected a number of GB or null")
    return gib(value)


def _round_gb(value: Optional[int]) -> Optional[float]:
    return None if value is None else round(to_gib(value), 2)


def profile_from_dict(data: dict) -> HardwareProfile:
    if not isinstance(data, dict):
        raise ProfileError("profile must be a JSON object")
    gpus = []
    for i, g in enumerate(data.get("gpus") or []):
        try:
            vendor = GpuVendor(str(g.get("vendor", "other")).lower())
        except ValueError:
            vendor = GpuVendor.OTHER
        cc = g.get("compute_capability")
        if isinstance(cc, str):
            major, _, minor = cc.partition(".")
            cc = (int(major), int(minor or 0))
        elif isinstance(cc, list):
            cc = tuple(cc)
        gpus.append(GpuInfo(
            name=str(g.get("name") or f"GPU {i}"), vendor=vendor,
            vram_total=_gb(g.get("vram_total_gb"), f"gpus[{i}].vram_total_gb"),
            vram_used=_gb(g.get("vram_used_gb"), f"gpus[{i}].vram_used_gb"),
            integrated=bool(g.get("integrated", False)),
            selected=bool(g.get("selected", True)),
            compute_capability=cc, driver_version=g.get("driver_version"),
            pci_bus=g.get("pci_bus"),
        ))
    return HardwareProfile(
        platform=str(data.get("platform", "unknown")),
        cpu_name=data.get("cpu_name"),
        ram_total=_gb(data.get("ram_total_gb"), "ram_total_gb"),
        ram_used=_gb(data.get("ram_used_gb"), "ram_used_gb"),
        ram_description=data.get("ram_description"),
        gpus=tuple(gpus),
    )


def profile_to_dict(p: HardwareProfile) -> dict:
    return {
        "platform": p.platform, "cpu_name": p.cpu_name,
        "ram_total_gb": _round_gb(p.ram_total), "ram_used_gb": _round_gb(p.ram_used),
        "ram_description": p.ram_description,
        "gpus": [{
            "name": g.name, "vendor": g.vendor.value,
            "vram_total_gb": _round_gb(g.vram_total), "vram_used_gb": _round_gb(g.vram_used),
            "integrated": g.integrated, "selected": g.selected,
            "compute_capability": ".".join(map(str, g.compute_capability))
            if g.compute_capability else None,
            "driver_version": g.driver_version, "pci_bus": g.pci_bus,
        } for g in p.gpus],
    }


def load_profile(path: Path) -> HardwareProfile:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ProfileError(f"{path}: {e}") from e
    return profile_from_dict(data)


class FakeDetector:
    """Zamiast skanu zwraca profil z pliku (--fake-hw)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def detect(self) -> HardwareProfile:
        return load_profile(self.path)
