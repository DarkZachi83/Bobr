"""Wybór detektora: plik --fake-hw, Linux, Windows albo „nieobsługiwany system”."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from aibobr.detection.base import Detector, PlatformUnsupported
from aibobr.domain.models import HardwareProfile


class _Unsupported:
    def __init__(self, platform: str):
        self.platform = platform

    def detect(self) -> HardwareProfile:
        raise PlatformUnsupported(self.platform)


def create_detector(fake_hw: Optional[Path] = None, platform: str = sys.platform) -> Detector:
    if fake_hw is not None:
        from aibobr.detection.profiles import FakeDetector
        return FakeDetector(fake_hw)
    if platform.startswith("linux"):
        from aibobr.detection.linux import LinuxDetector
        return LinuxDetector()
    if platform.startswith("win"):
        from aibobr.detection.windows import WindowsDetector
        return WindowsDetector()
    # macOS: poza zakresem wersji 1 (SPEC §2, opcja A) – tryb Zaawansowany.
    return _Unsupported(platform)
