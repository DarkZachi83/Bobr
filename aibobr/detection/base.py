"""Wspólne elementy detekcji."""
from __future__ import annotations

import logging
import subprocess
from typing import Optional, Protocol, Sequence

from aibobr.domain.enums import GpuVendor
from aibobr.domain.models import HardwareProfile

log = logging.getLogger(__name__)

COMMAND_TIMEOUT = 8.0   # s – zawieszony sterownik nie może zamrozić programu

PCI_VENDORS = {0x10DE: GpuVendor.NVIDIA, 0x1002: GpuVendor.AMD, 0x1022: GpuVendor.AMD,
               0x8086: GpuVendor.INTEL}

# Intel Arc (karty osobne): Alchemist 0x56xx, Battlemage 0xE2xx. Reszta Intela to
# grafika zintegrowana.
def is_intel_discrete(device_id: int) -> bool:
    return 0x5600 <= device_id <= 0x56FF or 0xE200 <= device_id <= 0xE2FF


# AMD APU raportuje jako „VRAM” mały wydzielony kawałek RAM (zwykle 512 MB – 2 GB).
APU_VRAM_LIMIT = 2 * 1024**3


class PlatformUnsupported(Exception):
    """System poza zakresem wersji 1 (macOS, SPEC §2) – GUI otwiera tryb Zaawansowany."""

    def __init__(self, platform: str):
        super().__init__(platform)
        self.platform = platform


class Detector(Protocol):
    def detect(self) -> HardwareProfile: ...


class CommandRunner:
    """Uruchamia zewnętrzne polecenie z limitem czasu. None = nie ma / nie zadziałało.

    W testach podmieniane na słownik gotowych odpowiedzi."""

    def __call__(self, args: Sequence[str], timeout: float = COMMAND_TIMEOUT) -> Optional[str]:
        try:
            done = subprocess.run(
                list(args), capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as e:
            log.info("command %s failed: %s", args[0], e)
            return None
        if done.returncode != 0:
            log.info("command %s exit %s: %s", args[0], done.returncode, done.stderr.strip()[:200])
            return None
        return done.stdout
