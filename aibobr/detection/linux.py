"""Linux: /proc i /sys/class/drm, NVIDIA przez NVML / nvidia-smi (SPEC §10).

Karty wyliczamy z sysfs (każda karta PCI z grafiką ma tam cardN), więc widzimy też
kartę NVIDIA bez działającego sterownika – wtedy z VRAM = None („?” w GUI).
Pamięć AMD: mem_info_vram_total / mem_info_vram_used (sterownik amdgpu).
Parametr `root` pozwala testom podłożyć fałszywe drzewo katalogów.
"""
from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import Callable, Optional

from aibobr.detection.base import (
    APU_VRAM_LIMIT, PCI_VENDORS, CommandRunner, is_intel_discrete,
)
from aibobr.detection.common import ram_info
from aibobr.detection.nvidia import detect_nvidia, normalize_bus
from aibobr.domain.enums import GpuVendor
from aibobr.domain.models import GpuInfo, HardwareProfile

log = logging.getLogger(__name__)
_VENDOR_SHORT = {GpuVendor.NVIDIA: "NVIDIA", GpuVendor.AMD: "AMD", GpuVendor.INTEL: "Intel"}


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_int(path: Path) -> Optional[int]:
    text = _read(path)
    try:
        return int(text, 0) if text is not None else None
    except ValueError:
        return None


class LinuxDetector:
    def __init__(self, root: Path = Path("/"), run: Callable = CommandRunner(),
                 nvidia: Optional[Callable[[], list[GpuInfo]]] = None,
                 ram: Callable = ram_info):
        self.root = Path(root)
        self.run = run
        self.nvidia = nvidia or (lambda: detect_nvidia(self.run))
        self.ram = ram

    # ------------------------------------------------------------------ CPU / RAM
    def cpu_name(self) -> Optional[str]:
        text = _read(self.root / "proc/cpuinfo") or ""
        for key in ("model name", "Model", "Hardware"):      # x86, potem ARM
            m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.M)
            if m:
                return re.sub(r"\s+", " ", m.group(1)).strip()
        return None

    def ram_from_proc(self) -> tuple[Optional[int], Optional[int]]:
        text = _read(self.root / "proc/meminfo") or ""
        values = {k: int(v) * 1024 for k, v in re.findall(r"^(\w+):\s+(\d+) kB", text, re.M)}
        total, avail = values.get("MemTotal"), values.get("MemAvailable")
        return total, (total - avail if total and avail is not None else None)

    # ------------------------------------------------------------------ karty
    def lspci_name(self, bus: Optional[str], vendor: GpuVendor) -> Optional[str]:
        if not bus:
            return None
        out = self.run(["lspci", "-mm", "-s", bus])
        if not out:
            return None
        try:
            fields = shlex.split(out.splitlines()[0])
        except ValueError:
            return None
        if len(fields) < 4:
            return None
        device = re.sub(r"\s*\[[^\]]*AMD/ATI[^\]]*\]", "", fields[3]).strip()
        return f"{_VENDOR_SHORT.get(vendor, '')} {device}".strip()

    def drm_cards(self) -> list[GpuInfo]:
        nvidia = {g.pci_bus: g for g in self.nvidia()}
        cards, seen = [], set()
        drm = self.root / "sys/class/drm"
        for card in sorted(drm.glob("card*"), key=lambda p: p.name):
            if not re.fullmatch(r"card\d+", card.name):
                continue                                   # card0-DP-1 to złącza
            dev = card / "device"
            vendor = PCI_VENDORS.get(_read_int(dev / "vendor") or -1)
            if vendor is None:
                continue                                   # np. ASPEED na serwerze, virtio
            uevent = _read(dev / "uevent") or ""
            m = re.search(r"^PCI_SLOT_NAME=(\S+)", uevent, re.M)
            bus = normalize_bus(m.group(1)) if m else None
            if bus in seen:
                continue
            seen.add(bus)
            device_id = _read_int(dev / "device") or 0

            if vendor == GpuVendor.NVIDIA:
                known = nvidia.pop(bus, None)
                cards.append(known or GpuInfo(
                    name=self.lspci_name(bus, vendor) or "NVIDIA GPU", vendor=vendor,
                    vram_total=None, pci_bus=bus))
                continue

            total = _read_int(dev / "mem_info_vram_total")
            used = _read_int(dev / "mem_info_vram_used")
            if vendor == GpuVendor.AMD:
                integrated = total is not None and total <= APU_VRAM_LIMIT
            else:
                integrated = not is_intel_discrete(device_id)
            name = (_read(dev / "product_name") or self.lspci_name(bus, vendor)
                    or f"{_VENDOR_SHORT[vendor]} GPU [{device_id:04x}]")
            cards.append(GpuInfo(name=name, vendor=vendor, vram_total=total, vram_used=used,
                                 integrated=integrated, pci_bus=bus))
        cards.extend(nvidia.values())                      # NVML widzi, sysfs nie (kontener)
        return cards

    def detect(self) -> HardwareProfile:
        total, used = self.ram()
        if total is None:
            total, used = self.ram_from_proc()
        return HardwareProfile(platform="linux", cpu_name=self.cpu_name(), ram_total=total,
                               ram_used=used, gpus=tuple(self.drm_cards()))
