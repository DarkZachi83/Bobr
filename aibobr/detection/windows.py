"""Windows: rejestr + jedno wywołanie PowerShell (SPEC §10).

* VRAM z rejestru: HardwareInformation.qwMemorySize (64-bit). Celowo NIE używamy
  Win32_VideoController.AdapterRAM – jest 32-bitowe i obcina VRAM do 4 GB.
* Lista kart obecnych w systemie z Win32_VideoController (rejestr trzyma też wpisy
  po kartach dawno wyjętych). Przy okazji typ i prędkość RAM z Win32_PhysicalMemory.
* NVIDIA: dokładne dane z NVML / nvidia-smi, jak na Linuksie.
* Zajętość VRAM AMD / Intel na Windowsie pomijamy (liczniki wydajności są powolne
  i trudne do przypisania karcie) – to tylko informacja, werdykt jej nie używa.

Rejestr i polecenia są wstrzykiwane, więc całość da się testować na Linuksie.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional, Protocol

from aibobr.detection.base import (
    APU_VRAM_LIMIT, PCI_VENDORS, CommandRunner, is_intel_discrete,
)
from aibobr.detection.common import ram_info
from aibobr.detection.nvidia import detect_nvidia
from aibobr.domain.models import GpuInfo, HardwareProfile
from aibobr.domain.enums import GpuVendor

log = logging.getLogger(__name__)

CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"

PS_SCRIPT = (
    "$g = Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID; "
    "$m = Get-CimInstance Win32_PhysicalMemory | "
    "Select-Object SMBIOSMemoryType,Speed,ConfiguredClockSpeed; "
    "@{gpu=@($g); mem=@($m)} | ConvertTo-Json -Compress -Depth 3"
)
# SMBIOS: typ pamięci (Win32_PhysicalMemory.SMBIOSMemoryType)
MEMORY_TYPES = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 27: "LPDDR", 28: "LPDDR2",
                29: "LPDDR3", 30: "LPDDR4", 34: "DDR5", 35: "LPDDR5"}


class Registry(Protocol):
    def subkeys(self, path: str) -> list[str]: ...
    def values(self, path: str) -> dict[str, Any]: ...


class WinRegistry:
    """Prawdziwy rejestr (HKEY_LOCAL_MACHINE), tylko odczyt."""

    def subkeys(self, path: str) -> list[str]:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                n = winreg.QueryInfoKey(key)[0]
                return [winreg.EnumKey(key, i) for i in range(n)]
        except OSError:
            return []

    def values(self, path: str) -> dict[str, Any]:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                n = winreg.QueryInfoKey(key)[1]
                return {name: value for name, value, _ in
                        (winreg.EnumValue(key, i) for i in range(n))}
        except OSError:
            return {}


def _pci_ids(text: str) -> tuple[Optional[int], Optional[int]]:
    m = re.search(r"VEN_([0-9A-F]{4}).*?DEV_([0-9A-F]{4})", text or "", re.I)
    return (int(m.group(1), 16), int(m.group(2), 16)) if m else (None, None)


def _memory_size(values: dict) -> Optional[int]:
    qw = values.get("HardwareInformation.qwMemorySize")
    if isinstance(qw, int) and qw > 0:
        return qw
    old = values.get("HardwareInformation.MemorySize")       # starsze sterowniki, max 4 GB
    if isinstance(old, (bytes, bytearray)) and len(old) >= 4:
        old = int.from_bytes(old[:8], "little")
    return old if isinstance(old, int) and 0 < old < 0xFFFFFFFF else None


def ram_description(mem: list[dict]) -> Optional[str]:
    """[{SMBIOSMemoryType: 26, ConfiguredClockSpeed: 3200}, …] → 'DDR4-3200'."""
    if not mem:
        return None
    first = mem[0]
    kind = MEMORY_TYPES.get(first.get("SMBIOSMemoryType"))
    speed = first.get("ConfiguredClockSpeed") or first.get("Speed")
    if kind and speed:
        return f"{kind}-{speed}"
    return kind


class WindowsDetector:
    def __init__(self, registry: Optional[Registry] = None, run: Callable = CommandRunner(),
                 nvidia: Optional[Callable[[], list[GpuInfo]]] = None,
                 ram: Callable = ram_info):
        self.registry = registry or WinRegistry()
        self.run = run
        self.nvidia = nvidia or (lambda: detect_nvidia(self.run))
        self.ram = ram

    def cpu_name(self) -> Optional[str]:
        name = self.registry.values(CPU_KEY).get("ProcessorNameString")
        return re.sub(r"\s+", " ", name).strip() if isinstance(name, str) else None

    def powershell(self) -> dict:
        out = self.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", PS_SCRIPT],
                       timeout=15)
        if not out:
            return {}
        try:
            data = json.loads(out.lstrip("﻿").strip())
        except ValueError:
            log.warning("PowerShell returned non-JSON output")
            return {}
        return data if isinstance(data, dict) else {}

    def registry_adapters(self) -> list[dict]:
        adapters = []
        for sub in self.registry.subkeys(DISPLAY_CLASS):
            if not re.fullmatch(r"\d{4}", sub):
                continue                              # "Properties", "Configuration"
            values = self.registry.values(f"{DISPLAY_CLASS}\\{sub}")
            vendor_id, device_id = _pci_ids(values.get("MatchingDeviceId", ""))
            if vendor_id is None:
                continue                              # karty wirtualne / zdalny pulpit
            adapters.append({"name": str(values.get("DriverDesc", "")).strip(),
                             "vendor_id": vendor_id, "device_id": device_id,
                             "vram": _memory_size(values)})
        return adapters

    def gpus(self, present: Optional[list[dict]]) -> list[GpuInfo]:
        adapters = self.registry_adapters()
        if present is not None:
            # Tylko karty obecne teraz w systemie (wpisy po wyjętych kartach zostają w rejestrze).
            wanted = [p.get("Name", "").strip().lower() for p in present]
            kept = []
            for a in adapters:
                if a["name"].lower() in wanted:
                    wanted.remove(a["name"].lower())
                    kept.append(a)
            # Karta bez sterownika („Microsoft Basic Display Adapter”) nie ma wpisu z VRAM.
            for p in present:
                vendor_id, device_id = _pci_ids(p.get("PNPDeviceID", ""))
                if p.get("Name", "").strip().lower() in wanted and vendor_id is not None:
                    kept.append({"name": p["Name"].strip(), "vendor_id": vendor_id,
                                 "device_id": device_id, "vram": None})
            adapters = kept

        nvidia = list(self.nvidia())
        # Karty NVIDIA: dane z NVML / nvidia-smi zamiast rejestru. Najpierw po nazwie,
        # resztę po kolejności (nazwy w rejestrze bywają inne niż w sterowniku).
        unmatched = []
        for a in adapters:
            if PCI_VENDORS.get(a["vendor_id"]) == GpuVendor.NVIDIA:
                match = next((g for g in nvidia if g.name.lower() == a["name"].lower()), None)
                if match is not None:
                    nvidia.remove(match)
                    a["nvidia"] = match
                else:
                    unmatched.append(a)
        for a in unmatched:
            if nvidia:
                a["nvidia"] = nvidia.pop(0)

        result = []
        for a in adapters:
            vendor = PCI_VENDORS.get(a["vendor_id"])
            if vendor is None:
                continue
            if vendor == GpuVendor.NVIDIA:
                result.append(a.get("nvidia") or GpuInfo(
                    name=a["name"] or "NVIDIA GPU", vendor=vendor, vram_total=a["vram"]))
                continue
            if vendor == GpuVendor.AMD:
                integrated = a["vram"] is not None and a["vram"] <= APU_VRAM_LIMIT
            else:
                integrated = not is_intel_discrete(a["device_id"] or 0)
            result.append(GpuInfo(name=a["name"], vendor=vendor, vram_total=a["vram"],
                                  integrated=integrated))
        result.extend(nvidia)
        return result

    def detect(self) -> HardwareProfile:
        ps = self.powershell()
        present = ps.get("gpu") if isinstance(ps.get("gpu"), list) else None
        mem = ps.get("mem") if isinstance(ps.get("mem"), list) else []
        total, used = self.ram()
        return HardwareProfile(platform="windows", cpu_name=self.cpu_name(), ram_total=total,
                               ram_used=used, ram_description=ram_description(mem),
                               gpus=tuple(self.gpus(present)))
