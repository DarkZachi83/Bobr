"""Karty NVIDIA: najpierw NVML (pakiet nvidia-ml-py, opcjonalny), potem nvidia-smi.

Oba działają tak samo na Linuksie i Windowsie i podają VRAM, compute capability
i wersję sterownika – potrzebne do ostrzeżeń o obsłudze przez runtime (SPEC §10).
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import replace
from typing import Callable, Optional

from aibobr.detection.base import CommandRunner
from aibobr.domain.enums import GpuVendor
from aibobr.domain.models import GpuInfo

log = logging.getLogger(__name__)
MIB = 1024**2


def normalize_bus(bus: str) -> str:
    """'00000000:01:00.0' (NVML) albo '0000:01:00.0' (sysfs) → '0000:01:00.0'."""
    bus = bus.strip().lower()
    domain, _, rest = bus.partition(":")
    return f"{domain[-4:].rjust(4, '0')}:{rest}" if rest else bus


def _text(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def from_nvml() -> Optional[list[GpuInfo]]:
    try:
        import pynvml  # pakiet nvidia-ml-py
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception as e:      # brak sterownika
        log.info("NVML init failed: %s", e)
        return None
    try:
        driver = _text(pynvml.nvmlSystemGetDriverVersion())
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                cc = tuple(pynvml.nvmlDeviceGetCudaComputeCapability(h))
            except Exception:
                cc = None
            gpus.append(GpuInfo(
                name=_text(pynvml.nvmlDeviceGetName(h)), vendor=GpuVendor.NVIDIA,
                vram_total=int(mem.total), vram_used=int(mem.used),
                compute_capability=cc, driver_version=driver,
                pci_bus=normalize_bus(_text(pynvml.nvmlDeviceGetPciInfo(h).busId)),
            ))
        return gpus
    except Exception as e:
        log.warning("NVML query failed: %s", e)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


_FIELDS_FULL = "name,memory.total,memory.used,pci.bus_id,driver_version,compute_cap"
_FIELDS_OLD = "name,memory.total,memory.used,pci.bus_id,driver_version"   # sterowniki < 510


def _mib(value: str) -> Optional[int]:
    try:
        return int(float(value) * MIB)
    except ValueError:          # "[N/A]", "[Not Supported]"
        return None


def from_nvidia_smi(run: Callable = CommandRunner()) -> Optional[list[GpuInfo]]:
    for fields in (_FIELDS_FULL, _FIELDS_OLD):
        out = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        if out is not None:
            break
    else:
        return None
    gpus = []
    for line in out.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 5:
            continue
        cc = None
        if len(cols) >= 6:
            try:
                major, minor = cols[5].split(".")
                cc = (int(major), int(minor))
            except ValueError:
                pass
        gpus.append(GpuInfo(
            name=cols[0], vendor=GpuVendor.NVIDIA, vram_total=_mib(cols[1]),
            vram_used=_mib(cols[2]), pci_bus=normalize_bus(cols[3]),
            driver_version=cols[4], compute_capability=cc,
        ))
    return gpus


def _watts(text: str) -> Optional[float]:
    m = re.search(r"([\d.]+)\s*W", text)
    return float(m.group(1)) if m else None


def parse_power(text: Optional[str]) -> dict[str, tuple[Optional[float], Optional[float]]]:
    """`nvidia-smi -q -d POWER` → {szyna PCI: (bieżący limit W, maksymalny limit W)}.

    W każdym bloku „GPU 00000000:01:00.0” bierzemy PIERWSZE wartości „Current Power Limit”
    i „Max Power Limit” – dalsze sekcje (moduł, pamięć) często mają N/A."""
    out: dict = {}
    bus, limit, top = None, None, None
    for line in (text or "").splitlines():
        head = re.match(r"^GPU\s+([0-9A-Fa-f:.]+)\s*$", line.strip())
        if head:
            if bus:
                out[bus] = (limit, top)
            bus, limit, top = normalize_bus(head.group(1)), None, None
            continue
        if bus and limit is None and "Current Power Limit" in line:
            limit = _watts(line)
        elif bus and top is None and "Max Power Limit" in line:
            top = _watts(line)
    if bus:
        out[bus] = (limit, top)
    return out


def with_power(gpus: list[GpuInfo], run: Callable = CommandRunner()) -> list[GpuInfo]:
    """Dopisuje limity mocy (i stan Dynamic Boost na Linuksie) do kart NVIDIA."""
    if not gpus:
        return gpus
    power = parse_power(run(["nvidia-smi", "-q", "-d", "POWER"]))
    if not power:
        return gpus
    boost = None
    if any((power.get(g.pci_bus) or (None, None))[0] for g in gpus):
        boost = _powerd_state(run)
    out = []
    for g in gpus:
        limit, top = power.get(g.pci_bus, (None, None))
        out.append(replace(g, power_limit_w=limit, power_max_w=top,
                           dynamic_boost=boost if limit and top and limit < top else None))
    return out


def _powerd_state(run: Callable) -> Optional[bool]:
    if not sys.platform.startswith("linux"):
        return None
    out = run(["systemctl", "is-active", "nvidia-powerd"])
    # is-active zwraca kod ≠ 0 dla „inactive” – CommandRunner daje wtedy None; odróżniamy
    # „usługa nieaktywna” od „brak systemctl” drugim pytaniem.
    if out is not None:
        return out.strip() == "active"
    known = run(["systemctl", "list-unit-files", "nvidia-powerd.service", "--no-legend"])
    return False if known and "nvidia-powerd" in known else None


def detect_nvidia(run: Callable = CommandRunner(), use_nvml: bool = True) -> list[GpuInfo]:
    """Lista kart NVIDIA; pusta, gdy brak sterownika (wtedy karta i tak pojawi się
    w wykrywaniu systemowym, ale bez VRAM → „?” w GUI)."""
    return with_power((from_nvml() if use_nvml else None) or from_nvidia_smi(run) or [], run)
