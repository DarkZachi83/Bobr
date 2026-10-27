"""Czy runtime (Ollama, LM Studio, llama.cpp) użyje tej karty? (SPEC §10)

Pasek zawsze liczy się z VRAM; tu powstają tylko ostrzeżenia (klucze i18n).
Progi według dokumentacji Ollamy (docs.ollama.com/gpu, stan 2026-10):
  * NVIDIA: compute capability ≥ 5.0 i sterownik ≥ 550; dla 5.0–6.2 sterownik ≥ 570.
  * CUDA 13 porzuciła Maxwella, Pascala i Voltę (< 7.5): działają tylko kompilacje
    z CUDA 12 albo Vulkan – mogą zniknąć w przyszłych wersjach.
  * AMD: oficjalna lista kart ROCm jest krótka i różna dla Linuksa i Windowsa, reszta
    zwykle działa przez Vulkan; Intel – tylko Vulkan / SYCL. Listy kart nie trzymamy,
    bo starzeje się szybciej niż katalog modeli.
"""
from __future__ import annotations

from typing import Optional

from aibobr.domain.enums import GpuVendor
from aibobr.domain.models import GpuInfo

CC_MIN = (5, 0)
CC_CUDA13 = (7, 5)
CC_NEEDS_NEWER_DRIVER = (6, 2)
DRIVER_MIN = 550
DRIVER_MIN_OLD_CARDS = 570


def _driver_major(version: Optional[str]) -> Optional[int]:
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError):
        return None


def gpu_warnings(gpu: GpuInfo) -> list[str]:
    if gpu.vendor == GpuVendor.AMD:
        return ["warn.amd_runtime"]
    if gpu.vendor == GpuVendor.INTEL:
        return ["warn.intel_runtime"]
    if gpu.vendor != GpuVendor.NVIDIA:
        return []
    out = []
    cc = gpu.compute_capability
    if cc is not None and cc < CC_MIN:
        return ["warn.nvidia_unsupported"]
    if cc is not None and cc < CC_CUDA13:
        out.append("warn.nvidia_legacy")
    driver = _driver_major(gpu.driver_version)
    needed = DRIVER_MIN_OLD_CARDS if cc is not None and cc <= CC_NEEDS_NEWER_DRIVER else DRIVER_MIN
    if driver is not None and driver < needed:
        out.append("warn.nvidia_driver_old")
    return out
