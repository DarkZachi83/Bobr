"""Wspólne dane testowe: prawdziwe modele i przykładowe komputery.

Architektury według config.json modeli na Hugging Face; rozmiary plików Q4_K_M
w przybliżeniu jak w popularnych repozytoriach GGUF (wystarczy do testów logiki).
"""
from __future__ import annotations

from pathlib import Path

from aibobr.calculator.quants import QuantTable
from aibobr.domain.enums import GpuVendor, ModelSource
from aibobr.domain.models import GpuInfo, HardwareProfile, ModelSpec
from aibobr.domain.units import gib

ROOT = Path(__file__).resolve().parent.parent
QUANTS = QuantTable.load(ROOT / "data" / "quants.json")

LLAMA_31_8B = ModelSpec(
    name="Llama 3.1 8B Instruct", source=ModelSource.HUGGINGFACE, quant="Q4_K_M",
    params_total=8_030_000_000, weights_bytes=4_920_000_000,
    n_layers=32, n_kv_heads=8, head_dim=128, max_context=131_072,
)
QWEN_25_32B = ModelSpec(
    name="Qwen 2.5 32B Instruct", source=ModelSource.HUGGINGFACE, quant="Q4_K_M",
    params_total=32_760_000_000, weights_bytes=19_850_000_000,
    n_layers=64, n_kv_heads=8, head_dim=128, max_context=32_768,
)
LLAMA_33_70B = ModelSpec(
    name="Llama 3.3 70B Instruct", source=ModelSource.HUGGINGFACE, quant="Q4_K_M",
    params_total=70_550_000_000, weights_bytes=42_520_000_000,
    n_layers=80, n_kv_heads=8, head_dim=128, max_context=131_072,
)
QWEN3_30B_A3B = ModelSpec(
    name="Qwen3 30B A3B", source=ModelSource.HUGGINGFACE, quant="Q4_K_M",
    params_total=30_530_000_000, weights_bytes=18_560_000_000,
    n_layers=48, n_kv_heads=4, head_dim=128, max_context=40_960,
    n_experts=128, n_experts_used=8, params_active=3_300_000_000,
)


def gpu(name="NVIDIA GeForce RTX 3060", vendor=GpuVendor.NVIDIA, vram_gb=12.0, **kw) -> GpuInfo:
    return GpuInfo(name=name, vendor=vendor,
                   vram_total=None if vram_gb is None else gib(vram_gb), **kw)


def pc(ram_gb=32.0, gpus=(), platform="linux") -> HardwareProfile:
    return HardwareProfile(platform=platform, cpu_name="AMD Ryzen 5 5600X",
                           ram_total=None if ram_gb is None else gib(ram_gb), gpus=tuple(gpus))


# Komputer z makiety: karta 12 GB (11 użytkowe), RAM 32 GB (30 użytkowe na Linuksie).
DESKTOP_3060 = pc(ram_gb=32, gpus=[gpu()])
