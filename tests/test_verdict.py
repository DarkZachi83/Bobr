import unittest
from dataclasses import replace

from aibobr.calculator.verdict import IncompleteProfileError, evaluate
from aibobr.domain.enums import FitLevel, GpuVendor
from aibobr.domain.models import RunSettings
from aibobr.domain.units import gib, to_gib
from tests.fixtures import (
    DESKTOP_3060, LLAMA_31_8B, LLAMA_33_70B, QUANTS, QWEN3_30B_A3B, QWEN_25_32B, gpu, pc,
)

AUTO = RunSettings()  # tryb Automatyczny: kontekst 8k, f16, rezerwy domyślne


def run(model, hw, settings=AUTO):
    return evaluate(model, hw, settings, QUANTS)


class MockupCasesTest(unittest.TestCase):
    """Trzy przypadki z makiety paska (karta 12 GB, RAM 32 GB)."""

    def test_8b_fits_entirely_in_vram(self):
        v = run(LLAMA_31_8B, DESKTOP_3060)
        self.assertEqual(v.level, FitLevel.GREEN)
        self.assertEqual(v.in_ram, 0)
        self.assertEqual(v.vram_fraction, 1.0)
        self.assertFalse(v.low_margin)

    def test_32b_spills_into_ram(self):
        v = run(QWEN_25_32B, DESKTOP_3060)
        self.assertEqual(v.level, FitLevel.YELLOW)
        self.assertEqual(v.in_vram, gib(11))
        self.assertAlmostEqual(v.vram_fraction, 0.53, places=2)
        self.assertEqual(v.shortfall, 0)

    def test_70b_does_not_fit(self):
        v = run(LLAMA_33_70B, DESKTOP_3060)
        self.assertEqual(v.level, FitLevel.RED)
        self.assertIsNone(v.margin)
        self.assertAlmostEqual(to_gib(v.shortfall), 1.4, places=1)

    def test_70b_fits_slowly_with_more_ram(self):
        self.assertEqual(run(LLAMA_33_70B, pc(ram_gb=64, gpus=[gpu()])).level, FitLevel.YELLOW)


class CpuOnlyTest(unittest.TestCase):
    def test_no_gpu_is_never_green(self):
        v = run(LLAMA_31_8B, pc(ram_gb=32))
        self.assertEqual(v.level, FitLevel.YELLOW)
        self.assertTrue(v.capacity.cpu_only)
        self.assertEqual(v.in_vram, 0)

    def test_integrated_graphics_count_as_cpu_only(self):
        igpu = gpu("Intel UHD Graphics 630", GpuVendor.INTEL, vram_gb=16, integrated=True)
        v = run(LLAMA_31_8B, pc(ram_gb=32, gpus=[igpu]))
        self.assertTrue(v.capacity.cpu_only)
        self.assertEqual(v.capacity.vram_usable, 0)

    def test_laptop_ignores_igpu_next_to_discrete_card(self):
        igpu = gpu("Intel UHD", GpuVendor.INTEL, vram_gb=8, integrated=True)
        v = run(LLAMA_31_8B, pc(gpus=[igpu, gpu(vram_gb=8)]))
        self.assertEqual(v.capacity.vram_usable, gib(7))
        self.assertEqual(v.warnings, ())


class MultiGpuTest(unittest.TestCase):
    def test_same_vendor_cards_add_up_with_reserve_per_card(self):
        v = run(QWEN_25_32B, pc(gpus=[gpu(), gpu()]))
        self.assertEqual(v.capacity.vram_usable, gib(22))
        self.assertEqual(v.level, FitLevel.GREEN)

    def test_mixed_vendors_use_bigger_group_and_warn(self):
        amd = gpu("AMD Radeon RX 7900 XTX", GpuVendor.AMD, vram_gb=24)
        v = run(LLAMA_31_8B, pc(gpus=[gpu(), amd]))
        self.assertEqual(v.capacity.vram_usable, gib(23))
        self.assertEqual(v.capacity.gpus_used, (amd,))
        self.assertIn("warn.mixed_gpu_vendors", v.warnings)
        self.assertIn("warn.amd_runtime", v.warnings)

    def test_deselected_card_is_ignored(self):
        v = run(LLAMA_31_8B, pc(gpus=[gpu(), gpu(selected=False)]))
        self.assertEqual(v.capacity.vram_usable, gib(11))


class RuntimeSupportTest(unittest.TestCase):
    """SPEC §10: karta liczy się z VRAM, ale dostaje ostrzeżenie."""

    def warnings(self, **kw):
        v = run(LLAMA_31_8B, pc(gpus=[gpu(**kw)]))
        self.assertEqual(v.capacity.vram_usable, gib(11))          # VRAM nadal liczony
        return v.warnings

    def test_modern_card_and_driver(self):
        self.assertEqual(self.warnings(compute_capability=(8, 6), driver_version="575.64"), ())

    def test_pascal_needs_cuda12_build_and_newer_driver(self):
        w = self.warnings(compute_capability=(6, 1), driver_version="560.35")
        self.assertEqual(w, ("warn.nvidia_legacy", "warn.nvidia_driver_old"))

    def test_kepler_is_not_supported(self):
        self.assertEqual(self.warnings(compute_capability=(3, 5)), ("warn.nvidia_unsupported",))

    def test_old_driver_on_modern_card(self):
        self.assertEqual(self.warnings(compute_capability=(8, 9), driver_version="535.1"),
                         ("warn.nvidia_driver_old",))

    def test_unknown_details_give_no_warning(self):
        self.assertEqual(self.warnings(), ())

    def test_intel_arc(self):
        self.assertEqual(self.warnings(name="Intel Arc B580", vendor=GpuVendor.INTEL),
                         ("warn.intel_runtime",))


class IncompleteProfileTest(unittest.TestCase):
    """SPEC §10: nieodczytana wartość to nie 0 GB – werdyktu nie wydajemy."""

    def test_unknown_vram(self):
        with self.assertRaises(IncompleteProfileError) as ctx:
            run(LLAMA_31_8B, pc(gpus=[gpu(vram_gb=None)]))
        self.assertEqual(ctx.exception.missing, ("gpu_vram:NVIDIA GeForce RTX 3060",))

    def test_unknown_ram(self):
        with self.assertRaises(IncompleteProfileError) as ctx:
            run(LLAMA_31_8B, pc(ram_gb=None, gpus=[gpu()]))
        self.assertEqual(ctx.exception.missing, ("ram_total",))

    def test_unknown_vram_of_deselected_card_does_not_block(self):
        v = run(LLAMA_31_8B, pc(gpus=[gpu(), gpu(vram_gb=None, selected=False)]))
        self.assertEqual(v.level, FitLevel.GREEN)


class MarginTest(unittest.TestCase):
    def test_low_margin_in_green(self):
        # 8B na karcie 7,5 GB: 6,5 GB użytkowe, model ~5,9 GB → zapas < 1 GB
        v = run(LLAMA_31_8B, pc(gpus=[gpu(vram_gb=7.5)]))
        self.assertEqual(v.level, FitLevel.GREEN)
        self.assertTrue(v.low_margin)

    def test_longer_context_pushes_model_out_of_vram(self):
        v = run(LLAMA_31_8B, pc(gpus=[gpu(vram_gb=8)]), replace(AUTO, context_tokens=32768))
        self.assertEqual(v.level, FitLevel.YELLOW)

    def test_margin_ratio_is_configurable(self):
        # 8B na karcie 24 GB (23 użytkowe): zapas ~16,9 GB.
        hw = pc(gpus=[gpu(vram_gb=24)])
        self.assertFalse(run(LLAMA_31_8B, hw, replace(AUTO, margin_ratio=0.15)).low_margin)
        self.assertTrue(run(LLAMA_31_8B, hw, replace(AUTO, margin_ratio=0.75)).low_margin)

    def test_windows_ram_reserve(self):
        linux = run(LLAMA_31_8B, pc(ram_gb=16), AUTO)
        windows = run(LLAMA_31_8B, pc(ram_gb=16), replace(AUTO, ram_reserve=gib(4)))
        self.assertEqual(linux.capacity.ram_usable - windows.capacity.ram_usable, gib(2))


class ModelPropertiesTest(unittest.TestCase):
    def test_context_beyond_model_limit_is_flagged(self):
        self.assertTrue(run(QWEN_25_32B, DESKTOP_3060,
                            replace(AUTO, context_tokens=65536)).context_exceeds_model)
        self.assertFalse(run(QWEN_25_32B, DESKTOP_3060).context_exceeds_model)

    def test_moe_memory_counts_all_parameters(self):
        # SPEC §7: pamięć bez zmian – 30B MoE nie mieści się w 11 GB mimo 3B aktywnych.
        v = run(QWEN3_30B_A3B, DESKTOP_3060)
        self.assertTrue(QWEN3_30B_A3B.is_moe)
        self.assertEqual(v.level, FitLevel.YELLOW)


if __name__ == "__main__":
    unittest.main()
