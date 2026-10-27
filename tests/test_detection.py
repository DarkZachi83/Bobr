import json
import tempfile
import unittest
from pathlib import Path

from aibobr.calculator.verdict import IncompleteProfileError, evaluate
from aibobr.detection.base import PlatformUnsupported
from aibobr.detection.factory import create_detector
from aibobr.detection.linux import LinuxDetector
from aibobr.detection.nvidia import from_nvidia_smi, normalize_bus
from aibobr.detection.profiles import load_profile, profile_from_dict, profile_to_dict
from aibobr.detection.windows import DISPLAY_CLASS, WindowsDetector, ram_description
from aibobr.domain.enums import FitLevel, GpuVendor
from aibobr.domain.models import GpuInfo, RunSettings
from aibobr.domain.units import gib
from tests.fixtures import LLAMA_31_8B, QUANTS

PROFILES = Path(__file__).resolve().parent / "fake_hardware"
GIB = 1024**3


class Commands:
    """Fałszywe polecenia: pierwsze słowo + fragment argumentów → wynik (None = błąd)."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def __call__(self, args, timeout=None):
        self.calls.append(list(args))
        for (cmd, needle), out in self.answers.items():
            if args[0] == cmd and needle in " ".join(args):
                return out
        return None


# --------------------------------------------------------------------------- NVIDIA

class NvidiaSmiTest(unittest.TestCase):
    def test_parse_full_query(self):
        run = Commands({("nvidia-smi", "compute_cap"):
                        "NVIDIA GeForce RTX 3060, 12288, 512, 00000000:01:00.0, 575.64.03, 8.6\n"})
        [g] = from_nvidia_smi(run)
        self.assertEqual((g.name, g.vram_total, g.vram_used), ("NVIDIA GeForce RTX 3060", 12 * GIB, 512 * 1024**2))
        self.assertEqual((g.compute_capability, g.driver_version, g.pci_bus),
                         ((8, 6), "575.64.03", "0000:01:00.0"))

    def test_old_driver_without_compute_cap_field(self):
        run = Commands({("nvidia-smi", "driver_version --format"):
                        "GeForce GTX 1070, 8192, [N/A], 00000000:01:00.0, 470.2\n"})
        [g] = from_nvidia_smi(run)
        self.assertIsNone(g.compute_capability)
        self.assertIsNone(g.vram_used)
        self.assertEqual(len(run.calls), 2)          # najpierw pełne zapytanie, potem stare

    def test_no_driver(self):
        self.assertIsNone(from_nvidia_smi(Commands()))

    def test_bus_normalization(self):
        self.assertEqual(normalize_bus("00000000:0A:00.0"), "0000:0a:00.0")
        self.assertEqual(normalize_bus("0000:0a:00.0"), "0000:0a:00.0")


# --------------------------------------------------------------------------- Linux

class FakeLinux:
    """Fałszywe /proc i /sys/class/drm w katalogu tymczasowym."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "proc").mkdir()
        (self.root / "proc/cpuinfo").write_text(
            "processor\t: 0\nmodel name\t: AMD Ryzen 7 5800X  8-Core Processor\n")
        (self.root / "proc/meminfo").write_text(
            "MemTotal:       32768000 kB\nMemFree: 1 kB\nMemAvailable:   16384000 kB\n")
        (self.root / "sys/class/drm").mkdir(parents=True)

    def card(self, n, vendor, device, bus, driver, vram=None, used=None, product=None):
        dev = self.root / f"sys/class/drm/card{n}/device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(f"0x{vendor:04x}\n")
        (dev / "device").write_text(f"0x{device:04x}\n")
        (dev / "uevent").write_text(f"DRIVER={driver}\nPCI_SLOT_NAME={bus}\n")
        if vram is not None:
            (dev / "mem_info_vram_total").write_text(f"{vram}\n")
        if used is not None:
            (dev / "mem_info_vram_used").write_text(f"{used}\n")
        if product:
            (dev / "product_name").write_text(product + "\n")
        (self.root / f"sys/class/drm/card{n}-DP-1").mkdir()          # złącze – pomijane

    def detector(self, nvidia=(), commands=None, ram=(None, None)):
        return LinuxDetector(self.root, commands or Commands(), nvidia=lambda: list(nvidia),
                             ram=lambda: ram)

    def close(self):
        self.tmp.cleanup()


class LinuxDetectorTest(unittest.TestCase):
    def setUp(self):
        self.fs = FakeLinux()

    def tearDown(self):
        self.fs.close()

    def test_amd_card_and_apu(self):
        self.fs.card(0, 0x1002, 0x744C, "0000:03:00.0", "amdgpu", 24 * GIB, GIB,
                     product="AMD Radeon RX 7900 XTX")
        self.fs.card(1, 0x1002, 0x164E, "0000:10:00.0", "amdgpu", 512 * 1024**2)
        p = self.fs.detector().detect()
        self.assertEqual(p.cpu_name, "AMD Ryzen 7 5800X 8-Core Processor")
        self.assertEqual((p.ram_total, p.ram_used), (32768000 * 1024, 16384000 * 1024))
        dgpu, apu = p.gpus
        self.assertEqual((dgpu.name, dgpu.vram_total, dgpu.vram_used, dgpu.integrated),
                         ("AMD Radeon RX 7900 XTX", 24 * GIB, GIB, False))
        self.assertTrue(apu.integrated)

    def test_name_from_lspci(self):
        self.fs.card(0, 0x1002, 0x73BF, "0000:03:00.0", "amdgpu", 16 * GIB)
        run = Commands({("lspci", "0000:03:00.0"):
                        '03:00.0 "VGA compatible controller" "Advanced Micro Devices, Inc. [AMD/ATI]" '
                        '"Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]" -rc1 "Sapphire" "Nitro+"\n'})
        [g] = self.fs.detector(commands=run).detect().gpus
        self.assertEqual(g.name, "AMD Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]")

    def test_nvidia_card_uses_nvml_data_matched_by_bus(self):
        self.fs.card(0, 0x10DE, 0x2504, "0000:01:00.0", "nvidia")
        smi = GpuInfo("NVIDIA GeForce RTX 3060", GpuVendor.NVIDIA, 12 * GIB,
                      compute_capability=(8, 6), pci_bus="0000:01:00.0")
        [g] = self.fs.detector(nvidia=[smi]).detect().gpus
        self.assertIs(g, smi)

    def test_nvidia_without_driver_has_unknown_vram(self):
        self.fs.card(0, 0x10DE, 0x1B81, "0000:01:00.0", "nouveau")
        [g] = self.fs.detector().detect().gpus
        self.assertEqual((g.vendor, g.vram_total), (GpuVendor.NVIDIA, None))
        with self.assertRaises(IncompleteProfileError):
            evaluate(LLAMA_31_8B, self.fs.detector().detect(), RunSettings(), QUANTS)

    def test_intel_igpu_and_arc(self):
        self.fs.card(0, 0x8086, 0x4680, "0000:00:02.0", "i915")          # UHD 770
        self.fs.card(1, 0x8086, 0xE20B, "0000:03:00.0", "xe")            # Arc B580
        igpu, arc = self.fs.detector().detect().gpus
        self.assertTrue(igpu.integrated)
        self.assertFalse(arc.integrated)
        self.assertIsNone(arc.vram_total)                                 # → „?” w GUI

    def test_server_bmc_and_virtual_cards_are_ignored(self):
        self.fs.card(0, 0x1A03, 0x2000, "0000:02:00.0", "ast")           # ASPEED
        self.fs.card(1, 0x1AF4, 0x1050, "0000:00:01.0", "virtio-pci")
        self.assertEqual(self.fs.detector().detect().gpus, ())

    def test_psutil_values_win_over_proc(self):
        p = self.fs.detector(ram=(64 * GIB, 8 * GIB)).detect()
        self.assertEqual(p.ram_total, 64 * GIB)


# --------------------------------------------------------------------------- Windows

class FakeRegistry:
    def __init__(self, tree):
        self.tree = tree

    def subkeys(self, path):
        prefix = path + "\\"
        return sorted({k[len(prefix):].split("\\")[0] for k in self.tree if k.startswith(prefix)})

    def values(self, path):
        return self.tree.get(path, {})


def adapter(name, ven, dev, qw=None, old=None):
    v = {"DriverDesc": name, "MatchingDeviceId": f"pci\\ven_{ven}&dev_{dev}"}
    if qw is not None:
        v["HardwareInformation.qwMemorySize"] = qw
    if old is not None:
        v["HardwareInformation.MemorySize"] = old
    return v


def powershell(gpus, mem=()):
    return "﻿" + json.dumps({"gpu": gpus, "mem": list(mem)})


class WindowsDetectorTest(unittest.TestCase):
    CPU = {r"HARDWARE\DESCRIPTION\System\CentralProcessor\0":
           {"ProcessorNameString": "AMD Ryzen 7 7800X3D 8-Core Processor           "}}

    def detect(self, adapters, ps, nvidia=()):
        tree = dict(self.CPU)
        for i, a in enumerate(adapters):
            tree[f"{DISPLAY_CLASS}\\{i:04d}"] = a
        tree[f"{DISPLAY_CLASS}\\Properties"] = {}
        return WindowsDetector(FakeRegistry(tree), Commands({("powershell", "Win32"): ps}),
                               nvidia=lambda: list(nvidia), ram=lambda: (32 * GIB, 10 * GIB)).detect()

    def test_amd_vram_above_4gb_from_qwmemorysize(self):
        p = self.detect(
            [adapter("AMD Radeon RX 7900 XTX", "1002", "744C", qw=24 * GIB, old=0xFFFFFFFF),
             adapter("AMD Radeon(TM) Graphics", "1002", "164E", qw=512 * 1024**2)],
            powershell([{"Name": "AMD Radeon RX 7900 XTX", "PNPDeviceID": "PCI\\VEN_1002&DEV_744C"},
                        {"Name": "AMD Radeon(TM) Graphics", "PNPDeviceID": "PCI\\VEN_1002&DEV_164E"}],
                       [{"SMBIOSMemoryType": 34, "Speed": 6000, "ConfiguredClockSpeed": 6000}]))
        self.assertEqual(p.cpu_name, "AMD Ryzen 7 7800X3D 8-Core Processor")
        self.assertEqual(p.ram_description, "DDR5-6000")
        dgpu, apu = p.gpus
        self.assertEqual((dgpu.vram_total, dgpu.integrated), (24 * GIB, False))
        self.assertTrue(apu.integrated)

    def test_removed_card_left_in_registry_is_ignored(self):
        p = self.detect(
            [adapter("NVIDIA GeForce GTX 970", "10DE", "13C2", qw=4 * GIB),
             adapter("AMD Radeon RX 6800", "1002", "73BF", qw=16 * GIB)],
            powershell([{"Name": "AMD Radeon RX 6800", "PNPDeviceID": "PCI\\VEN_1002&DEV_73BF"}]))
        self.assertEqual([g.name for g in p.gpus], ["AMD Radeon RX 6800"])

    def test_without_powershell_all_registry_entries_are_used(self):
        p = self.detect([adapter("AMD Radeon RX 6800", "1002", "73BF", qw=16 * GIB),
                         {"DriverDesc": "Microsoft Remote Display Adapter",
                          "MatchingDeviceId": "swd\\remotedisplayenum"}], None)
        self.assertEqual([g.name for g in p.gpus], ["AMD Radeon RX 6800"])
        self.assertIsNone(p.ram_description)

    def test_nvidia_from_driver_and_card_without_driver(self):
        smi = GpuInfo("NVIDIA GeForce RTX 4070", GpuVendor.NVIDIA, 12 * GIB, compute_capability=(8, 9))
        p = self.detect(
            [adapter("NVIDIA GeForce RTX 4070", "10DE", "2786", qw=12 * GIB),
             adapter("Intel(R) UHD Graphics 770", "8086", "4680", qw=128 * 1024**2)],
            powershell([{"Name": "NVIDIA GeForce RTX 4070", "PNPDeviceID": "PCI\\VEN_10DE&DEV_2786"},
                        {"Name": "Intel(R) UHD Graphics 770", "PNPDeviceID": "PCI\\VEN_8086&DEV_4680"},
                        {"Name": "Microsoft Basic Display Adapter", "PNPDeviceID": "PCI\\VEN_10DE&DEV_1B81"}]),
            nvidia=[smi])
        nv, intel, basic = p.gpus
        self.assertIs(nv, smi)
        self.assertTrue(intel.integrated)
        self.assertEqual((basic.vendor, basic.vram_total), (GpuVendor.NVIDIA, None))

    def test_nvidia_name_mismatch_paired_by_order(self):
        smi = GpuInfo("NVIDIA GeForce RTX 3060 Laptop GPU", GpuVendor.NVIDIA, 6 * GIB)
        p = self.detect([adapter("NVIDIA GeForce RTX 3060", "10DE", "2560", qw=6 * GIB)],
                        powershell([{"Name": "NVIDIA GeForce RTX 3060", "PNPDeviceID": "PCI\\VEN_10DE"}]),
                        nvidia=[smi])
        self.assertEqual(p.gpus, (smi,))

    def test_ram_description(self):
        self.assertEqual(ram_description([{"SMBIOSMemoryType": 26, "Speed": 3200}]), "DDR4-3200")
        self.assertEqual(ram_description([{"SMBIOSMemoryType": 0}]), None)
        self.assertIsNone(ram_description([]))


# --------------------------------------------------------------------------- profile

class ProfilesTest(unittest.TestCase):
    EXPECTED = {   # Llama 3.1 8B Q4_K_M, kontekst 8k
        "desktop_rtx3060_linux": FitLevel.GREEN,
        "windows_amd_rx7900xtx": FitLevel.GREEN,
        "laptop_no_gpu": FitLevel.YELLOW,
        "two_nvidia": FitLevel.GREEN,
        "old_nvidia_pascal": FitLevel.GREEN,
        "mixed_vendors": FitLevel.GREEN,
        "detection_failed": None,
    }

    def test_every_profile_file_is_covered(self):
        self.assertEqual({p.stem for p in PROFILES.glob("*.json")}, set(self.EXPECTED))

    def test_profiles_give_expected_verdicts(self):
        for name, level in self.EXPECTED.items():
            hw = load_profile(PROFILES / f"{name}.json")
            if level is None:
                with self.assertRaises(IncompleteProfileError, msg=name):
                    evaluate(LLAMA_31_8B, hw, RunSettings(), QUANTS)
            else:
                self.assertEqual(evaluate(LLAMA_31_8B, hw, RunSettings(), QUANTS).level, level, name)

    def test_profile_warnings(self):
        def warnings(name):
            return evaluate(LLAMA_31_8B, load_profile(PROFILES / f"{name}.json"),
                            RunSettings(), QUANTS).warnings
        self.assertIn("warn.nvidia_legacy", warnings("old_nvidia_pascal"))
        self.assertIn("warn.mixed_gpu_vendors", warnings("mixed_vendors"))
        self.assertTrue(evaluate(LLAMA_31_8B, load_profile(PROFILES / "laptop_no_gpu.json"),
                                 RunSettings(), QUANTS).capacity.cpu_only)

    def test_round_trip(self):
        for path in PROFILES.glob("*.json"):
            p = load_profile(path)
            self.assertEqual(profile_from_dict(profile_to_dict(p)), p, path.name)

    def test_factory(self):
        self.assertEqual(create_detector(PROFILES / "two_nvidia.json").detect().ram_total, gib(64))
        with self.assertRaises(PlatformUnsupported):
            create_detector(platform="darwin").detect()
        self.assertEqual(type(create_detector(platform="linux")).__name__, "LinuxDetector")
        self.assertEqual(type(create_detector(platform="win32")).__name__, "WindowsDetector")


if __name__ == "__main__":
    unittest.main()
