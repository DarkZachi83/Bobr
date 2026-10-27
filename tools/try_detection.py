"""Narzędzie deweloperskie: pokaż, co program wykrył na tym komputerze.

    python tools/try_detection.py            # czytelny opis
    python tools/try_detection.py --json     # profil JSON (do zgłoszeń i jako --fake-hw)
    python tools/try_detection.py --fake-hw tests/fake_hardware/two_nvidia.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aibobr.calculator.verdict import capacity  # noqa: E402
from aibobr.detection.base import PlatformUnsupported  # noqa: E402
from aibobr.detection.factory import create_detector  # noqa: E402
from aibobr.detection.profiles import profile_to_dict  # noqa: E402
from aibobr.domain.models import RunSettings  # noqa: E402
from aibobr.domain.units import gib, to_gib  # noqa: E402


def gb(value):
    return "?" if value is None else f"{to_gib(value):.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fake-hw", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    t = time.time()
    try:
        hw = create_detector(args.fake_hw).detect()
    except PlatformUnsupported as e:
        print(f"System {e.platform} nie jest obsługiwany w wersji 1 – tryb Zaawansowany.")
        return 1
    if args.json:
        print(json.dumps(profile_to_dict(hw), ensure_ascii=False, indent=2))
        return 0

    print(f"System: {hw.platform}   (skan {time.time() - t:.1f} s)")
    print(f"Procesor: {hw.cpu_name or '?'}")
    ram = f"RAM: {gb(hw.ram_total)}"
    if hw.ram_used is not None:
        ram += f" (zajęte teraz {gb(hw.ram_used)})"
    print(ram + (f"  {hw.ram_description}" if hw.ram_description else ""))
    if not hw.gpus:
        print("Karty graficzne: brak")
    for g in hw.gpus:
        extra = []
        if g.integrated:
            extra.append("zintegrowana – nie liczy się")
        if g.compute_capability:
            extra.append(f"CC {g.compute_capability[0]}.{g.compute_capability[1]}")
        if g.driver_version:
            extra.append(f"sterownik {g.driver_version}")
        if g.vram_used is not None:
            extra.append(f"zajęte teraz {gb(g.vram_used)}")
        print(f"Karta: {g.name} [{g.vendor.value}]  VRAM {gb(g.vram_total)}"
              + (f"  ({', '.join(extra)})" if extra else ""))

    reserve = gib(4) if hw.platform == "windows" else gib(2)
    cap, warnings = capacity(hw, RunSettings(ram_reserve=reserve))
    print("\nPasek systemu (rezerwy domyślne):")
    if cap.missing:
        print(f"  nie da się policzyć, brakuje: {', '.join(cap.missing)} → tryb Zaawansowany")
    else:
        mode = "tryb CPU" if cap.cpu_only else f"{len(cap.gpus_used)} karta/y"
        print(f"  VRAM użytkowy {gb(cap.vram_usable)} + RAM użytkowy {gb(cap.ram_usable)}  ({mode})")
    for w in warnings:
        print(f"  ostrzeżenie: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
