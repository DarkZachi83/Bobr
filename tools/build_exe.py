"""Pakuje AI Bobra w samodzielne pliki wykonywalne (PyInstaller) – SPEC §12.

    pip install pyinstaller
    python tools/build_exe.py            # dist/AI-Bobr(.exe) – okno, dist/AI-Bobr-cli(.exe) – konsola

Na Windowsie robi to automatycznie GitHub Actions (.github/workflows/release.yml) po
wypchnięciu taga v*. Lokalnie działa też na Linuksie (do sprawdzenia pakowania).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (źródło, katalog docelowy w paczce) – te same ścieżki, których szuka aibobr/resources.py
DATA = [
    ("aibobr/i18n/*.json", "aibobr/i18n"),
    ("data/quants.json", "data"),
    ("data/models/catalog.json", "data/models"),
    ("config/settings.json", "config"),
    ("assets/aibobr.png", "assets"),
]


def build(name: str, entry: str, windowed: bool) -> None:
    import PyInstaller.__main__
    args = [str(ROOT / entry), "--name", name, "--onefile", "--noconfirm", "--clean",
            "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / name),
            "--specpath", str(ROOT / "build"), "--paths", str(ROOT),
            "--icon", str(ROOT / "assets" / "aibobr.ico"),
            "--windowed" if windowed else "--console"]
    for src, dest in DATA:
        args += ["--add-data", f"{ROOT / src}{os.pathsep}{dest}"]
    PyInstaller.__main__.run(args)


def main() -> int:
    if shutil.which("pyinstaller") is None and "PyInstaller" not in sys.modules:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print("Brak PyInstallera: pip install pyinstaller")
            return 1
    build("AI-Bobr", "main.py", windowed=True)
    build("AI-Bobr-cli", "packaging/cli_entry.py", windowed=False)
    print("\nGotowe:", ", ".join(p.name for p in sorted((ROOT / "dist").iterdir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
