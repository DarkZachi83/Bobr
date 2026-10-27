"""AI Bóbr (AI BEAVER) – AI Hardware Checker. Punkt startowy.

main.py tylko składa obiekty (ustawienia → tabela kwantyzacji → detektor → kontroler)
i wybiera interfejs. Logika żyje w pakiecie aibobr/ – jak w NIUCHACZU.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from aibobr import resources
from aibobr.app.config import ConfigStore, user_cache_dir, user_config_dir
from aibobr.app.controller import AppController
from aibobr.calculator.quants import QuantTable
from aibobr.cli.cli import Cli, parse_context, parse_gb_list
from aibobr.detection.factory import create_detector
from aibobr.i18n import LANGUAGES, set_language, t
from aibobr.knowledge import catalog
from aibobr.sources.cache import ModelCache

ROOT = resources.root()
PATHS = {
    "config": resources.path("config"),
    "quants": resources.path("data", "quants.json"),
    "catalog": resources.path("data", "models", "catalog.json"),
}


def build_app(fake_hw: Optional[Path] = None, user_dir: Optional[Path] = None,
              cache_dir: Optional[Path] = None) -> AppController:
    """Złożenie warstw bez interfejsu – używane też w testach."""
    config = ConfigStore(PATHS["config"], user_dir)
    downloaded = (user_dir or user_config_dir()) / "catalog.json"
    return AppController(
        config=config,
        quants=QuantTable.load(PATHS["quants"]),
        detector=create_detector(fake_hw),
        cache=ModelCache(cache_dir or user_cache_dir() / "models"),
        catalog=catalog.load(PATHS["catalog"], downloaded),
        catalog_download=downloaded,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=t("args.description"))
    p.add_argument("model", nargs="?", help=t("args.model"))
    p.add_argument("--cli", action="store_true", help=t("args.cli"))
    p.add_argument("--quant", help=t("args.quant"))
    p.add_argument("--context", type=parse_context, help=t("args.context"))
    p.add_argument("--kv-cache", choices=("f16", "q8_0", "q4_0"), help=t("args.kv_cache"))
    p.add_argument("-e", "--explain", action="store_true", help=t("args.explain"))
    p.add_argument("--ram", type=float, metavar="GB", help=t("args.ram"))
    p.add_argument("--vram", type=parse_gb_list, metavar="GB[,GB]", help=t("args.vram"))
    p.add_argument("--fake-hw", type=Path, metavar="JSON", help=t("args.fake_hw"))
    p.add_argument("--lang", choices=LANGUAGES, help=t("args.lang"))
    p.add_argument("--no-color", action="store_true", help=t("args.no_color"))
    p.add_argument("--update-catalog", action="store_true", help=t("args.update_catalog"))
    p.add_argument("-v", "--verbose", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--bobr", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # Język trzeba znać przed zbudowaniem pomocy (--help jest tłumaczone).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang", choices=LANGUAGES)
    pre.add_argument("--fake-hw", type=Path)
    known, _ = pre.parse_known_args(argv)
    controller = build_app(known.fake_hw)
    set_language(known.lang or controller.settings.get("language"))

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.context:
        controller.settings["default_context"] = args.context
    if not args.cli:
        code = _try_gui(controller, args)
        if code is not None:
            return code
        print(t("gui.unavailable"), file=sys.stderr)

    from aibobr.cli.bar import PLAIN, palette_for
    cli = Cli(controller, palette=PLAIN if args.no_color else palette_for(sys.stdout))
    return cli.run(args)


def _try_gui(controller: AppController, args) -> Optional[int]:
    """Okno, jeśli się da; None = brak tkintera albo ekranu → tryb tekstowy."""
    try:
        import tkinter
        from aibobr.gui.main_window import run_gui
    except ImportError:
        return None
    try:
        return run_gui(controller, args.model, args.quant)
    except tkinter.TclError as e:                    # np. brak ekranu (SSH bez X)
        logging.getLogger(__name__).info("GUI unavailable: %s", e)
        return None


if __name__ == "__main__":
    sys.exit(main())
