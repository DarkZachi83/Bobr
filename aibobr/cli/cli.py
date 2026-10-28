"""Tryb tekstowy: wskaż model → skan → werdykt (+ wyjaśnienie z --explain)."""
from __future__ import annotations

import re
import sys
from argparse import Namespace
from typing import TextIO

from aibobr import __version__
from aibobr.app.controller import AppController, Overrides
from aibobr.calculator.verdict import IncompleteProfileError
from aibobr.cli.bar import Palette, palette_for, render_bar
from aibobr.detection.base import PlatformUnsupported
from aibobr.domain.enums import FitLevel
from aibobr.i18n import fmt_context, fmt_gb, fmt_params, t
from aibobr.knowledge.advisor import Explanation, render, render_list
from aibobr.sources.base import SourceError
from aibobr.sources.factory import source_kind

BOBR_MESSAGE = "Patrz, jaki bóbr!"      # zawsze po polsku – żart nie przetrwa tłumaczenia

EXIT_OK, EXIT_USAGE, EXIT_SOURCE, EXIT_HARDWARE = 0, 1, 2, 3


def parse_context(text: str) -> int:
    """'8192', '16k', '32K' → liczba tokenów."""
    m = re.fullmatch(r"\s*(\d+)\s*([kK]?)\s*", str(text))
    if not m:
        raise ValueError(text)
    value = int(m.group(1)) * (1024 if m.group(2) else 1)
    if value <= 0:
        raise ValueError(text)
    return value


def parse_gb_list(text: str) -> list[float]:
    """'12' albo '24,24' (dwie karty) → [12.0] / [24.0, 24.0]."""
    return [float(x.replace(" ", "")) for x in re.split(r"[;,+]", str(text)) if x.strip()]


class Cli:
    def __init__(self, controller: AppController, out: TextIO = sys.stdout,
                 palette: Palette | None = None, stdin: TextIO = sys.stdin):
        self.c = controller
        self.out = out
        self.p = palette if palette is not None else palette_for(out)
        self.stdin = stdin

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def error(self, code: str, detail: str = "", **fields) -> None:
        self.say(self.p.paint(t(code, **fields), self.p.red))
        if detail:
            self.say(self.p.paint(f"  ({detail})", self.p.dim))

    # ------------------------------------------------------------------ kroki
    def _ask_model(self) -> str | None:
        if not (hasattr(self.stdin, "isatty") and self.stdin.isatty()):
            return None
        self.say(t("cli.ask_model_hint"))
        try:
            return input(t("cli.ask_model") + " ").strip() or None
        except (EOFError, KeyboardInterrupt):
            return None

    def run(self, args: Namespace) -> int:
        self.say(self.p.paint(t("app.title", version=__version__), self.p.bold_yellow))
        self.say()

        if getattr(args, "update_catalog", False):
            code = self._update_catalog()
            if not args.model:
                return code
        text = args.model or self._ask_model()
        if not text:
            self.error("cli.no_model")
            return EXIT_USAGE
        if source_kind(text) == "remote":
            self.say(self.p.paint(t("cli.loading_remote"), self.p.dim))
        try:
            model = self.c.load_model(text)
            spec = model.spec(args.quant) if args.quant else model.spec()
        except SourceError as e:
            if e.code == "source.unknown_quant":
                self.error("cli.unknown_quant", quant=args.quant,
                           available=", ".join(v.quant for v in model.variants))
            else:
                self.error(e.code, e.detail)
            return EXIT_SOURCE

        overrides = Overrides(ram_gb=args.ram, vram_gb=args.vram or ())
        try:
            hw = self.c.scan()
        except PlatformUnsupported as e:
            if args.ram is None:
                self.error("detection.unsupported", platform=e.platform)
                return EXIT_HARDWARE
            hw = None
        hw = self.c.apply_overrides(hw, overrides, sys.platform)
        settings = self.c.run_settings(hw.platform, args.context, args.kv_cache)

        try:
            v, expl = self.c.explain(model, spec, hw, settings)
        except IncompleteProfileError as e:
            self.error("detection.missing", ", ".join(e.missing))
            return EXIT_HARDWARE

        self._header(spec, hw, settings)
        for line in render_bar(v, self.p, bobr=args.bobr):
            self.say(line)
        self.say()
        self._verdict(v, expl, args.bobr)
        if args.explain:
            self._explanation(expl)
        else:
            self.say(self.p.paint(t("cli.hint_explain"), self.p.dim))
        others = [x.quant for x in model.variants if x.quant != spec.quant]
        if others and not args.quant:
            self.say(self.p.paint(t("cli.hint_quant", available=", ".join(others[:12])),
                                  self.p.dim))
        return EXIT_OK

    def _update_catalog(self) -> int:
        try:
            newer = self.c.update_catalog()
        except SourceError as e:
            self.error(e.code if e.code.startswith("catalog.") else "catalog.failed", e.detail)
            return EXIT_SOURCE
        cat = self.c.catalog
        key = "catalog.updated" if newer else "catalog.current"
        self.say(t(key, count=len(cat), date=cat.updated or "—"))
        self.say()
        return EXIT_OK

    # ------------------------------------------------------------------ wydruk
    def _header(self, spec, hw, settings) -> None:
        model = f"{spec.name} · {spec.quant} · {t('cli.context')} {fmt_context(settings.context_tokens)}"
        if spec.is_moe and spec.params_active:
            model += f" · {t('cli.moe', active=fmt_params(spec.params_active))}"
        parts = [hw.cpu_name or "?", f"RAM {fmt_gb(hw.ram_total)}"]
        parts += [f"{g.name} {fmt_gb(g.vram_total)}" for g in hw.gpus if not g.integrated]
        self.say(f"{t('cli.model'):<10} {model}")
        self.say(f"{t('cli.computer'):<10} {' · '.join(parts)}")
        self.say()

    def _verdict(self, v, expl: Explanation, bobr: bool) -> None:
        color = {FitLevel.GREEN: self.p.green, FitLevel.YELLOW: self.p.yellow,
                 FitLevel.RED: self.p.red}[v.level]
        self.say(self.p.paint(">> " + render(expl.summary), color))
        for note in expl.notes:
            self.say(self.p.paint("!  " + render(note), self.p.yellow))
        if bobr and v.level == FitLevel.GREEN:
            self.say(self.p.paint(BOBR_MESSAGE, self.p.bold_yellow))
        self.say()

    def _explanation(self, expl: Explanation) -> None:
        sections = [(None, expl.need), (t("expl.pc.title"), expl.computer),
                    (t(expl.actions_title), expl.actions)]
        for title, lines in sections:
            if title is None:                  # pierwsza linia sekcji „need” jest tytułem
                title, lines = render(lines[0]), lines[1:]
            self.say(title)
            for text in render_list(lines):
                self.say(f"  → {text}")
            self.say()
