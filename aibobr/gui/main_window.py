"""Główne okno AI Bobra (SPEC §3, §6, §8, §11; układ etapu 13b).

Zakładki: Sprawdź | Statystyka | Zaawansowane (w 13c dojdą Modele na dysku).
Sprawdź: po lewej Model (z kontekstem i KV cache) i Wynik (pasek, werdykt, uwagi,
wyjaśnienie), po prawej Twój komputer z miernikami. Zaawansowane działają jak osobne okno:
ten sam model, sprzęt i parametry wpisane ręcznie, własny wynik. Na dole pasek stanu.
Pobieranie z HF i skan idą w tle (worker.py). Zmiana języka i stylu przebudowuje okno,
zachowując stan.
"""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Optional

from aibobr import __version__, resources
from aibobr.app.controller import AppController
from aibobr.calculator.memory import MissingModelData
from aibobr.calculator.verdict import IncompleteProfileError
from aibobr.detection.base import PlatformUnsupported
from aibobr.gui.dialogs.manual_model import ManualModelDialog
from aibobr.gui.theme import COLORS, THEMES, VERDICT_COLORS, apply_style, theme_name
from aibobr.gui.widgets.hardware_panel import HardwarePanel
from aibobr.gui.widgets.meters import Meters
from aibobr.gui.widgets.model_picker import ModelPicker
from aibobr.gui.widgets.result_view import ResultView
from aibobr.gui.widgets.run_options import RunOptions
from aibobr.gui.widgets.scroll_page import ScrollPage
from aibobr.gui.widgets.stats_panel import StatsPanel
from aibobr.gui.worker import Worker
from aibobr.i18n import LANGUAGES, fmt_context, fmt_params, set_language, system_language, t
from aibobr.knowledge.advisor import audio_label, image_label
from aibobr.sources.base import LoadedModel, SourceError
from aibobr.sources.factory import OLLAMA_PREFIX, source_kind

# Błędy, przy których szczegół techniczny pomaga w zgłoszeniu (np. „no general.architecture”).
DETAIL_CODES = {"source.bad_file", "source.gguf_version", "source.no_model_gguf", "source.http_error"}


class MainWindow:
    def __init__(self, root: tk.Tk, controller: AppController,
                 initial_model: Optional[str] = None, initial_quant: Optional[str] = None):
        self.root, self.c = root, controller
        self.worker = Worker(root)
        self.model: Optional[LoadedModel] = None
        self.loading: Optional[str] = None       # komunikat wczytywania nowego modelu
        self.last = None                         # (spec, werdykt, sprzęt, ustawienia) – do zapisu
        self.tab = "check"                       # otwarta zakładka (zachowana przy zmianie języka)
        self.quant: Optional[str] = initial_quant
        self.model_text = initial_model or ""
        self.hw = None
        self.unsupported: Optional[str] = None
        self.theme = apply_style(root, theme_name(self.c.settings.get("theme")))
        root.minsize(760, 560)
        self.build()
        self.start_scan()
        if initial_model:
            self.check(initial_model)

    # ------------------------------------------------------------------ budowa
    def build(self, state: Optional[dict] = None) -> None:
        for child in self.root.winfo_children():
            child.destroy()
        self.root.title(t("app.title", version=__version__))
        self._menu()

        outer = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        outer.pack(fill="both", expand=True)
        head = ttk.Frame(outer)
        head.pack(fill="x", pady=(0, 6))
        ttk.Label(head, text=t("gui.app_name"), font=("TkDefaultFont", 16, "bold")).pack(side="left")
        ttk.Label(head, text="  AI Hardware Checker", foreground=COLORS["muted"]).pack(side="left", pady=(6, 0))
        other = "light" if self.theme == "dark" else "dark"
        ttk.Button(head, text="☀" if self.theme == "dark" else "☾", width=3,
                   command=lambda: self.change_theme(other)).pack(side="right")
        self.status_bar = ttk.Label(outer, text="", foreground=COLORS["muted"], anchor="w")
        self.status_bar.pack(side="bottom", fill="x", pady=(4, 0))

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.pages: dict[str, tk.Widget] = {}

        # --- Sprawdź
        check = ScrollPage(self.tabs, padding=(0, 8, 0, 0))
        self.tabs.add(check, text=t("gui.tab.check"))
        self.pages["check"] = check
        self.check_page = check
        self.run_options = None
        self.picker = ModelPicker(check.left, self.check, self.manual, self.select_quant,
                                  self.from_disk, self.set_status, options=self._options)
        self.picker.pack(fill="x", pady=(0, 8))
        self.picker.text.set(self.model_text)
        self.result = ResultView(check.left, lambda: check.wrap)
        self.result.pack(fill="both", expand=True)
        self.save_button = ttk.Button(self.result.actions, text="💾 " + t("gui.save_compare"),
                                      command=self.save_compare)
        self.save_button.state(["disabled"])
        self.save_button.pack(side="left")
        self.bench_button = ttk.Button(self.result.actions, text="⏱ " + t("bench.button"),
                                       command=lambda: self.open_benchmark([self.model_text]))
        self.hardware = HardwarePanel(check.right, self.recompute, self.save_defaults, mode="auto",
                                      wrap=230)
        self.hardware.pack(fill="both", expand=True)
        self.meters = Meters(self.hardware.meter_box)
        self.meters.pack(fill="x", pady=(4, 6))
        self.manual_hint = ttk.Frame(self.hardware, style="Inner.TFrame")   # brak skanu → Zaawansowane

        # --- Statystyka
        self.stats = StatsPanel(self.tabs, self.c.saved, self.open_saved, self.compare_saved,
                                self.fetch_hf, self.open_benchmark) if self.c.saved else None
        if self.stats:
            self.tabs.add(self.stats, text=t("gui.tab.stats"))
            self.pages["stats"] = self.stats

        # --- Zaawansowane: jak osobne okno – ten sam model, sprzęt wpisany ręcznie, własny wynik
        adv = ScrollPage(self.tabs, padding=(0, 8, 0, 0))
        self.tabs.add(adv, text=t("gui.tab.advanced"))
        self.pages["advanced"] = adv
        self.adv_model = ttk.Frame(adv.left, style="Panel.TFrame")
        self.adv_model.pack(fill="x", pady=(0, 8))
        ttk.Label(self.adv_model, text=t("gui.model"), style="Title.TLabel").pack(anchor="w", padx=10, pady=(8, 0))
        intro = ttk.Label(self.adv_model, text=t("gui.adv_intro"), style="Muted.TLabel", wraplength=500)
        intro.pack(anchor="w", padx=10, fill="x")
        intro.bind("<Configure>", lambda e: intro.configure(wraplength=max(300, e.width - 4)))
        self.adv_model_label = ttk.Label(self.adv_model, text="", style="Panel.TLabel")
        self.adv_model_label.pack(anchor="w", padx=10, pady=(4, 8))
        self.adv_result = ResultView(adv.left, lambda: adv.wrap)
        self.adv_result.pack(fill="both", expand=True)
        self.advanced = HardwarePanel(adv.right, self.recompute, self.save_defaults, mode="advanced",
                                      title=t("gui.adv_computer"))
        self.advanced.pack(fill="both", expand=True)
        self.adv_meters = Meters(self.advanced.meter_box)
        self.adv_meters.pack(fill="x", pady=(4, 6))

        self.tabs.select(self.pages.get(self.tab, check))
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed())

        if self.hw is not None or self.unsupported:
            platform = self._platform()
            for panel in (self.hardware, self.advanced):
                panel.set_scan(self.hw, self.c.run_settings(platform), platform, self.unsupported)
            self.run_options.set_defaults(self.c.run_settings(platform))
        if state:
            self.run_options.import_state(state.get("options", {}))
            if state.get("advanced"):
                self.advanced.import_state(state["advanced"])
        self.picker.set_model(self.model, self.quant)
        self.recompute()

    # Wynik zakładki Sprawdź – skróty (używane też w testach)
    header = property(lambda self: self.result.header)
    bar = property(lambda self: self.result.bar)
    verdict = property(lambda self: self.result.verdict)
    notes = property(lambda self: self.result.notes)
    explanation = property(lambda self: self.result.explanation)

    def _options(self, parent) -> RunOptions:
        self.run_options = RunOptions(parent, self.recompute)
        return self.run_options

    def export_state(self) -> dict:
        return {"options": self.run_options.export_state(), "advanced": self.advanced.export_state()}

    def set_status(self, text: str = "", kind: str = "Muted") -> None:
        color = COLORS["error"] if kind == "Error" else COLORS["muted"]
        if hasattr(self, "status_bar"):
            self.status_bar.configure(text=text, foreground=color)

    def _menu(self) -> None:
        bar = tk.Menu(self.root)
        file = tk.Menu(bar, tearoff=False)
        file.add_command(label=t("gui.menu.open"), command=lambda: self.picker.browse())
        file.add_command(label=t("gui.manual"), command=self.manual)
        file.add_separator()
        file.add_command(label=t("gui.menu.quit"), command=self.root.destroy)
        bar.add_cascade(label=t("gui.menu.file"), menu=file)

        settings = tk.Menu(bar, tearoff=False)
        settings.add_command(label=t("gui.menu.token"), command=self.set_token)
        settings.add_command(label=t("gui.menu.update_catalog"), command=self.update_catalog)
        settings.add_separator()
        style = tk.Menu(settings, tearoff=False)
        self.theme_var = tk.StringVar(value=self.theme)
        for name in THEMES:
            style.add_radiobutton(label=t(f"gui.theme.{name}"), value=name, variable=self.theme_var,
                                  command=lambda n=name: self.change_theme(n))
        settings.add_cascade(label=t("gui.menu.theme"), menu=style)
        bar.add_cascade(label=t("gui.menu.settings"), menu=settings)

        # Język na wierzchu, nie w Ustawieniach – program startuje w języku systemu, a osoba,
        # która nie zna tego języka, musi szybko znaleźć zmianę (uwaga autora).
        lang = tk.Menu(bar, tearoff=False)
        # "" = język systemu (zapis "language": "") – powrót do automatu bez edycji pliku
        saved = str(self.c.settings.get("language") or "")
        self.lang_var = tk.StringVar(value=saved if saved in LANGUAGES else "")
        lang.add_radiobutton(label=t("gui.lang_system", name=t(f"lang.{system_language()}")), value="",
                             variable=self.lang_var, command=lambda: self.change_language(""))
        lang.add_separator()
        for code in LANGUAGES:
            lang.add_radiobutton(label=t(f"lang.{code}"), value=code, variable=self.lang_var,
                                 command=lambda c=code: self.change_language(c))
        bar.add_cascade(label=t("gui.menu.language"), menu=lang)

        help_menu = tk.Menu(bar, tearoff=False)
        help_menu.add_command(label=t("gui.menu.about"), command=self.about)
        bar.add_cascade(label=t("gui.menu.help"), menu=help_menu)
        self.root.configure(menu=bar)

    # ------------------------------------------------------------------ sprzęt
    @staticmethod
    def _platform() -> str:
        return sys.platform

    def start_scan(self) -> None:
        def apply(hw, platform):
            for panel in (self.hardware, self.advanced):
                panel.set_scan(hw, self.c.run_settings(platform), platform, self.unsupported)
            self.run_options.set_defaults(self.c.run_settings(platform))
            self.recompute()

        def done(hw):
            self.hw = hw
            apply(hw, hw.platform)

        def failed(e):
            if isinstance(e, PlatformUnsupported):
                self.unsupported = e.platform
            else:
                self.unsupported = str(e) or type(e).__name__
            apply(None, self._platform())

        self.worker.run("scan", self.c.scan, done, failed)

    def save_defaults(self, changes: dict) -> None:
        self.c.config.save_user(changes)
        self.c.settings = self.c.config.load()
        messagebox.showinfo(t("gui.app_name"), t("gui.saved"), parent=self.root)

    # ------------------------------------------------------------------ model
    def check(self, text: str, quant: Optional[str] = None) -> None:
        self.model_text = text
        # kwantyzacja: wskazana (zapisany model) albo z zapisanego stanu – tylko za 1. razem
        keep = quant or (self.quant if self.model is None else None)
        loading = {"remote": t("cli.loading_remote"), "file": t("gui.reading_file"),
                   "ollama": t("gui.reading_ollama")}[source_kind(text)]
        # Stary wynik znika od razu – inaczej do końca wczytywania wygląda jak wynik nowego modelu.
        self.model = None
        self.picker.set_model(None, None)
        self.loading = loading
        self._clear_result()
        self.picker.set_busy(True)
        self.picker.set_status(loading)

        def done(model):
            self.loading = None
            self.picker.set_busy(False)
            self.picker.set_status()
            self.set_model(model, keep)

        def failed(e):
            self.picker.set_busy(False)
            message = t(e.code) if isinstance(e, SourceError) else t("gui.unexpected", error=repr(e))
            if isinstance(e, SourceError) and e.detail and e.code in DETAIL_CODES:
                message += f"\n({e.detail})"         # szczegół techniczny – do zgłoszenia błędu
            self.loading = None
            self.picker.set_status(message, "Error")
            self._clear_result(message, color=COLORS["error"])

        self.worker.run("model", lambda: self.c.load_model(text), done, failed)

    def set_model(self, model: LoadedModel, quant: Optional[str] = None) -> None:
        self.model = model
        kind = "image" if model.arch.image is not None else "text"
        self.run_options.set_kind(kind)
        self.advanced.set_model_kind(kind)
        names = [v.quant for v in model.variants]
        self.quant = quant if quant in names else model.default_variant().quant
        self.picker.set_model(model, self.quant)
        self.recompute()

    def select_quant(self, quant: str) -> None:
        self.quant = quant
        self.recompute()

    def manual(self) -> None:
        dialog = ManualModelDialog(self.root, self.c.quants)
        self.root.wait_window(dialog)
        if dialog.result is not None:
            self.model_text = ""
            self.picker.text.set("")
            self.picker.set_status()
            self.set_model(dialog.result)

    # ------------------------------------------------------------------ wynik
    def _clear_result(self, message: str = "", style: str = "Verdict.TLabel", color=None) -> None:
        self.last = None
        if hasattr(self, "save_button"):
            self.save_button.state(["disabled"])
            self.bench_button.pack_forget()
        if not message and self.loading:     # trwa wczytywanie nowego modelu
            message, color = self.loading, COLORS["muted"]
        hint = self.model is None and not self.loading
        self.result.clear(message, style, color, hint)
        self.meters.set_verdict(None)
        self.adv_result.clear(message, style, color, hint)
        self.adv_meters.set_verdict(None)
        self.adv_model_label.configure(text=t("gui.adv_no_model") if self.model is None else "")

    def from_disk(self) -> None:
        from aibobr.gui.dialogs.found_models import FoundModelsDialog
        dialog = FoundModelsDialog(self.root, self.worker, self.c.find_models,
                                   self.c.scan_consent, self.c.set_scan_consent)
        self.root.wait_window(dialog)
        if dialog.bench_result:
            self.open_benchmark(dialog.bench_result)
        elif dialog.result:
            self.picker.text.set(dialog.result)
            self.check(dialog.result)

    def current(self):
        """Sprzęt i ustawienia zakładki Sprawdź: skan + kontekst / KV cache z sekcji Model."""
        hw, settings = self.hardware.current()
        return hw, self.run_options.apply(settings)

    def open_benchmark(self, targets: list[str]) -> None:
        """Okno benchmarku (etap 12a) – z listy modeli na dysku, z zakładki Sprawdź albo Statystyka."""
        from aibobr.gui.dialogs.benchmark import BenchmarkDialog
        try:
            hw, settings = self.current()
        except ValueError:
            messagebox.showinfo(t("bench.title"), t("stats.need_hw"), parent=self.root)
            return
        BenchmarkDialog(self.root, self.c, self.worker, targets, hw, settings)

    def recompute(self) -> None:
        if not hasattr(self, "adv_result") or self.run_options is None:
            return
        if self.model is None:
            self._clear_result()
            return
        spec = self.model.spec(self.quant)
        o = self.run_options.export_state()
        self.advanced.copy_run(o["context"], o["kv"], o["resolution"], o["batch"])
        self.adv_model_label.configure(text=f"{spec.name}  ·  {spec.quant}")
        self.last = None
        shown = self._render(self.result, self.meters, self.current, spec, main=True)
        self._render(self.adv_result, self.adv_meters, self.advanced.current, spec, main=False)
        if shown is not None:
            v, hw, settings = shown
            self.last = (spec, v, hw, settings)
            self.save_button.state(["!disabled"])
            if self.model_text.startswith(OLLAMA_PREFIX) and not spec.image and not spec.audio:
                self.bench_button.pack(side="left", padx=(6, 0))
            else:
                self.bench_button.pack_forget()
        else:
            self.save_button.state(["disabled"])
            self.bench_button.pack_forget()

    def _render(self, view: ResultView, meters: Meters, source, spec, main: bool):
        """Jeden wynik (Sprawdź albo Zaawansowane). Zwraca (werdykt, sprzęt, ustawienia) albo None."""
        meters.set_verdict(None)
        try:
            hw, settings = source()
        except ValueError as e:
            if e.args and e.args[0] == ["scan"] and self.unsupported:
                view.clear(t("detection.unsupported_gui", platform=self.unsupported), color=COLORS["error"])
                if main:                           # bez skanu: sprzęt wpisuje się w Zaawansowanych
                    ttk.Button(view.notes, text=t("gui.set_manually"),
                               command=self.show_advanced).pack(anchor="w", pady=4)
            elif e.args and e.args[0] == ["scan"]:
                view.clear(t("gui.scanning"), "Muted.TLabel")
            else:
                view.clear(t("gui.fix_fields"), color=COLORS["error"])
            return None
        try:
            v, expl = self.c.explain(self.model, spec, hw, settings)
        except IncompleteProfileError:
            view.clear(t("gui.missing"), color=COLORS["error"])
            if main:
                ttk.Button(view.notes, text=t("gui.set_manually"),
                           command=self.show_advanced).pack(anchor="w", pady=4)
            return None
        except MissingModelData:
            view.clear(t("note.estimate"), color=COLORS["error"])
            return None

        image = image_label(spec, settings)
        if image:                                    # obrazy: rozdzielczość zamiast kontekstu
            header = f"{spec.name}  ·  {spec.quant}  ·  {image}"
        else:
            header = f"{spec.name}  ·  {spec.quant}  ·  {t('cli.context')} {fmt_context(settings.context_tokens)}"
        if spec.is_moe and spec.params_active:
            header += f"  ·  {t('cli.moe', active=fmt_params(spec.params_active))}"
        audio = audio_label(spec, settings)
        if audio:
            header += f"\n{audio}"
        view.show(header, v, expl)
        meters.set_verdict(v)
        return v, hw, settings

    def show_advanced(self) -> None:
        self.tabs.select(self.pages["advanced"])

    # ------------------------------------------------------------------ porównanie (etap 11a)
    def _tab_changed(self) -> None:
        current = self.tabs.nametowidget(self.tabs.select())
        self.tab = next((k for k, w in self.pages.items() if w is current), "check")
        if self.stats and self.tab == "stats":
            self.stats.refresh()
            self.compare_saved()

    def compare_saved(self) -> None:
        """Kolumny porównania (etap 11b) – w tle: modele z HF wczytują się z pamięci podręcznej."""
        if not self.stats or not self.c.saved.items:
            return
        try:
            hw, settings = self.current()
        except ValueError:
            self.stats.set_status(t("stats.need_hw"))
            return
        items = list(self.c.saved.items)
        self.stats.set_status(t("stats.computing", total=len(items)))
        self.worker.run("compare", lambda: self.c.compare_saved(items, hw, settings),
                        lambda comps: self.stats.show(comps, self.c.hardware_text(hw)),
                        lambda e: self.stats.set_status(t("stats.error_other")))

    def fetch_hf(self) -> None:
        """Etap 11c: dane z Hugging Face – tylko na żądanie (przycisk), bo to połączenie z siecią."""
        if not self.stats or not self.c.saved.items:
            return
        items = list(self.c.saved.items)
        self.stats.set_status(t("stats.hf.fetching"))

        def done(out):
            errors = sum(1 for v in out.values() if "error" in v)
            self.stats.set_status(t("stats.hf.none") if not out else
                                  t("stats.hf.done", n=len(out) - errors) +
                                  (" " + t("stats.hf.errors", n=errors) if errors else ""))
            self.stats.refresh()

        self.worker.run("hf", lambda: self.c.fetch_hf_stats(items), done,
                        lambda e: self.stats.set_status(t("source.network")))

    def save_compare(self) -> None:
        if not self.last or not self.c.saved:
            return
        if not self.model_text:
            self.set_status(t("gui.saved_manual"))
            return
        from aibobr.app.saved import MAX_PER_KIND, ListFull
        item = self.c.saved_entry(self.model_text, *self.last)
        try:
            new = self.c.saved.add(item)
        except ListFull as e:
            self.set_status(t("gui.saved_full", group=t(f"stats.kind.{e.kind}"), max=MAX_PER_KIND))
            return
        self.set_status(t("gui.saved_new" if new else "gui.saved_updated"))
        if self.stats:
            self.stats.refresh()

    def open_saved(self, item) -> None:
        """Zakładka Statystyka → Otwórz: model z zapisanymi ustawieniami w zakładce Sprawdź."""
        self.tabs.select(self.pages["check"])
        self.run_options.set_kind("image" if item.kind == "image" else "text")
        self.run_options.apply_saved(item)
        self.picker.text.set(item.target)
        self.check(item.target, quant=item.quant)

    # ------------------------------------------------------------------ menu
    def change_language(self, code: str) -> None:
        """code = "" – język systemu (zapisujemy pusty wybór, więc działa też po zmianie systemu)."""
        set_language(code)
        self.c.settings["language"] = code
        self.c.config.save_user({"language": code})
        self.build(self.export_state())

    def change_theme(self, theme: str) -> None:
        self.theme = apply_style(self.root, theme)
        self.c.settings["theme"] = self.theme
        self.c.config.save_user({"theme": self.theme})
        self.build(self.export_state())

    def set_token(self) -> None:
        token = simpledialog.askstring(t("gui.menu.token"), t("gui.token_prompt"),
                                       show="*", parent=self.root)
        if token is not None:
            self.c.config.save_user({"hf_token": token.strip()})
            self.c.settings = self.c.config.load()

    def update_catalog(self) -> None:
        def done(newer):
            cat = self.c.catalog
            key = "catalog.updated" if newer else "catalog.current"
            messagebox.showinfo(t("gui.menu.update_catalog"),
                                t(key, count=len(cat), date=cat.updated or "—"), parent=self.root)
            if newer:
                self.recompute()

        def failed(e):
            code = e.code if isinstance(e, SourceError) and e.code.startswith("catalog.") else "catalog.failed"
            messagebox.showerror(t("gui.menu.update_catalog"), t(code), parent=self.root)

        self.worker.run("catalog", self.c.update_catalog, done, failed)

    def about(self) -> None:
        from aibobr.gui.dialogs.about import AboutDialog
        AboutDialog(self.root)


def run_gui(controller: AppController, model: Optional[str] = None,
            quant: Optional[str] = None) -> int:
    from aibobr.gui.theme import enable_dpi_awareness
    enable_dpi_awareness()
    root = tk.Tk()
    # Dwie kolumny, gdy ekran pozwala; mniejszy ekran – jedna kolumna z przewijaniem (etap 13b).
    width = min(1180, root.winfo_screenwidth() - 40)
    height = min(880, root.winfo_screenheight() - 80)
    root.geometry(f"{max(760, width)}x{max(560, height)}")
    icon = resources.path("assets", "aibobr.png")
    if icon.exists():
        root.iconphoto(True, tk.PhotoImage(file=str(icon)))
    MainWindow(root, controller, model, quant)
    root.mainloop()
    return 0
