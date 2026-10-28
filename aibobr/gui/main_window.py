"""Główne okno AI Bobra (SPEC §3, §6, §8, §11).

Model → Twój komputer (Automatyczny / Zaawansowany) → Wynik: pasek, werdykt,
uwagi i przycisk „Wyjaśnienie”. Pobieranie z HF i skan idą w tle (worker.py).
Zmiana języka przebudowuje okno, zachowując stan.
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
from aibobr.gui.theme import COLORS, VERDICT_COLORS, apply_style
from aibobr.gui.widgets.explanation import ExplanationPanel
from aibobr.gui.widgets.hardware_panel import HardwarePanel
from aibobr.gui.widgets.memory_bar import MemoryBar
from aibobr.gui.widgets.model_picker import ModelPicker
from aibobr.gui.worker import Worker
from aibobr.i18n import LANGUAGES, current_language, fmt_context, fmt_params, set_language, t
from aibobr.knowledge.advisor import render
from aibobr.sources.base import LoadedModel, SourceError
from aibobr.sources.factory import OLLAMA_PREFIX, source_kind

class MainWindow:
    def __init__(self, root: tk.Tk, controller: AppController,
                 initial_model: Optional[str] = None, initial_quant: Optional[str] = None):
        self.root, self.c = root, controller
        self.worker = Worker(root)
        self.model: Optional[LoadedModel] = None
        self.quant: Optional[str] = initial_quant
        self.model_text = initial_model or ""
        self.hw = None
        self.unsupported: Optional[str] = None
        apply_style(root)
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

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        head = ttk.Frame(outer)
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text=t("gui.app_name"), font=("TkDefaultFont", 16, "bold")).pack(side="left")
        ttk.Label(head, text="  AI Hardware Checker", foreground=COLORS["muted"]).pack(side="left", pady=(6, 0))

        self.picker = ModelPicker(outer, self.check, self.manual, self.select_quant, self.from_ollama)
        self.picker.pack(fill="x", pady=(0, 8))
        self.picker.text.set(self.model_text)

        self.hardware = HardwarePanel(outer, self.recompute, self.save_defaults)
        self.hardware.pack(fill="x", pady=(0, 8))

        result = ttk.Frame(outer, style="Panel.TFrame")
        result.pack(fill="x")
        self.header = ttk.Label(result, text="", style="Title.TLabel")
        self.header.pack(anchor="w", padx=10, pady=(8, 0))
        self.bar = MemoryBar(result)
        self.bar.pack(fill="x", padx=6, pady=(4, 0))
        self.verdict = ttk.Label(result, text="", style="Verdict.TLabel", wraplength=820)
        self.verdict.pack(anchor="w", padx=10)
        self.notes = ttk.Frame(result, style="Inner.TFrame")
        self.notes.pack(fill="x", padx=10, pady=(2, 0))
        self.explanation = ExplanationPanel(result)
        self.explanation.pack(fill="x", padx=1, pady=(4, 1))

        if self.hw is not None or self.unsupported:
            self.hardware.set_scan(self.hw, self.c.run_settings(self._platform()),
                                   self._platform(), self.unsupported)
        if state:
            self.hardware.import_state(state)
        self.picker.set_model(self.model, self.quant)
        self.recompute()

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
        lang = tk.Menu(settings, tearoff=False)
        self.lang_var = tk.StringVar(value=current_language())
        for code in LANGUAGES:
            lang.add_radiobutton(label=t(f"lang.{code}"), value=code, variable=self.lang_var,
                                 command=lambda c=code: self.change_language(c))
        settings.add_cascade(label=t("gui.menu.language"), menu=lang)
        bar.add_cascade(label=t("gui.menu.settings"), menu=settings)

        help_menu = tk.Menu(bar, tearoff=False)
        help_menu.add_command(label=t("gui.menu.about"), command=self.about)
        bar.add_cascade(label=t("gui.menu.help"), menu=help_menu)
        self.root.configure(menu=bar)

    # ------------------------------------------------------------------ sprzęt
    @staticmethod
    def _platform() -> str:
        return sys.platform

    def start_scan(self) -> None:
        def done(hw):
            self.hw = hw
            self.hardware.set_scan(hw, self.c.run_settings(hw.platform), hw.platform)
            self.recompute()

        def failed(e):
            if isinstance(e, PlatformUnsupported):
                self.unsupported = e.platform
            else:
                self.unsupported = str(e) or type(e).__name__
            self.hardware.set_scan(None, self.c.run_settings(self._platform()),
                                   self._platform(), self.unsupported)
            self.recompute()

        self.worker.run("scan", self.c.scan, done, failed)

    def save_defaults(self, changes: dict) -> None:
        self.c.config.save_user(changes)
        self.c.settings = self.c.config.load()
        messagebox.showinfo(t("gui.app_name"), t("gui.saved"), parent=self.root)

    # ------------------------------------------------------------------ model
    def check(self, text: str) -> None:
        self.model_text = text
        self.picker.set_busy(True)
        self.picker.set_status({"remote": t("cli.loading_remote"), "file": t("gui.reading_file"),
                                "ollama": t("gui.reading_ollama")}[source_kind(text)])

        def done(model):
            self.picker.set_busy(False)
            self.picker.set_status()
            self.set_model(model, self.quant if self.model is None else None)

        def failed(e):
            self.picker.set_busy(False)
            if isinstance(e, SourceError):
                self.picker.set_status(t(e.code), "Error")
            else:
                self.picker.set_status(t("gui.unexpected", error=repr(e)), "Error")

        self.worker.run("model", lambda: self.c.load_model(text), done, failed)

    def set_model(self, model: LoadedModel, quant: Optional[str] = None) -> None:
        self.model = model
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
        self.bar.set_verdict(None, hint=self.model is None)
        self.header.configure(text="")
        self.verdict.configure(text=message, style=style, foreground=color or COLORS["text"])
        for child in self.notes.winfo_children():
            child.destroy()
        self.explanation.set_explanation(None)

    def from_ollama(self) -> None:
        from aibobr.gui.dialogs.ollama_models import OllamaModelsDialog
        dialog = OllamaModelsDialog(self.root, self.worker, self.c.ollama_models)
        self.root.wait_window(dialog)
        if dialog.result:
            text = OLLAMA_PREFIX + dialog.result
            self.picker.text.set(text)
            self.check(text)

    def recompute(self) -> None:
        if not hasattr(self, "bar"):
            return
        self.picker.set_advanced(self.hardware.mode.get() == "advanced")
        if self.model is None:
            self._clear_result()
            return
        try:
            hw, settings = self.hardware.current()
        except ValueError as e:
            if e.args and e.args[0] == ["scan"]:
                self._clear_result(t("gui.scanning"), "Muted.TLabel")
            else:
                self._clear_result(t("gui.fix_fields"), color=COLORS["error"])
            return
        spec = self.model.spec(self.quant)
        try:
            v, expl = self.c.explain(self.model, spec, hw, settings)
        except IncompleteProfileError:
            self._clear_result(t("gui.missing"), color=COLORS["error"])
            ttk.Button(self.notes, text=t("gui.set_manually"),
                       command=self.hardware.show_advanced).pack(anchor="w", pady=4)
            return
        except MissingModelData:
            self._clear_result(t("note.estimate"), color=COLORS["error"])
            return

        header = f"{spec.name}  ·  {spec.quant}  ·  {t('cli.context')} {fmt_context(settings.context_tokens)}"
        if spec.is_moe and spec.params_active:
            header += f"  ·  {t('cli.moe', active=fmt_params(spec.params_active))}"
        self.header.configure(text=header)
        self.bar.set_verdict(v)
        self.verdict.configure(text=render(expl.summary), style="Verdict.TLabel",
                               foreground=VERDICT_COLORS[v.level])
        for child in self.notes.winfo_children():
            child.destroy()
        for note in expl.notes:
            ttk.Label(self.notes, text="⚠ " + render(note), style="Note.TLabel",
                      wraplength=820).pack(anchor="w")
        self.explanation.set_explanation(expl)

    # ------------------------------------------------------------------ menu
    def change_language(self, code: str) -> None:
        set_language(code)
        self.c.config.save_user({"language": code})
        expanded = self.explanation.expanded
        self.build(self.hardware.export_state())
        if expanded:
            self.explanation.toggle()

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
    root = tk.Tk()           # bez stałej wysokości: okno rośnie po rozwinięciu wyjaśnienia
    icon = resources.path("assets", "aibobr.png")
    if icon.exists():
        root.iconphoto(True, tk.PhotoImage(file=str(icon)))
    MainWindow(root, controller, model, quant)
    root.mainloop()
    return 0
