"""GUI: geometria paska (zawsze) i testy dymne okna (tylko z ekranem i tkinterem).

Na komputerze z pulpitem testy okna uruchamiają się same; bez ekranu (serwer, CI)
są pomijane. W środowisku bez ekranu można je puścić przez: xvfb-run python -m unittest
"""
import tempfile
import time
import unittest
from pathlib import Path

from aibobr.calculator.verdict import evaluate
from aibobr.domain.models import RunSettings
from aibobr.gui.bar_layout import bar_layout
from aibobr.i18n import set_language
from tests.fixtures import DESKTOP_3060, LLAMA_31_8B, LLAMA_33_70B, QUANTS, pc
from tests.gguf_writer import llama_like
from tests.sizes import stop_all, write_model

PROFILES = Path(__file__).resolve().parent / "fake_hardware"


class BarLayoutTest(unittest.TestCase):
    def layout(self, model, hw):
        return bar_layout(evaluate(model, hw, RunSettings(), QUANTS), 100, 800)

    def test_green_model_ends_inside_vram(self):
        lay = self.layout(LLAMA_31_8B, DESKTOP_3060)
        self.assertEqual([s.kind for s in lay.system], ["vram", "ram"])
        self.assertEqual([s.kind for s in lay.model], ["weights", "kv", "overhead"])
        self.assertLessEqual(lay.model[-1].x1, lay.markers[0])
        self.assertLess(lay.system[-1].x1, 900)                 # zapas na prawym brzegu

    def test_red_model_has_missing_segment(self):
        lay = self.layout(LLAMA_33_70B, DESKTOP_3060)
        self.assertEqual(lay.system[-1].kind, "missing")
        self.assertAlmostEqual(lay.system[-1].x1, lay.model[-1].x1)

    def test_cpu_only_has_no_vram_segment(self):
        lay = self.layout(LLAMA_31_8B, pc(ram_gb=16))
        self.assertEqual([s.kind for s in lay.system], ["ram"])
        self.assertEqual(len(lay.markers), 1)


def _display_available() -> bool:
    try:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(_display_available(), "brak tkintera albo ekranu")
class GuiSmokeTest(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import main
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.gguf = d / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        write_model(self.gguf, llama_like(name="Meta Llama 3.1 8B Instruct"), 4_920_000_000)
        self.addCleanup(stop_all)
        set_language("pl")
        self.build = lambda profile: main.build_app(PROFILES / profile, user_dir=d / "cfg",
                                                    cache_dir=d / "cache")
        self.root = tk.Tk()

    def tearDown(self):
        import gc
        self.root.destroy()
        del self.root
        gc.collect()                  # okno tego testu sprzątamy tutaj, w wątku okna
        self.tmp.cleanup()
        set_language("en")

    def pump(self, until=lambda: False, seconds=8.0):
        end = time.time() + seconds
        while time.time() < end:
            self.root.update()
            if until():
                return
            time.sleep(0.02)

    def window(self, profile="real_laptop_rtx5050_linux.json", model=None):
        from aibobr.gui.main_window import MainWindow
        w = MainWindow(self.root, self.build(profile), str(model or self.gguf))
        self.pump(lambda: w.model is not None and w.hw is not None and w.verdict.cget("text"))
        return w

    def test_full_flow_on_authors_laptop(self):
        w = self.window()
        self.assertIn("Pójdzie na Twoim komputerze", w.verdict.cget("text"))
        auto = w.verdict.cget("text")
        # Zaawansowane: start z wartości Sprawdź – ten sam wynik, dopóki nic nie zmienisz
        self.assertEqual(w.adv_result.verdict.cget("text"), auto)
        w.run_options.context.set("20k")                    # kontekst w zakładce Sprawdź (etap 13b)
        self.pump(lambda: "wolniej" in w.verdict.cget("text"), 3)
        self.assertIn("nieco wolniej", w.verdict.cget("text"))      # 16k mieści się (pomiar autora)
        self.assertIn("Co możesz zrobić", w.explanation.text.get("1.0", "end"))   # zawsze widoczne
        self.assertEqual(w.advanced.context.get(), "20k")           # Zaawansowane idą za Sprawdź…
        w.advanced.context.set("8k")                                # …dopóki ich nie zmienisz
        self.pump(lambda: "całość" in w.adv_result.verdict.cget("text"), 3)
        w.run_options.context.set("32k")
        self.pump(seconds=0.6)
        self.assertEqual(w.advanced.context.get(), "8k")            # niezależne
        w.run_options.context.set("20k")
        self.pump(lambda: "wolniej" in w.verdict.cget("text"), 3)
        w.change_language("en")
        self.pump(seconds=0.5)
        self.assertIn("AI BEAVER", self.root.title())
        self.assertEqual(w.run_options.context.get(), "20k")        # stan zachowany
        self.assertEqual(w.advanced.context.get(), "8k")
        self.assertIn("a bit slower", w.verdict.cget("text"))

    def test_vram_busy_shown_and_noted(self):
        import json
        profile = json.loads((PROFILES / "desktop_rtx3060_linux.json").read_text(encoding="utf-8"))
        profile["gpus"][0]["vram_used_gb"] = 1.12                # pulpit na karcie (pomiar autora)
        path = Path(self.tmp.name) / "busy.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        w = self.window(path)
        texts = []
        stack = [w.hardware.body]
        while stack:
            widget = stack.pop()
            stack += widget.winfo_children()
            if widget.winfo_class() == "TLabel":
                texts.append(str(widget.cget("text")))
        self.assertTrue(any("zajęte teraz" in x and "VRAM" in x for x in texts))
        self.assertIn("10,8 GB użytkowe", w.explanation.text.get("1.0", "end"))   # 12 − 1,25 zajęte
        notes = [str(x.cget("text")) for x in w.notes.winfo_children() if x.winfo_class() == "TLabel"]
        self.assertTrue(any("system i otwarte programy" in x for x in notes), notes)

    def test_language_back_to_system(self):
        import json
        w = self.window()
        w.change_language("en")
        self.pump(seconds=0.3)
        self.assertIn("AI BEAVER", self.root.title())
        w.change_language("")                                   # „Język systemu”
        self.pump(seconds=0.3)
        saved = json.loads((Path(self.tmp.name) / "cfg" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["language"], "")
        self.assertEqual(w.lang_var.get(), "")

    def test_menu_language_on_top_and_theme_switch(self):
        import json
        from aibobr.gui import theme
        w = self.window()
        menu = self.root.nametowidget(self.root["menu"])
        labels = [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)
                  if menu.type(i) == "cascade"]
        self.assertEqual(labels, ["Plik", "Ustawienia", "Język (Language)", "Pomoc"])   # uwaga autora
        self.assertEqual(w.theme, "dark")                                  # domyślnie ciemny
        self.assertEqual(theme.COLORS["bg"], theme.PALETTES["dark"]["bg"])
        w.change_theme("light")
        self.pump(seconds=0.3)
        self.assertEqual(theme.COLORS["bg"], theme.PALETTES["light"]["bg"])
        self.assertEqual(w.bar.cget("background"), theme.PALETTES["light"]["panel"])   # okno przebudowane
        self.assertIn("Pójdzie", w.verdict.cget("text"))                   # wynik zachowany
        saved = json.loads((Path(self.tmp.name) / "cfg" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["theme"], "light")
        self.assertEqual(w.theme_var.get(), "light")

    def test_theme_without_ttkbootstrap(self):
        """Uruchomienie z kodu bez ttkbootstrap – zwykły ttk („clam”) z naszą paletą."""
        import sys
        from unittest import mock
        from aibobr.gui import theme
        with mock.patch.dict(sys.modules, {"ttkbootstrap": None, "ttkbootstrap.style": None}):
            self.assertEqual(theme.apply_style(self.root, "dark"), "dark")
        from tkinter import ttk
        self.assertEqual(ttk.Style(self.root).theme_use(), "clam")
        self.assertEqual(self.root.cget("background"), theme.PALETTES["dark"]["bg"])
        self.assertEqual(theme.apply_style(self.root, "nieznany"), "dark")   # zły wpis w ustawieniach

    def test_new_check_clears_old_result(self):
        from aibobr.i18n import t
        w = self.window()
        self.assertTrue(w.header.cget("text"))                           # wynik pierwszego modelu
        w.check(str(self.gguf))
        # Od razu, zanim nowy model się wczyta: bez starego wyniku, z komunikatem wczytywania.
        self.assertEqual(w.header.cget("text"), "")
        self.assertEqual(w.verdict.cget("text"), t("gui.reading_file"))
        self.assertIsNone(w.bar.verdict)
        self.pump(lambda: w.model is not None and w.verdict.cget("text") != t("gui.reading_file"), 5)
        self.assertIn("Pójdzie", w.verdict.cget("text"))

    def test_failed_check_does_not_leave_old_result(self):
        w = self.window()
        w.check(str(Path(self.tmp.name) / "nie-ma-takiego.gguf"))
        self.pump(lambda: w.loading is None, 5)
        self.assertEqual(w.header.cget("text"), "")
        self.assertIsNone(w.bar.verdict)
        self.assertNotIn("Pójdzie", w.verdict.cget("text"))

    def test_save_for_comparison_and_open_from_stats(self):
        w = self.window()
        self.assertFalse(w.save_button.instate(["disabled"]))
        w.save_compare()
        self.assertIn("Statystyka", w.status_bar.cget("text"))
        section = w.stats.sections["text"]
        self.assertEqual(len(section.items), 1)
        saved = next(iter(section.items.values()))
        self.assertEqual(saved.target, str(self.gguf))
        self.assertIn("(1/10)", section.toggle.cget("text"))
        self.assertIn("(1/10)", w.stats.filters["text"].cget("text"))          # przycisk filtra
        self.assertIn("(0/10)", w.stats.filters["image"].cget("text"))
        w.save_compare()                                                 # ten sam model – odświeżenie
        self.assertIn("odświeżyłem", w.status_bar.cget("text"))
        self.assertEqual(len(section.items), 1)
        # Otwórz z innymi ustawieniami: kontekst 32k → pole kontekstu w Sprawdź i nowy wynik
        from dataclasses import replace
        w.open_saved(replace(saved, context=32768))
        self.assertEqual(w.tab, "check")
        self.assertEqual(w.run_options.context.get(), "32k")
        self.pump(lambda: w.model is not None and w.verdict.cget("text") and w.loading is None, 5)
        self.assertIn("32k", w.header.cget("text").replace(" ", ""))
        w.stats.kind.set("image")                                        # filtr: jedna tabela naraz
        w.stats._show_kind()
        self.assertFalse(section.winfo_manager())
        self.assertTrue(w.stats.sections["image"].winfo_manager())
        w.stats.remove(saved)
        self.assertEqual(len(section.items), 0)

    def test_stats_tab_compares_on_this_computer(self):
        w = self.window()
        w.save_compare()
        w.tabs.select(w.pages["stats"])
        self.pump(lambda: "Porównanie dla" in w.stats.status.cget("text"), 5)
        section = w.stats.sections["text"]
        row = section.tree.item(section.tree.get_children()[0], "values")
        self.assertEqual(row[3], "zmieści się")                          # Dziś
        self.assertEqual(row[4], "1. w karcie")                          # jeden model – bez ★
        self.assertEqual(row[9], "ten plik")
        self.assertIn("zielone: 1", section.summary.cget("text"))

    def test_benchmark_dialog_plan_run_and_results(self):
        from aibobr.bench.runner import PerfResult
        from aibobr.gui.dialogs.benchmark import BenchmarkDialog
        w = self.window()
        self.assertFalse(w.bench_button.winfo_ismapped())                # plik .gguf – bez benchmarku
        targets = ["ollama:llama3.1:8b", "ollama:qwen3:8b"]
        w.c.bench_predict = lambda tg, ctxs, hw, s: {(t_, c): ("red" if c == 32768 else "green")
                                                     for t_ in tg for c in ctxs}

        def fake_run(tg, ctxs, predicted, hw, report, stop, db=None):
            out = []
            for n, t_ in enumerate(tg):
                report({"n": n + 1, "total": len(tg), "model": t_[7:], "context": ctxs[0], "step": "warmup"})
                out.append(PerfResult(t_[7:], ctxs[0], pp_tps=800, ttft_ms=1200, tg_tps=30 + n * 10,
                                      gpu_share=1.0, predicted="green"))
            return out

        w.c.run_benchmark = fake_run
        asked = []
        hw, settings = w.hardware.current()
        d = BenchmarkDialog(self.root, w.c, w.worker, targets, hw, settings,
                            confirm=lambda title, text: asked.append(text) or True)
        self.pump(lambda: "Do zmierzenia" in d.plan.cget("text"), 3)
        self.assertIn("ok.", d.plan.cget("text"))
        self.assertIn("Kilka modeli", d.warning.cget("text"))               # ostrzeżenie autora
        d.ctx_vars[32768].set(True)
        d._update_plan()
        self.assertIn("Pominięte: 2", d.plan.cget("text"))                  # 32k nie zmieści się
        d.ctx_vars[32768].set(False)
        d.start()
        self.assertEqual(len(asked), 1)                                     # potwierdzenie przy kilku modelach
        self.pump(lambda: d.summary.cget("text"), 3)
        rows = [d.tree.item(i, "values") for i in d.tree.get_children()]
        self.assertEqual(rows[1][5], "40,0")                                # generowanie tok/s
        self.assertIn("qwen3:8b", d.summary.cget("text"))                   # najszybszy u Ciebie
        d.close()

    def test_save_disabled_without_result(self):
        w = self.window()
        w.check(str(Path(self.tmp.name) / "nie-ma.gguf"))
        self.pump(lambda: w.loading is None, 5)
        self.assertTrue(w.save_button.instate(["disabled"]))

    def test_fields_follow_model_kind(self):
        from tests.test_images import SDXL, safetensors
        w = self.window()
        self.assertIn("context", w.advanced.entries)                     # model tekstowy
        self.assertNotIn("resolution", w.advanced.entries)
        self.assertIn("context", w.run_options.widgets)                  # …także w Sprawdź
        path = safetensors(Path(self.tmp.name) / "sdxl.safetensors", SDXL)
        w.check(str(path))
        self.pump(lambda: "resolution" in w.advanced.entries, 5)
        self.assertNotIn("context", w.advanced.entries)                  # obrazy: bez kontekstu
        self.assertEqual(set(w.run_options.widgets), {"resolution", "batch"})
        w.advanced.context.set("zle")                                    # błąd w ukrytym polu…
        self.pump(seconds=0.6)
        self.assertNotIn("Popraw", w.adv_result.verdict.cget("text"))    # …nie blokuje wyniku

    def test_meters_show_share_of_hardware(self):
        from aibobr.gui.widgets.meters import meter_values
        w = self.window()
        values = meter_values(w.meters.verdict)
        self.assertEqual(values["vram"][0], 100 * w.meters.verdict.in_vram / w.meters.verdict.capacity.vram_usable)
        self.assertIn("z 7,0 GB", values["vram"][1])
        w.run_options.context.set("128k")                                # nie zmieści się
        self.pump(lambda: w.meters.verdict is not None and w.meters.verdict.level.value == "red", 3)
        self.assertIn("brakuje", meter_values(w.meters.verdict)["ram"][1])

    def test_narrow_window_stacks_columns(self):
        from types import SimpleNamespace
        from aibobr.gui.widgets.scroll_page import BREAKPOINT
        w = self.window()
        page = w.check_page
        page._resize(SimpleNamespace(width=BREAKPOINT + 180))       # ekran serwera CI bywa mały
        self.assertTrue(page.wide)
        self.assertEqual(int(page.right.grid_info()["column"]), 1)
        page._resize(SimpleNamespace(width=BREAKPOINT - 200))       # laptop 1366×768 przy 125%
        self.assertFalse(page.wide)
        self.assertEqual(int(page.right.grid_info()["row"]), 1)     # prawa kolumna pod lewą

    def test_ollama_dialog_lists_models(self):
        from aibobr.gui.dialogs.ollama_models import OllamaModelsDialog
        from aibobr.gui.worker import Worker
        models = [{"name": "qwen3:8b", "size": 5_225_000_000, "details": {"quantization_level": "Q4_K_M"}},
                  {"name": "llama3.2:3b", "size": 2_019_000_000, "details": {"quantization_level": "Q4_K_M"}}]
        d = OllamaModelsDialog(self.root, Worker(self.root), lambda: models)
        self.pump(lambda: d.listbox.size() == 2, 3)
        self.assertIn("llama3.2:3b", d.listbox.get(0))           # alfabetycznie
        d._ok()
        self.assertEqual(d.result, "llama3.2:3b")

    def test_ollama_dialog_when_ollama_is_down(self):
        from aibobr.gui.dialogs.ollama_models import OllamaModelsDialog
        from aibobr.gui.worker import Worker
        from aibobr.sources.base import SourceError

        def down():
            raise SourceError("ollama.not_running", "http://127.0.0.1:11434")
        d = OllamaModelsDialog(self.root, Worker(self.root), down)
        self.pump(lambda: "Ollama nie odpowiada" in d.status.cget("text"), 3)
        self.assertIn("Ollama nie odpowiada", d.status.cget("text"))
        d.destroy()

    def test_found_models_dialog_asks_first(self):
        from aibobr.gui.dialogs.found_models import FoundModelsDialog
        from aibobr.gui.worker import Worker
        from aibobr.scan.model import FoundModel, Kind, ScanReport
        saved, calls = [], []

        def find(folder, stop, progress):
            calls.append(folder)
            progress(42)
            return ScanReport(models=[
                FoundModel("sdxl", Kind.IMAGE, "safetensors", 6 * 1024**3, Path("/m/sdxl.safetensors"), "folder"),
                FoundModel("llama3.1:8b", Kind.TEXT, "GGUF", 4 * 1024**3, Path("/m/blob"), "Ollama",
                           quant="Q4_K_M", target="ollama:llama3.1:8b")])
        d = FoundModelsDialog(self.root, Worker(self.root), find, False, saved.append)
        self.pump(seconds=0.2)
        self.assertEqual(calls, [])                              # bez zgody nic nie szukamy
        d.consent_yes.invoke()
        self.assertEqual(saved, [True])
        self.pump(lambda: len(d.tree.get_children()) == 2, 3)
        self.assertEqual(d.tree.item(d.tree.get_children()[0], "text"), "llama3.1:8b · Q4_K_M")
        self.assertIn("Znaleziono 2 modeli", d.status.cget("text"))
        d._ok()
        self.assertEqual(d.result, "ollama:llama3.1:8b")

    def test_found_button_in_both_modes(self):
        w = self.window()
        self.assertTrue(w.picker.found_button.winfo_ismapped())

    def test_failed_detection_offers_manual_mode(self):
        w = self.window("detection_failed.json")
        self.assertIn("Nie udało się odczytać", w.verdict.cget("text"))
        button = w.notes.winfo_children()[0]
        button.invoke()                                        # → zakładka Zaawansowane
        self.pump(seconds=0.5)
        self.assertEqual(w.tab, "advanced")
        w.advanced.rows[0].vram.set("24")
        v = w.adv_result.verdict
        self.pump(lambda: "Zapomnij" in v.cget("text") or "Pójdzie" in v.cget("text"), 3)
        self.assertIn("Pójdzie", v.cget("text"))

    def test_bad_file_shows_error_in_picker(self):
        w = self.window(model=Path(self.tmp.name) / "brak.gguf")
        self.pump(lambda: "Nie ma takiego pliku" in w.status_bar.cget("text"), 3)
        self.assertIn("Nie ma takiego pliku", w.status_bar.cget("text"))
        self.assertIsNone(w.model)

    def test_about_dialog_shows_picture(self):
        from aibobr.gui.dialogs.about import AboutDialog
        d = AboutDialog(self.root)
        self.pump(seconds=0.2)
        self.assertEqual(d.image.width(), 520)
        d.destroy()

    def test_worker_pauses_gc_while_threads_run(self):
        import gc
        import threading
        from aibobr.gui.worker import Worker
        release, results = threading.Event(), []
        w = Worker(self.root)
        w.run("x", lambda: release.wait(5) and "old", results.append, results.append)
        w.run("x", lambda: "new", results.append, results.append)        # unieważnia pierwsze
        self.pump(lambda: results == ["new"], 3)
        self.assertEqual(results, ["new"])
        self.assertFalse(gc.isenabled())                # pierwszy wątek wciąż pracuje
        release.set()
        self.pump(lambda: w.running == 0, 3)
        self.assertTrue(gc.isenabled())
        self.assertEqual(results, ["new"])              # spóźniony wynik pominięty

    def test_manual_dialog(self):
        from aibobr.gui.dialogs.manual_model import ManualModelDialog
        c = self.build("two_nvidia.json")
        d = ManualModelDialog(self.root, c.quants)
        d.vars["params"].set("abc")
        d._ok()
        self.assertIsNone(d.result)
        self.assertTrue(d.error.cget("text"))
        d.vars["params"].set("8,5")
        d.vars["layers"].set("32")
        d._ok()
        self.assertIsNone(d.result)                             # warstwy bez głowic i wymiaru
        for key, value in (("kv_heads", "8"), ("head_dim", "128")):
            d.vars[key].set(value)
        d._ok()
        self.assertEqual(d.result.spec().params_total, 8_500_000_000)


if __name__ == "__main__":
    unittest.main()
