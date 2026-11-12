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
        boxes = dict(section.summary_data(list(section.items.values())))
        self.assertEqual(boxes["Statystyka działania"][0][1], "1")          # zielone: 1
        self.assertEqual(len(section.summary.winfo_children()), 3)          # trzy ramki pod tabelą

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

    def speed_run(self, model, run_no, tps=60.0, ttft8k=4350.0):
        from aibobr.bench import speed
        from tests.test_speed import task
        tasks = [task(n, gen_tokens=200, eval_ms=200 / tps * 1000, ttft_wall_ms=50.0 + n) for n in range(1, 16)]
        tasks[0].response_text = "Zażółć gęślą jaźń"
        summary = speed.summarize(tasks)
        warnings = summary.pop("warnings")
        longs = [{"num_ctx": 1024, "prompt_tokens": 769, "gen_tokens": 128, "pp_tps": 1849.6, "tg_tps": 55.9,
                  "ttft_wall_ms": 430.0, "valid": 1},
                 {"num_ctx": 8192, "prompt_tokens": 7238, "gen_tokens": 128, "pp_tps": 1667.5, "tg_tps": 45.2,
                  "ttft_wall_ms": ttft8k, "valid": 1}]
        return {"id": f"r{run_no}", "model": model, "run_no": run_no, **summary, "warnings_json": warnings,
                "thresholds_json": speed.DEFAULT_THRESHOLDS, "stabilization": "stable", "cooling_seconds": 146.0,
                "baseline_json": {"samples": 10, "mean": {"cpu_pct": 7.3, "vram_used_mb": 91.0}},
                "tasks": tasks, "long_context": longs}

    def test_benchmark_speed_run_monitor_and_answers(self):
        from aibobr.bench.telemetry import Cooling
        w = self.window()
        b = w.bench
        targets = ["ollama:llama3.1:8b", "ollama:qwen3:8b"]
        w.c.bench_predict = lambda tg, ctxs, hw, s: {(t_, c): ("red" if t_.endswith("qwen3:8b") else "green")
                                                     for t_ in tg for c in ctxs}
        calls = []

        def fake_run(run, hw, report, stop, runs=1, contexts=(), on_timeout=None, skip_contexts=None):
            calls.append((run, runs, contexts, on_timeout))
            report({"event": "s_start", "session_id": "abc12345", "models": ["llama3.1:8b"], "runs": runs,
                    "tasks": 15, "contexts": contexts, "version": "0.40.15"})
            out = []
            for n in range(1, runs + 1):
                report({"event": "s_release", "model": "llama3.1:8b", "run": n, "released": True, "recovered": True})
                report({"event": "s_cooling", "key": "bench.cooling.start", "window": 30, "tolerance": 1.0})
                report({"event": "telemetry", "sample": {"cpu_pct": 7.0, "cpu_cores_json": [5.0, 90.0],
                                                         "gpu_temp_c": 43.0, "cpu_temp_c": None,
                                                         "vram_used_mb": 91.0, "vram_total_mb": 8151.0,
                                                         "gpu_throttle": "sw_power_cap"}})
                report({"event": "s_cooled", "model": "llama3.1:8b", "run": n,
                        "cooling": Cooling("stable", 146.0, 56.0, 43.0)})
                report({"event": "s_baseline", "model": "llama3.1:8b", "run": n,
                        "baseline": {"mean": {"cpu_pct": 7.3, "gpu_util_pct": 0.0, "vram_used_mb": 91.0}}})
                r = self.speed_run("llama3.1:8b", n, tps=60.0 - n)
                report({"event": "s_task", "model": "llama3.1:8b", "n": 1, "total": 15, "task": "pl_diacritics_restore",
                        "prompt": "Przywróć znaki…"})
                report({"event": "chunk", "text": "Zażółć"})
                report({"event": "s_task_done", "model": "llama3.1:8b", "n": 1, "total": 15, "result": r["tasks"][0]})
                report({"event": "s_long_done", "model": "llama3.1:8b", "run": n, "result": r["long_context"][1]})
                report({"event": "s_run_done", "model": "llama3.1:8b", "run": n, "runs": runs, "result": r})
                out.append(r)
            report({"event": "s_done", "status": "finished", "session_id": "abc12345"})
            return {"session_id": "abc12345", "runs": out, "status": "finished"}

        w.c.run_speed_benchmark = fake_run
        asked = []
        b.confirm = lambda title, text: asked.append(text) or True
        b.select_targets(targets)
        self.pump(lambda: all((t_, 1024) in b.predicted for t_ in targets), 3)
        b.runs.set(3)
        b._update_plan()
        self.assertIn("Pominięte (nie zmieszczą się): 1", b.plan.cget("text"))
        self.assertEqual(len(b.table.get_children()), 4)                     # 3 przebiegi + pominięty
        b.start()
        self.assertEqual(asked, [])                                          # jeden model – bez pytania
        self.pump(lambda: not b.running, 5)
        run, runs, contexts, on_timeout = calls[0]
        self.assertEqual((run, runs, contexts), (["ollama:llama3.1:8b"], 3, [1024, 8192]))
        self.assertIsNotNone(on_timeout)                                     # okno stabilizacji z wątku
        rows = {b.table.item(i, "values")[1]: b.table.item(i, "values") for i in b.table.get_children()}
        col = {c: i for i, c in enumerate(b.table["columns"])}
        first = rows["1"]
        self.assertEqual(first[col["verdict"]], "🟢 Szybko")
        self.assertEqual(first[col["gen"]], "59,0 tok/s z 3 000 tokenów w 50,8 s")
        self.assertEqual(first[col["pp"]], "1 850 tok/s (wejście 769 tokenów)")
        self.assertEqual(first[col["l8192"]], "45,2 tok/s · TTFT 4,35 s 🟡")        # rzeczywisty TTFT, ocena info
        self.assertEqual(first[col["l32768"]], "—")
        self.assertIn("57,0–59,0 tok/s", b.summary.cget("text"))             # rozrzut 3 przebiegów
        log = "\n".join(b.log_lines)
        self.assertIn("Ollama 0.40.15", log)
        self.assertIn("qwen3:8b: pominięty", log)
        self.assertIn("nie dowód stabilności", log)
        self.assertIn("Stabilizacja: ustabilizowana po 146 s", log)
        m = b.monitor                                                        # osobne okno (decyzja 4a)
        self.assertIn("Zażółć", m.content())
        self.assertIn("limit mocy", m.throttle.cget("text"))
        self.assertIn("Pomiar aktywności swap niedostępny", m.values.cget("text"))
        m.withdraw()                                                         # zamknięcie tylko chowa
        self.assertTrue(m.winfo_exists())
        b.show_monitor()                                                     # „Monitor: Przywróć”
        self.assertEqual(m.state(), "normal")
        self.assertEqual(len(b.answers.get_children()), 45)                  # 3 × 15 odpowiedzi
        first_answer = b.answers.get_children()[0]
        b.answers.selection_set(first_answer)
        b._show_answer()
        self.assertIn("Zażółć gęślą jaźń", b.answer_text.get("1.0", "end"))
        b.table.selection_set(b.job_rows[("llama3.1:8b", 1)])
        b.show_details()
        text = b.details_window.winfo_children()[0].get("1.0", "end")
        self.assertIn("Obie miary", text)
        self.assertIn("informacyjnie", text)
        self.assertIn("Prompt w pakiecie", text)
        b.details_window.destroy()

    def test_benchmark_missing_1k_and_export_session(self):
        import aibobr.gui.widgets.bench_panel as bp
        from aibobr.gui import speed_format as sf
        w = self.window()
        b = w.bench
        r = self.speed_run("m", 1)
        r["long_context"] = [{"num_ctx": 1024, "valid": 0, "invalid_reason": "possible_truncation"}]
        self.assertEqual(sf.prompt_main(r), "Pomiar 1k niedostępny")              # bez podstawiania pakietu
        self.assertEqual(sf.long_cell(r, 1024, {"gen_tps": [30, 15, 6], "ttft_s": [1, 3, 8]}),
                         "Brak danych (możliwe obcięcie wejścia)")
        info = []
        old_info, old_ask = bp.messagebox.showinfo, bp.filedialog.askdirectory
        d = tempfile.mkdtemp()
        bp.messagebox.showinfo = lambda title, text, parent=None: info.append(text)
        bp.filedialog.askdirectory = lambda **kw: d
        try:
            b.export_session()                                               # nic jeszcze nie zmierzono
            self.assertEqual(len(info), 1)
            db = w.c.bench_db
            b.session_id = db.start_session(started_at="2026-11-09T14:32:00+01:00")
            rid = db.start_run(b.session_id, model="m", run_no=1, started_at="2026-11-09T14:33:00+01:00",
                               gen_tps=54.21, verdict="fast", thresholds_json={"gen_tps": [30, 15, 6]})
            db.add_task(rid, task_no=1, task_id="t", response_text="odp", eval_ms=1234.5678)
            b.export_session()
            folder = next(Path(d).iterdir())
            self.assertEqual(folder.name, f"benchmark_2026-11-09_1432_{b.session_id[:8]}")
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             ["dlugi_kontekst.csv", "manifest.json", "telemetria.csv", "werdykt.csv", "zadania.csv"])
            import json
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["export_schema_version"], 1)
            self.assertEqual(manifest["runs"][0]["thresholds"], {"gen_tps": [30, 15, 6]})
            self.assertIn("1234,5678", (folder / "zadania.csv").read_text(encoding="utf-8-sig"))   # bez zaokrągleń
        finally:
            bp.messagebox.showinfo, bp.filedialog.askdirectory = old_info, old_ask

    def test_benchmark_history_tab(self):
        import aibobr.gui.widgets.bench_panel  # noqa: F401
        import aibobr.gui.widgets.history_view as hv
        from aibobr.bench.runner import PerfResult
        from tests.test_history import add_run
        w = self.window()
        b = w.bench
        db = w.c.bench_db
        db.add_perf(PerfResult("stary", 8192, pp_tps=1000.5, tg_tps=41.2), measured_at="2026-11-04 10:00",
                    runtime="ollama", runtime_version="0.40.14", hardware="hw", power="ac", suite="perf_v1")
        old = db.start_session(started_at="2026-11-08T10:00:00+01:00", runs_per_model=1, status="finished",
                               models_json=["gemma"])
        add_run(db, old, "gemma", 1, 31.0)
        sid = db.start_session(started_at="2026-11-09T14:00:00+01:00", runs_per_model=2, status="finished",
                               models_json=["minitron"])
        add_run(db, sid, "minitron", 1, 61.2)
        add_run(db, sid, "minitron", 2, 59.8, warnings=("limited_basis",))
        b.results_tabs.select(b.history)                                    # otwarcie zakładki = odświeżenie
        self.pump(seconds=0.2)
        h = b.history
        col = {c: i for i, c in enumerate(h.table["columns"])}
        rows = [h.table.item(i, "values") for i in h.table.get_children()]
        self.assertEqual([r[col["model"]] for r in rows], ["minitron", "gemma"])
        self.assertEqual(rows[0][col["verdict"]], "🟢 Szybko")
        self.assertEqual(rows[0][col["gen"]], "59,8–61,2 tok/s")
        self.assertIn("ograniczona podstawa", rows[0][col["notes"]])
        self.assertEqual(rows[1][col["now"]], "")
        w.c.settings["bench_thresholds"] = {"gen_tps": [40, 20, 8]}         # zaostrzone progi
        h.current.set(True)
        h.refresh()
        rows = [h.table.item(i, "values") for i in h.table.get_children()]
        self.assertEqual((rows[0][col["now"]], rows[1][col["now"]]), ("", "🔵 Użytecznie"))
        self.assertEqual(rows[1][col["verdict"]], "🟢 Szybko")             # werdykt z dnia testu bez zmian
        h.table.selection_set(h.table.get_children()[1])
        h.details()
        text = h.details_window.winfo_children()[0].get("1.0", "end")
        self.assertIn("Według obecnych progów: 🔵 Użytecznie", text)
        h.details_window.destroy()
        h.model.set("gemma")
        h.refresh()
        self.assertEqual(len(h.table.get_children()), 1)
        self.assertEqual(h.archive.item(h.archive.get_children()[0], "values")[1], "stary")   # archiwum
        h.archive_kind.set("quality")
        h.fill_archive()
        self.assertEqual(h.archive.get_children(), ())
        asked = []
        h.confirm = lambda title, text: asked.append(text) or True
        h.table.selection_set(h.table.get_children()[0])
        h.delete()
        self.assertIn("2026-11-08 10:00", asked[0])
        self.assertIsNone(db.session(old))
        self.assertEqual(db.count(), 1)                                     # archiwum nietknięte
        d = tempfile.mkdtemp()
        old_ask = hv.filedialog.askdirectory
        hv.filedialog.askdirectory = lambda **kw: d
        try:
            h.model.set(h.model_box["values"][0])
            h.refresh()
            h.table.selection_set(h.table.get_children()[0])
            h.export()
            h.export_archive()
        finally:
            hv.filedialog.askdirectory = old_ask
        names = sorted(p.name for p in Path(d).iterdir())
        self.assertEqual(len(names), 2)
        self.assertTrue(names[0].startswith("archiwum_przed_0.9_") and names[1].startswith("benchmark_2026-11-09"))

    def test_quality_tab_run_grade_history(self):
        import aibobr.gui.widgets.quality_panel as qp
        from aibobr.quality import suites
        from tests.test_quality_runner import Bench, NoSandbox, by_prompt
        w = self.window()
        q = w.quality
        h, sci = suites.standard()
        answers = {**by_prompt(h), **by_prompt(sci)}
        answers[h["tasks"][1]["prompt"]["pl"]] = "On nielubi niedobrych ludzi."            # H2 – błąd
        original = w.c.run_quality
        clock = [0.0]

        def fake(targets, tests, report, stop, attempts=1, hw=None, sandbox=None):
            return original(targets, tests, report, stop, attempts=attempts, bench=Bench(lambda p, n: answers[p]),
                            sandbox=NoSandbox(), runner_kw={"clock": lambda: clock[0],
                                                            "sleep": lambda x: clock.__setitem__(0, clock[0] + x),
                                                            "unload_timeout": 1})
        w.c.run_quality = fake
        q._sandbox_obj = NoSandbox()
        q.select_targets(["ollama:llama3.1:8b"])
        self.assertIn("zadania: 45", q.plan.cget("text"))
        self.assertIn("ocenisz go ręcznie", q.sandbox_line.cget("text"))
        q.confirm = lambda title, text: (self.assertIn("trzy razy dłużej", text), False)[1]
        q.attempts.set(3)
        q._three()
        self.assertEqual(q.attempts.get(), 1)                                   # „Wróć do jednej próby”
        q.start()
        self.pump(lambda: not q.running, 15)
        rows = [q.rtable.item(i, "values") for i in q.rtable.get_children()]
        self.assertEqual([r[1] for r in rows], ["Test humanistyczny (PL)", "Test ścisły (PL)", "Test ścisły (EN)"])
        self.assertEqual(rows[0][2], "14/15 (93%)")
        self.assertEqual((rows[1][2], rows[1][3]), ("12/12 (100%)", "pokrycie 12/15 – wynik częściowy"))
        self.assertIn("W tych testach podobnie", q.label_line.cget("text").replace("wypadł ", ""))
        self.assertIn("wynik częściowy", q.label_line.cget("text"))
        live = q.live.get("1.0", "end")
        self.assertIn("Tłumaczenie pytania (model go nie widzi): Oblicz 17 × 23.", live)
        q.rtable.selection_set(q.rtable.get_children()[0])
        q._show_breakdown()
        self.assertIn("Ortografia", q.breakdown.get("1.0", "end"))
        self.assertIn("2/3", q.breakdown.get("1.0", "end"))
        py = next(i for i, row in q.task_rows.items() if row[3].task_id == "S13" and row[2] == "pl")
        q.ttable.selection_set(py)
        q._show_task()
        self.assertIn("!disabled", str(q.pass_button.state()) + "!disabled")
        self.assertNotIn("disabled", q.pass_button.state())
        q.grade(True)                                                           # ocena ręczna kodu
        rows = [q.rtable.item(i, "values") for i in q.rtable.get_children()]
        sci_pl = next(r for r in rows if r[1] == "Test ścisły (PL)")
        self.assertEqual((sci_pl[2], sci_pl[3]), ("13/13 (100%)", "pokrycie 13/15 – wynik częściowy"))
        q.results_tabs.select(q.history_tab)
        self.pump(seconds=0.2)
        self.assertEqual(len(q.htable.get_children()), 1)
        d = tempfile.mkdtemp()
        old_ask = qp.filedialog.askdirectory
        qp.filedialog.askdirectory = lambda **kw: d
        try:
            q.htable.selection_set(q.htable.get_children()[0])
            q.export_history()
        finally:
            qp.filedialog.askdirectory = old_ask
        folder = next(Path(d).iterdir())
        self.assertTrue(folder.name.startswith("jakosc_"))
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["manifest.json", "wyniki.csv", "zadania.csv"])
        q.history_details()
        self.assertIn("H2", q.details_window.winfo_children()[0].get("1.0", "end"))
        q.details_window.destroy()
        q.confirm = lambda title, text: True
        q.htable.selection_set(q.htable.get_children()[0])
        q.delete_history()
        self.assertEqual(q.htable.get_children(), ())

    def test_diagnostics_tab(self):
        from aibobr.quality import custom
        from tests.test_diagnostics_bos import FakeOllama as BosOllama
        from tests.test_diagnostics_fragility import TIE, FakeOllama as FragOllama, STABLE
        w = self.window()
        names = [w.tabs.tab(i, "text") for i in range(len(w.tabs.tabs()))]
        self.assertEqual(names[names.index("Jakość") + 1], "Diagnostyka")          # między Jakością a Statystyką
        self.assertEqual(names[names.index("Diagnostyka") + 1], "Statystyka")
        menu = w.settings_menu
        entry = next(i for i in range(menu.index("end") + 1)
                     if menu.type(i) == "command" and menu.entrycget(i, "label") == "Diagnostyka")
        menu.invoke(entry)                                     # Ustawienia → Diagnostyka
        self.assertEqual(w.tabs.select(), str(w.diag))
        d = w.diag
        clock = [0.0]
        kw = {"clock": lambda: clock[0], "sleep": lambda s: clock.__setitem__(0, clock[0] + s), "unload_timeout": 1}
        bos_fake = BosOllama(expects=False, chat="{{bos_token}}{% for m in messages %}")      # jak Minitron
        original_bos, original_frag = w.c.check_bos, w.c.run_fragility
        w.c.check_bos = lambda targets, report, stop, force=False: original_bos(
            targets, report, stop, force=force, bench=bos_fake, probe_kw=kw)
        d.select_targets(["ollama:minitron"])
        self.assertIn("Wybrane modele: 1", d.plan.cget("text"))
        d.check_bos(False)
        self.pump(lambda: not d.running, 10)
        row = d.btable.item(d.btable.get_children()[0], "values")
        self.assertEqual((row[0], row[1], row[2], row[3], row[5]), ("minitron", "⚠ BRAK tokenu początku", "tak", "nie", "0"))
        d.btable.selection_set(d.btable.get_children()[0])
        self.pump(seconds=0.1)
        text = d.bos_text.get("1.0", "end")
        self.assertIn("Ollama go nie podaje", text)
        self.assertIn("szablon czatu z pliku GGUF podaje bos_token: tak", text)
        probes = len(bos_fake.prompts)
        d.check_bos(False)                                     # ta sama wersja pliku – zapamiętany wynik
        self.pump(lambda: not d.running, 10)
        self.assertEqual(len(bos_fake.prompts), probes)
        self.assertIn("z zapamiętanych wyników: 1", d.stage.cget("text"))
        d.check_bos(True)                                      # „Sprawdź ponownie”
        self.pump(lambda: not d.running, 10)
        self.assertGreater(len(bos_fake.prompts), probes)
        # kruchość na własnym teście
        data = custom.new_suite("diag_test", "Diag", ["pl"])
        data["tasks"] = [
            {"id": "T1", "category": "x", "level": "easy", "max_tokens": 64, "prompt": {"pl": "Stolica Francji?"},
             "check": {"type": "text", "expected": ["Paryż"]}},
            {"id": "T2", "category": "x", "level": "easy", "max_tokens": 64, "prompt": {"pl": "Popraw znaki"},
             "check": {"type": "text", "expected": ["Zażółć gęślą"]}}]
        w.c.custom_suites.save(data)
        frag_fake = FragOllama({"Stolica Francji?": STABLE, "Popraw znaki": TIE})
        w.c.run_fragility = lambda target, test, report, stop, thresholds=None: original_frag(
            target, test, report, stop, thresholds=thresholds, bench=frag_fake, probe_kw=kw)
        d.tabs.select(1)
        self.pump(seconds=0.1)
        d.f_test.set("Diag (PL)")
        d.f_fragile.set("0,15")
        d.f_tie.set("0.3")                                     # remis > kruche – błąd, nic nie rusza
        d.run_fragility()
        self.assertFalse(d.running)
        self.assertIn("remis nie większy niż kruche", d.f_note.cget("text"))
        d.f_tie.set("0,02")
        d.run_fragility()
        self.pump(lambda: not d.running, 10)
        self.assertEqual((w.c.settings["fragile_threshold"], w.c.settings["tie_threshold"]), (0.15, 0.02))
        rows = [d.ftable.item(i, "values") for i in d.ftable.get_children()]
        self.assertEqual([(r[0], r[1]) for r in rows], [("T1", "stabilne"), ("T2", "remis")])
        d.ftable.selection_set(d.ftable.get_children()[1])
        self.pump(seconds=0.1)
        self.assertIn("Pierwszy prawie remis", d.f_text.get("1.0", "end"))
        self.assertIn("'żółć' 0.410", d.f_text.get("1.0", "end"))
        # historia i znacznik w zakładce Jakość
        d.tabs.select(2)
        self.pump(seconds=0.1)
        hrow = d.htable.item(d.htable.get_children()[0], "values")
        self.assertEqual((hrow[1], hrow[3], hrow[5]), ("minitron", "stabilne 1 · kruche 0 · remis 1",
                                                       "⚠ BRAK tokenu początku"))
        q = w.quality
        q._suites_changed()
        self.assertEqual(q._fragility("minitron", "diag_test", "pl", "T2"), "remis")
        self.assertEqual(q._fragility("minitron", "diag_test", "pl", "T9"), "")
        d.htable.selection_set(d.htable.get_children()[0])
        d.confirm = lambda title, text: True
        d.delete_history()
        self.assertEqual(d.htable.get_children(), ())

    def test_bos_gate_both_comparison(self):
        """Pkt 5 (12.11.2026): brak tokenu początku → pytanie; „oba” – poprawiona wersja, dwa przebiegi, porównanie
        zmienionych zadań z opisem z diagnostyki kruchości; z jakim tokenem liczono – w bazie."""
        from aibobr.diagnostics.fragility import TaskFragility
        from aibobr.quality import custom
        from tests.test_diagnostics_bos import FakeOllamaWithCreate
        from tests.test_quality_runner import Bench, NoSandbox
        w = self.window()
        q = w.quality
        q._sandbox_obj = NoSandbox()
        clock = [0.0]
        kw = {"clock": lambda: clock[0], "sleep": lambda s: clock.__setitem__(0, clock[0] + s), "unload_timeout": 1}
        fake = FakeOllamaWithCreate(expects=False, chat="{{bos_token}}{% for m in messages %}")
        orig_check, orig_create, orig_quality = w.c.check_bos, w.c.create_bos_variant, w.c.run_quality

        def check(targets, report=lambda e: None, stop=lambda: False, force=False, **_):
            return orig_check(targets, report, stop, force=force, bench=fake, probe_kw=kw)
        w.c.check_bos = check
        w.c.create_bos_variant = lambda target, c, report=lambda e: None: orig_create(target, c, report, bench=fake,
                                                                                      probe_kw=kw)
        data = custom.new_suite("bos_test", "Bos", ["pl"])
        data["tasks"] = [
            {"id": "T1", "category": "x", "level": "easy", "max_tokens": 64, "prompt": {"pl": "Ile to 6 × 7?"},
             "check": {"type": "number", "expected": 42}},
            {"id": "T2", "category": "x", "level": "easy", "max_tokens": 64, "prompt": {"pl": "Popraw: Zazolc"},
             "check": {"type": "text", "expected": ["Zażółć"]}}]
        w.c.custom_suites.save(data)
        db = w.c.bench_db
        ds = db.start_diag_session(started_at="2026-11-12T10:00:00+01:00", model="minitron", suite="bos_test",
                                   suite_version="1", lang="pl", fragile_threshold=0.1, tie_threshold=0.02)
        db.add_diag_task(ds, 1, TaskFragility("T1", "tie"))
        db.add_diag_task(ds, 2, TaskFragility("T2", "stable"))
        db.finish_diag_session(ds, finished_at="x")
        q._suites_changed()
        for key, var in q.test_vars.items():
            var.set(key[0] == "bos_test")

        class ModelBench(Bench):
            def generate(self, model, prompt, *a, **k):
                self.model = model
                return super().generate(model, prompt, *a, **k)
        answers = {("minitron", "Ile to 6 × 7?"): "41", ("minitron", "Popraw: Zazolc"): "Zazółć",
                   ("minitron-bos", "Ile to 6 × 7?"): "42", ("minitron-bos", "Popraw: Zazolc"): "Zażółć"}
        bench = ModelBench(lambda p, n: answers[(bench.model, p)])
        w.c.run_quality = lambda targets, tests, report, stop, attempts=1, hw=None, sandbox=None: orig_quality(
            targets, tests, report, stop, attempts=attempts, bench=bench, sandbox=NoSandbox(),
            runner_kw={"clock": lambda: clock[0], "sleep": lambda x: clock.__setitem__(0, clock[0] + x),
                       "unload_timeout": 1})
        asked = []
        q.ask_bos = lambda checks, variants: (asked.append((checks, variants)), "both")[1]
        q.select_targets(["ollama:minitron"])
        q.start()
        self.assertTrue(q.running)                                      # sprawdzenie w tle – przycisk zablokowany
        self.pump(lambda: q.q_results and not q.running, 15)
        self.assertEqual([c.status for c in asked[0][0]], ["missing"])
        self.assertEqual(asked[0][1], {"minitron": "minitron-bos"})
        self.assertEqual(fake.created, [("minitron-bos", "minitron")])
        self.assertEqual(fake.models["minitron-bos"][:3], "<s>")         # szablon z tokenem początku
        self.assertEqual(sorted(q.targets()), ["ollama:minitron", "ollama:minitron-bos"])
        label = q.label_line.cget("text")
        self.assertIn("Token początku – minitron, Bos (PL): jak podaje Ollama 0/2, poprawiony 2/2", label)
        self.assertIn("T1 ✗→✓ (przeważony remis)", label)
        self.assertIn("T2 ✗→✓ (rzeczywista różnica wejścia)", label)
        runs = db.quality_runs(q.q_session)
        self.assertEqual({r["model"]: r["bos_status"] for r in runs}, {"minitron": "missing", "minitron-bos": "ok"})
        # drugi raz: wersja już jest – bez tworzenia; „jak podaje Ollama” – tylko oryginał
        q.ask_bos = lambda checks, variants: "as_is"
        q.select_targets(["ollama:minitron"])
        q.start()
        self.pump(lambda: not q.running and len(db.quality_sessions()) == 2, 15)
        self.assertEqual(fake.created, [("minitron-bos", "minitron")])
        # anulowanie – nic się nie uruchamia
        q.ask_bos = lambda checks, variants: None
        q.start()
        self.pump(lambda: not q.running, 5)
        self.assertEqual(len(db.quality_sessions()), 2)

    def test_quality_custom_test_run(self):
        from aibobr.quality import custom
        from tests.test_quality_runner import Bench, NoSandbox
        w = self.window()
        q = w.quality
        q._sandbox_obj = NoSandbox()
        data = custom.new_suite("moj_test", "Mój test", ["ko", "en"])
        data["tasks"] = [
            {"id": "T1", "category": "arytmetyka", "level": "easy", "max_tokens": 64,
             "prompt": {"ko": "7 곱하기 6은? 숫자만.", "en": "7 times 6? Number only."},
             "check": {"type": "number", "expected": 42}},
            {"id": "T2", "category": "logika", "level": "medium", "max_tokens": 64,
             "prompt": {"ko": "7은 소수입니까? 예/아니요", "en": "Is 7 prime? yes/no"},
             "check": {"type": "choice", "expected": {"ko": "예", "en": "yes"},
                       "options": {"ko": ["예", "아니요"], "en": ["yes", "no"]}}}]
        w.c.custom_suites.save(data)
        q._suites_changed()                                   # jak po zapisie w edytorze
        self.assertIn(("moj_test", "ko"), q.test_vars)
        self.assertFalse(q.test_vars[("moj_test", "ko")].get())         # własne – domyślnie niezaznaczone
        for key, var in q.test_vars.items():
            var.set(key[0] == "moj_test")
        answers = {"7 곱하기 6은? 숫자만.": "42", "7은 소수입니까? 예/아니요": "아니요",
                   "7 times 6? Number only.": "42", "Is 7 prime? yes/no": "yes"}
        original = w.c.run_quality
        clock = [0.0]

        def fake(targets, tests, report, stop, attempts=1, hw=None, sandbox=None):
            return original(targets, tests, report, stop, attempts=attempts, bench=Bench(lambda p, n: answers[p]),
                            sandbox=NoSandbox(), runner_kw={"clock": lambda: clock[0],
                                                            "sleep": lambda x: clock.__setitem__(0, clock[0] + x),
                                                            "unload_timeout": 1})
        w.c.run_quality = fake
        q.select_targets(["ollama:llama3.1:8b"])
        self.assertIn("zadania: 4", q.plan.cget("text"))
        self.assertEqual(q.sandbox_line.cget("text"), "")                # bez zadań w Pythonie
        q.start()
        self.pump(lambda: not q.running, 15)
        rows = [q.rtable.item(i, "values") for i in q.rtable.get_children()]
        self.assertEqual([(r[1], r[2]) for r in rows], [("Mój test (KO)", "1/2 (50%)"), ("Mój test (EN)", "2/2 (100%)")])
        self.assertIn("Mój test: KO 1/2, EN 2/2", q.label_line.cget("text"))
        live = q.live.get("1.0", "end")
        self.assertIn("Tłumaczenie pytania (model go nie widzi): 7은 소수입니까? 예/아니요", live)
        runs = w.c.bench_db.quality_runs(q.q_session)
        self.assertEqual({(r["suite"], r["suite_version"], r["lang"]) for r in runs},
                         {("moj_test", "1", "ko"), ("moj_test", "1", "en")})
        self.assertTrue(w.c.bench_db.quality_suite_used("moj_test", "1"))

    def test_quality_test_editor(self):
        import json
        from tests.test_quality_runner import NoSandbox
        w = self.window()
        q = w.quality
        q._sandbox_obj = NoSandbox()
        menu = w.settings_menu
        entry = next(i for i in range(menu.index("end") + 1)
                     if menu.type(i) == "command" and menu.entrycget(i, "label") == "Edytor testów…")
        menu.invoke(entry)                                   # Ustawienia → Edytor testów…
        ed = q.editor
        q.open_editor()
        self.assertIs(q.editor, ed)                          # jedno okno naraz
        self.pump(seconds=0.2)
        folder = Path(self.tmp.name) / "cfg" / "quality"
        # testy standardowe: tylko do odczytu, podgląd oceny działa
        rows = {v: k for k, v in ed.list_rows.items()}
        ed.tests.selection_set(rows[("std", "scisly_v1")])
        self.pump(lambda: ed.data is not None, 2)
        self.assertTrue(ed.readonly)
        self.assertIn("disabled", ed.save_button.state())
        self.assertIn("tylko do odczytu", ed.info.cget("text"))
        ed.preview_answer.insert("1.0", "17 × 23 = 391")
        ed.run_preview()
        self.assertEqual(ed.preview_result.cget("text"),
                         "✓ poprawnie, ale nie zachowano wymaganego formatu odpowiedzi")
        # nowy test: błędy blokują zapis i wskazują zadanie i pole
        ed.ask_name = lambda title, initial: "Mój test"
        ed.new_suite()
        self.assertEqual((ed.data["suite_info"]["id"], len(ed.data["tasks"])), ("moj_test", 1))
        self.assertFalse(ed.save())
        texts = ed.problem_list.get(0, "end")
        self.assertIn("T1 · Kategoria – brak wartości", texts)
        self.assertIn("T1 · Treść zadania (PL) – brak treści zadania", texts)
        self.assertFalse(folder.exists() and any(folder.iterdir()))
        # wypełnienie: liczba
        ed.inputs["category"][0].set("rachunki")
        ed.inputs["prompt.pl"][1].insert("1.0", "Ile to 6 × 7? Podaj tylko liczbę.")
        ed.inputs["type"][0].set("liczba")
        ed._change_type()
        ed.inputs["expected"][0].set("42")
        self.assertTrue(ed.save())
        saved = json.loads((folder / "moj_test.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["tasks"][0]["check"], {"type": "number", "expected": 42})
        self.assertIn("Zapisano (wersja 1)", ed.status.cget("text"))
        # drugie zadanie: Python z tabelą przypadków, błąd składni wiersza
        ed.add_task()
        self.assertEqual(ed.data["tasks"][1]["id"], "T2")
        ed.inputs["category"][0].set("kod")
        ed.inputs["prompt.pl"][1].insert("1.0", "Napisz add(a, b).")
        ed.inputs["type"][0].set("kod Pythona")
        ed._change_type()
        ed.inputs["function"][0].set("add")
        ed.inputs["tests"][1].insert("1.0", "[1, 2] -> 3\n[0, 0] => 0")
        self.assertFalse(ed.save())
        self.assertIn("T2 · Przypadki testowe – nie da się odczytać wiersza – zapis: [argumenty] -> wynik, w JSON",
                      ed.problem_list.get(0, "end"))
        ed.inputs["tests"][1].delete("1.0", "end")
        ed.inputs["tests"][1].insert("1.0", "[1, 2] -> 3\n[\"a\", \"b\"] -> \"ab\"")
        ed.preview_answer.delete("1.0", "end")
        ed.preview_answer.insert("1.0", "def add(a, b):\n    return a + b")
        ed.run_preview()
        self.assertIn("nieoceniony", ed.preview_result.cget("text"))           # bez piaskownicy – nie uruchamia
        # angielski: drugi język dochodzi do treści i do błędów
        self.assertEqual(ed.main_lang.get(), "polski (PL)")
        ed.english_var.set(True)
        ed._toggle_english()
        self.assertEqual(ed.data["suite_info"]["languages"], ["pl", "en"])
        self.assertFalse(ed.save())
        self.assertIn("T1 · Treść zadania (EN) – brak treści zadania", ed.problem_list.get(0, "end"))
        # wariant B: jedno pole treści i przełącznik wersji; druga wersja zostaje w danych
        self.assertIn("prompt.pl", ed.inputs)
        self.assertNotIn("prompt.en", ed.inputs)
        ed.form_lang.set("en")
        ed._switch_form_lang()
        self.assertEqual((list(ed.inputs).count("prompt.en"), "prompt.pl" in ed.inputs), (1, False))
        ed.inputs["prompt.en"][1].insert("1.0", "Write add(a, b).")
        ed.form_lang.set("pl")
        ed._switch_form_lang()
        self.assertEqual(ed.inputs["prompt.pl"][1].get("1.0", "end-1c"), "Napisz add(a, b).")
        self.assertEqual(ed.data["tasks"][1]["prompt"], {"pl": "Napisz add(a, b).", "en": "Write add(a, b)."})
        # język główny: koreański – treść przechodzi z polskiego, angielski zostaje
        ed.main_lang.set("koreański (KO)")
        ed._change_main_lang()
        self.assertEqual(ed.data["suite_info"]["languages"], ["ko", "en"])
        self.assertEqual(ed.data["tasks"][1]["prompt"], {"ko": "Napisz add(a, b).", "en": "Write add(a, b)."})
        ed.main_lang.set("polski (PL)")
        ed._change_main_lang()
        ed.english_var.set(False)
        ed._toggle_english()
        self.assertEqual(ed.data["tasks"][1]["prompt"], {"pl": "Napisz add(a, b)."})
        q.running = True                                                     # w trakcie pomiaru – bez zapisu
        messages = []
        ed.inform = messages.append
        self.assertFalse(ed.save())
        self.assertIn("Trwa test jakości", messages[0])
        q.running = False
        self.assertTrue(ed.save())
        saved = json.loads((folder / "moj_test.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["tasks"][1]["check"]["tests"][1], {"args": ["a", "b"], "expected": "ab"})
        # wersja rośnie tylko, gdy ta wersja ma już wyniki
        db = w.c.bench_db
        qs = db.start_quality_session(started_at="2026-11-11T10:00:00+01:00", attempts=1)
        db.start_quality_run(qs, model="m", suite="moj_test", suite_version="1", lang="pl", started_at="x")
        ed.store = w.c.custom_suites
        ed.inputs["function"][0].set("add2")
        self.assertTrue(ed.save())
        self.assertIn("Zapisano (wersja 2)", ed.status.cget("text"))
        # eksport, import jako kopia, kopia testu standardowego
        out = Path(self.tmp.name) / "eksport.json"
        ed.export_suite(str(out))
        ed.ask_replace = lambda name: False
        ed.import_suite(str(out))
        self.assertEqual(ed.saved_id, "moj_test_2")
        ed.tests.selection_set(next(k for k, v in ed.list_rows.items() if v == ("std", "humanistyczny_v1")))
        self.pump(lambda: ed.readonly, 2)
        ed.ask_name = lambda title, initial: initial
        ed.copy_suite()
        self.assertEqual(ed.data["suite_info"]["id"], "kopia_test_humanistyczny")
        self.assertEqual(len(ed.data["tasks"]), 15)
        self.assertTrue(ed.save())
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         ["kopia_test_humanistyczny.json", "moj_test.json", "moj_test_2.json"])
        # niezapisane zmiany przy zamykaniu: „Anuluj” zostawia okno
        ed.inputs["category"][0].set("inna")
        ed.ask_save = lambda: None
        ed.close()
        self.assertTrue(ed.winfo_exists())
        ed.ask_save = lambda: False
        ed.close()
        self.pump(seconds=0.1)
        self.assertFalse(ed.winfo_exists())

    def test_benchmark_tab_finds_models_and_remembers(self):
        from aibobr.scan.model import FoundModel, Kind, ScanReport
        calls = []

        def find(folder, stop, progress):
            calls.append(folder)
            progress(42)
            report = ScanReport(models=[
                FoundModel("sdxl", Kind.IMAGE, "safetensors", 6 * 1024**3, Path("/m/sdxl.safetensors"), "folder"),
                FoundModel("llama3.1:8b", Kind.TEXT, "GGUF", 4 * 1024**3, Path("/m/blob"), "Ollama",
                           quant="Q4_K_M", target="ollama:llama3.1:8b")])
            w.c.save_found(report)
            return report
        w = self.window()
        w.c.find_models = find
        b = w.bench
        self.assertEqual(calls, [])                              # bez kliknięcia nic nie szukamy
        b.search()                                               # kliknięcie = zgoda
        self.assertTrue(w.c.scan_consent)
        self.pump(lambda: len(b.tree.get_children()) == 2, 3)
        first = b.tree.get_children()[0]
        self.assertEqual(b.tree.item(first, "values")[1], "llama3.1:8b · Q4_K_M")
        self.assertEqual(b.tree.item(first, "values")[0], "☐")
        self.assertEqual(b.tree.item(b.tree.get_children()[1], "values")[0], "")   # obraz – bez pola
        b._toggle(first)
        self.assertEqual(b.targets(), ["ollama:llama3.1:8b"])
        self.assertIn("Znaleziono 2 modeli", b.scan_status.cget("text"))
        opened = []
        w.check_from_disk = opened.append
        b.on_check = opened.append
        b.tree.selection_set(first)
        b._check()
        self.assertEqual(opened, ["ollama:llama3.1:8b"])
        # Nowe okno: lista z pliku, bez szukania
        w.change_language("pl")
        self.pump(seconds=0.3)
        self.assertEqual(len(w.bench.tree.get_children()), 2)
        self.assertIn("Lista z", w.bench.scan_status.cget("text"))

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
