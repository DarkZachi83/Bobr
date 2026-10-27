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
        self.root.destroy()
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
        w.hardware.show_advanced()
        self.pump(seconds=0.5)
        self.assertEqual(w.verdict.cget("text"), auto)          # samo przełączenie nic nie zmienia
        w.hardware.context.set("16k")
        self.pump(lambda: "wolniej" in w.verdict.cget("text"), 3)
        self.assertIn("nieco wolniej", w.verdict.cget("text"))
        w.explanation.toggle()
        self.assertIn("Co możesz zrobić", w.explanation.text.get("1.0", "end"))
        w.change_language("en")
        self.pump(seconds=0.5)
        self.assertIn("AI BEAVER", self.root.title())
        self.assertEqual(w.hardware.context.get(), "16k")         # stan zachowany
        self.assertIn("a bit slower", w.verdict.cget("text"))
        self.assertTrue(w.explanation.expanded)

    def test_failed_detection_offers_manual_mode(self):
        w = self.window("detection_failed.json")
        self.assertIn("Nie udało się odczytać", w.verdict.cget("text"))
        button = w.notes.winfo_children()[0]
        button.invoke()
        self.pump(seconds=0.5)
        self.assertEqual(w.hardware.mode.get(), "advanced")
        w.hardware.rows[0].vram.set("24")
        self.pump(lambda: "Zapomnij" in w.verdict.cget("text") or "Pójdzie" in w.verdict.cget("text"), 3)
        self.assertIn("Pójdzie", w.verdict.cget("text"))

    def test_bad_file_shows_error_in_picker(self):
        w = self.window(model=Path(self.tmp.name) / "brak.gguf")
        self.pump(lambda: w.picker.status.winfo_ismapped(), 3)
        self.assertIn("Nie ma takiego pliku", w.picker.status.cget("text"))
        self.assertIsNone(w.model)

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
