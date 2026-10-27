import io
import tempfile
import unittest
from pathlib import Path

import main
from aibobr.cli.bar import ANSI, PLAIN, render_bar
from aibobr.cli.cli import BOBR_MESSAGE, EXIT_HARDWARE, EXIT_OK, EXIT_SOURCE, Cli, parse_context, parse_gb_list
from aibobr.calculator.verdict import evaluate
from aibobr.domain.models import RunSettings
from aibobr.i18n import set_language
from tests.fake_http import FakeHttp
from tests.fixtures import DESKTOP_3060, LLAMA_31_8B, LLAMA_33_70B, QUANTS
from tests.gguf_writer import llama_like
from tests.sizes import stop_all, write_model
from tests.test_huggingface import gguf_fake

PROFILES = Path(__file__).resolve().parent / "fake_hardware"
LAPTOP = PROFILES / "real_laptop_rtx5050_linux.json"


class CliRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gguf = self.dir / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        write_model(self.gguf, llama_like(name="Meta Llama 3.1 8B Instruct"), 4_920_000_000)
        self.addCleanup(stop_all)
        set_language("pl")

    def tearDown(self):
        self.tmp.cleanup()
        set_language("en")

    def run_cli(self, *argv, profile=LAPTOP, palette=PLAIN, http=None):
        controller = main.build_app(profile, user_dir=self.dir / "cfg", cache_dir=self.dir / "cache")
        if http is not None:
            controller.http = lambda: http
        out = io.StringIO()
        args = main.build_parser().parse_args(list(argv))
        set_language(args.lang or "pl")                     # w main.main robi to main()
        code = Cli(controller, out=out, palette=palette).run(args)
        return code, out.getvalue()


class CliTest(CliRun):
    def test_laptop_green_with_low_margin(self):
        code, out = self.run_cli(str(self.gguf))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("Pójdzie na Twoim komputerze", out)
        self.assertIn("Mały zapas", out)
        self.assertIn("[=================", out)
        self.assertIn("--explain", out)                       # podpowiedź, bez wyjaśnienia
        self.assertNotIn("Model potrzebuje", out)

    def test_explain_sections_with_full_stop(self):
        code, out = self.run_cli(str(self.gguf), "--context", "16k", "--explain")
        self.assertIn("Model potrzebuje 7,1 GB:", out)
        self.assertIn("Twój komputer:", out)
        self.assertIn("Co możesz zrobić:", out)
        self.assertIn("  → cache KV w q8_0 (opcja runtime'u) zmieści się w całości w karcie", out)
        actions = out.split("Co możesz zrobić:")[1].strip().splitlines()
        self.assertTrue(all(a.endswith(",") for a in actions[:-1]))
        self.assertTrue(actions[-1].endswith("."))

    def test_bobr(self):
        code, out = self.run_cli(str(self.gguf), "--bobr", "--lang", "en",
                                 profile=PROFILES / "two_nvidia.json")
        self.assertIn("[BB", out)
        self.assertIn(BOBR_MESSAGE, out)                       # po polsku także w trybie EN
        self.assertIn("It will run", out)

    def test_bobr_is_yellow_with_colors_and_silent_when_not_green(self):
        _, out = self.run_cli(str(self.gguf), "--bobr", profile=PROFILES / "two_nvidia.json",
                              palette=ANSI)
        self.assertIn("\033[1;33mBB", out)
        _, out = self.run_cli(str(self.gguf), "--bobr", "--context", "16k")
        self.assertIn("[BBB", out)
        self.assertNotIn(BOBR_MESSAGE, out)

    def test_bobr_is_hidden_from_help(self):
        self.assertNotIn("bobr", main.build_parser().format_help())

    def test_plain_output_has_no_escape_codes(self):
        _, out = self.run_cli(str(self.gguf))
        self.assertNotIn("\033[", out)

    def test_missing_file(self):
        code, out = self.run_cli(str(self.dir / "nope.gguf"))
        self.assertEqual(code, EXIT_SOURCE)
        self.assertIn("Nie ma takiego pliku", out)

    def test_unknown_vram_then_manual_override(self):
        code, out = self.run_cli(str(self.gguf), profile=PROFILES / "detection_failed.json")
        self.assertEqual(code, EXIT_HARDWARE)
        self.assertIn("--vram", out)
        code, out = self.run_cli(str(self.gguf), "--vram", "8", profile=PROFILES / "detection_failed.json")
        self.assertEqual(code, EXIT_OK)

    def test_what_if_bigger_card(self):
        _, out = self.run_cli(str(self.gguf), "--context", "16k", "--vram", "16")
        self.assertIn("Pójdzie na Twoim komputerze", out)

    def test_huggingface_unknown_quant_lists_available(self):
        code, out = self.run_cli("hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "--quant", "Q9_K",
                                 http=gguf_fake())
        self.assertEqual(code, EXIT_SOURCE)
        self.assertIn("Q4_K_M", out)
        self.assertIn("Pobieram opis modelu", out)

    def test_huggingface_second_run_comes_from_cache(self):
        http = gguf_fake()
        self.run_cli("hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q8_0", http=http)
        calls = len(http.calls)
        code, out = self.run_cli("hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", http=http)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(len(http.calls), calls)               # bez sieci
        self.assertIn("Q4_K_M", out)


class BarTest(unittest.TestCase):
    def setUp(self):
        set_language("en")

    def test_red_bar_shows_missing_memory(self):
        v = evaluate(LLAMA_33_70B, DESKTOP_3060, RunSettings(), QUANTS)
        system, model, _ = render_bar(v, width=50)
        self.assertIn("x", system)
        self.assertEqual(len(system), len(model))

    def test_green_bar_ends_inside_vram(self):
        v = evaluate(LLAMA_31_8B, DESKTOP_3060, RunSettings(), QUANTS)
        system, model, legend = render_bar(v, width=50)
        end_of_model = len(model.rstrip(" ]"))
        end_of_vram = system.rindex("=") + 1
        self.assertLessEqual(end_of_model, end_of_vram)
        self.assertIn("VRAM 11.0 GB", legend)


class ParsingTest(unittest.TestCase):
    def test_context(self):
        self.assertEqual(parse_context("16k"), 16384)
        self.assertEqual(parse_context("12288"), 12288)
        for bad in ("0", "abc", "-5"):
            with self.assertRaises(ValueError):
                parse_context(bad)

    def test_gb_list(self):
        self.assertEqual(parse_gb_list("24,24"), [24.0, 24.0])
        self.assertEqual(parse_gb_list("12"), [12.0])


if __name__ == "__main__":
    unittest.main()
