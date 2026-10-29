"""Modele mowy i muzyki typu LLM (etap 9): rozpoznanie, dekoder, sekundy nagrania."""
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import main
from aibobr.calculator.memory import memory_need
from aibobr.calculator.verdict import evaluate
from aibobr.cli.bar import PLAIN
from aibobr.cli.cli import Cli
from aibobr.domain.enums import FitLevel, ModelSource
from aibobr.domain.models import ModelSpec, RunSettings
from aibobr.domain.units import gib
from aibobr.i18n import set_language
from aibobr.knowledge.advisor import Advisor, audio_label, fmt_duration, render
from aibobr.knowledge.audio import AudioCatalog
from aibobr.sources.base import ArchInfo, LoadedModel, Variant
from aibobr.sources.gguf_reader import load_gguf_file, token_pattern
from aibobr.sources.huggingface import HF, HfRef, HuggingFaceSource
from tests.fake_http import FakeHttp
from tests.fixtures import QUANTS, gpu, pc
from tests.gguf_writer import llama_like
from tests.sizes import stop_all, write_model

ROOT = Path(__file__).resolve().parents[1]
AUDIO = AudioCatalog.load(ROOT / "data" / "audio_models.json")
LAPTOP = ROOT / "tests" / "fake_hardware" / "real_laptop_rtx5050_linux.json"

# Słownik jak w Orpheusie: Llama 3.2 (128 256) + tokeny dźwięku <custom_token_N>
ORPHEUS_TOKENS = tuple([f"t{i}" for i in range(128_256)] + [f"<custom_token_{i}>" for i in range(28_682)])


def loaded(name="Model", pattern=None, task=None, origin=None) -> LoadedModel:
    return LoadedModel(name, ModelSource.GGUF_FILE, ArchInfo(token_pattern=pattern, task=task),
                       (Variant("Q4_K_M", gib(2)),), origin=origin)


class TokenPatternTest(unittest.TestCase):
    def test_audio_tokens(self):
        self.assertEqual(token_pattern(list(ORPHEUS_TOKENS)), "<custom_token_#>")
        oute = [f"w{i}" for i in range(2000)] + [f"<|c1_{i}|>" for i in range(1025)] \
            + [f"<|c2_{i}|>" for i in range(1025)] + ["<|features|>"] * 40
        self.assertEqual(token_pattern(oute), "<|c#_#|>")              # obie książki kodowe

    def test_ordinary_vocabulary(self):
        words = ["the", "Ġcat", "ing", "Ġ2024", "ĠPolska"] * 2000
        llama = words + [f"<|reserved_special_token_{i}|>" for i in range(248)]
        self.assertIsNone(token_pattern(llama))                     # 248 zastrzeżonych to za mało
        self.assertIsNone(token_pattern([]))


class RecognizeTest(unittest.TestCase):
    def test_tokens_first(self):
        a = AUDIO.recognize(loaded("Whatever", pattern="<custom_token_#>"))
        self.assertEqual((a.family, a.by, a.kind, a.decoder), ("orpheus", "tokens", "speech", "SNAC 24 kHz"))
        self.assertEqual(a.tokens_per_second, 83)
        self.assertEqual(AUDIO.recognize(loaded(pattern="<|c#_#|>")).family, "outetts")

    def test_tag_then_name(self):
        a = AUDIO.recognize(loaded("Orpheus 3B", task="text-to-speech"))
        self.assertEqual((a.family, a.by), ("orpheus", "tag"))
        b = AUDIO.recognize(loaded("Some Voice Model", task="text-to-speech"))
        self.assertEqual((b.family, b.by, b.kind, b.decoder_bytes), (None, "tag", "speech", 0))

    def test_name_only_as_last_resort(self):
        a = AUDIO.recognize(loaded("x", origin="/m/orpheus-3b-0.1-ft-q4_k_m.gguf"))
        self.assertEqual((a.family, a.by), ("orpheus", "name"))
        music = AUDIO.recognize(loaded("facebook/musicgen-small"))
        self.assertEqual((music.kind, music.by, music.family), ("music", "name", None))
        self.assertEqual(AUDIO.recognize(loaded("Dia-1.6B")).kind, "speech")

    def test_ordinary_models_and_folders(self):
        self.assertIsNone(AUDIO.recognize(loaded("Meta Llama 3.1 8B", pattern="t#")))
        self.assertIsNone(AUDIO.recognize(loaded("Qwen3 8B", origin="/home/u/tts/qwen3-8b.gguf")))
        self.assertIsNone(AUDIO.recognize(loaded("Media Llama")))               # „dia” w środku słowa

    def test_broken_file_is_harmless(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "a.json"
            bad.write_text("{nie json")
            self.assertIsNone(AudioCatalog.load(bad).recognize(loaded(pattern="<custom_token_#>")))


def orpheus_spec(**kw) -> ModelSpec:
    a = AUDIO.recognize(loaded(pattern="<custom_token_#>"))
    base = dict(name="Orpheus 3B", source=ModelSource.GGUF_FILE, quant="Q4_K_M",
                weights_bytes=gib(2.3), n_layers=28, n_kv_heads=8, head_dim=128,
                embd_params=156_938 * 3072, tied_embeddings=True, audio=a)
    base.update(kw)
    return ModelSpec(**base)


class MemoryTest(unittest.TestCase):
    def test_decoder_counts_on_the_card(self):
        n = memory_need(orpheus_spec(), RunSettings(), QUANTS)
        self.assertEqual(n.decoder, gib(0.5))
        self.assertIn(n.decoder, (n.total - n.weights_total - n.kv_cache - n.overhead,))
        self.assertEqual(n.vram_part, n.total - n.embd_in_ram)

    def test_decoder_on_cpu_moves_to_ram(self):
        n = memory_need(orpheus_spec(), RunSettings(decoder_on_gpu=False), QUANTS)
        self.assertEqual(n.vram_part, n.total - n.embd_in_ram - gib(0.5))
        self.assertEqual(n.ram_fixed, n.embd_in_ram + gib(0.5))

    def test_text_models_unchanged(self):
        self.assertEqual(memory_need(orpheus_spec(audio=None), RunSettings(), QUANTS).decoder, 0)

    def test_tip_moves_decoder_to_cpu(self):
        set_language("pl")
        self.addCleanup(set_language, "en")
        hw = pc(gpus=[gpu(vram_gb=5.0)])
        v = evaluate(orpheus_spec(), hw, RunSettings(), QUANTS)
        self.assertEqual(v.level, FitLevel.YELLOW)                  # z dekoderem nie mieści się
        model = LoadedModel("Orpheus 3B", ModelSource.GGUF_FILE, ArchInfo(),
                            (Variant("Q4_K_M", gib(2.3)),))
        _, e = Advisor(QUANTS).explain(model, orpheus_spec(), hw, RunSettings())
        first = e.actions[0]
        self.assertEqual(first.key, "expl.act.decoder_cpu")
        self.assertIn("zwolni 0,5 GB w karcie", render(first))
        self.assertIn("w całości w karcie", render(first))
        self.assertIn("note.audio_decoder_estimate", [n.key for n in e.notes])
        self.assertIn("expl.need.decoder", [n.key for n in e.need])


class LabelTest(unittest.TestCase):
    def setUp(self):
        set_language("pl")
        self.addCleanup(set_language, "en")

    def test_duration(self):
        self.assertEqual(fmt_duration(42.4), "42 s")
        self.assertEqual(fmt_duration(98.7), "1 min 39 s")

    def test_label(self):
        label = audio_label(orpheus_spec(), RunSettings())
        self.assertEqual(label, "model mowy Orpheus (dekoder SNAC 24 kHz) · ≈ 1 min 39 s nagrania")
        self.assertIsNone(audio_label(orpheus_spec(audio=None), RunSettings()))

    def test_name_note_uses_authors_wording(self):
        spec = orpheus_spec(audio=AUDIO.recognize(loaded("facebook/musicgen-small")))
        model = LoadedModel("m", ModelSource.GGUF_FILE, ArchInfo(), (Variant("Q4_K_M", gib(2.3)),))
        _, e = Advisor(QUANTS).explain(model, spec, pc(gpus=[gpu()]), RunSettings())
        notes = [render(n) for n in e.notes]
        self.assertIn("Rozpoznano z nazwy pliku – prawdopodobnie model muzyczny.", notes)
        self.assertTrue(any("dekodera dźwięku nie znamy" in n for n in notes))


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(stop_all)
        self.addCleanup(self.tmp.cleanup)
        set_language("pl")
        self.addCleanup(set_language, "en")

    def test_orpheus_gguf_file(self):
        path = write_model(self.dir / "model.gguf",
                           llama_like(name="Orpheus", layers=28, heads=24, embed=3072,
                                      tokens=ORPHEUS_TOKENS), 2_490_000_000)
        self.assertEqual(load_gguf_file(path).arch.token_pattern, "<custom_token_#>")
        c = main.build_app(LAPTOP, user_dir=self.dir / "cfg", cache_dir=self.dir / "cache")
        self.assertEqual(c.load_model(str(path)).spec().audio.family, "orpheus")
        out = io.StringIO()
        Cli(c, out=out, palette=PLAIN).run(main.build_parser().parse_args([str(path), "--explain"]))
        text = out.getvalue()
        self.assertIn("model mowy Orpheus (dekoder SNAC 24 kHz) · ≈ 1 min 39 s nagrania", text)
        self.assertIn("~ dekoder 0,5 GB", text)
        self.assertIn("0,5 GB dekoder dźwięku (SNAC 24 kHz),", text)
        self.assertIn("Pójdzie na Twoim komputerze", text)

    def test_hf_tag_reaches_the_model(self):
        repo = "someone/voice-GGUF"
        tree = f"{HF}/api/models/{repo}/tree/main?recursive=true"
        file = f"{HF}/{repo}/resolve/main/voice-Q4_K_M.gguf"
        http = FakeHttp(json={tree: [{"type": "file", "path": "voice-Q4_K_M.gguf", "size": 2_000_000_000}],
                              f"{HF}/api/models/{repo}": {"pipeline_tag": "text-to-speech"}},
                        files={file: llama_like(name="Voice")})
        m = HuggingFaceSource(http).load(HfRef(repo))
        self.assertEqual(m.arch.task, "text-to-speech")
        a = AUDIO.annotate(m).spec().audio
        self.assertEqual((a.kind, a.by), ("speech", "tag"))

    def test_ordinary_hf_repo_has_no_task(self):
        repo = "someone/text-GGUF"
        tree = f"{HF}/api/models/{repo}/tree/main?recursive=true"
        http = FakeHttp(json={tree: [{"type": "file", "path": "text-Q4_K_M.gguf", "size": 2_000_000_000}],
                              f"{HF}/api/models/{repo}": {"pipeline_tag": "text-generation"}},
                        files={f"{HF}/{repo}/resolve/main/text-Q4_K_M.gguf": llama_like(name="Text")})
        m = HuggingFaceSource(http).load(HfRef(repo))
        self.assertIsNone(m.arch.task)
        self.assertIsNone(AUDIO.annotate(m).spec().audio)


if __name__ == "__main__":
    unittest.main()
