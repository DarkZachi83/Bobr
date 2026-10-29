"""Szukanie modeli na dysku (etap 9a): znane miejsca, magazyny programów, rozpoznanie plików."""
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main
from aibobr.cli.bar import PLAIN
from aibobr.cli.cli import Cli
from aibobr.i18n import set_language
from aibobr.scan import ollama_store
from aibobr.scan.files import kind_from_keys
from aibobr.scan.locations import known_locations
from aibobr.scan.model import Kind
from aibobr.scan.scanner import Scanner
from aibobr.sources.base import SourceError
from aibobr.sources.factory import load_model
from tests.fake_http import FakeHttp
from tests.gguf_writer import gguf_bytes, llama_like
from tests.sizes import fake_sizes

GB = 1024**3
ROOT = Path(__file__).resolve().parents[1]


def safetensors(path: Path, keys, size: int) -> Path:
    header = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    _sizes[path.name] = size
    return path


_sizes: dict = {}


def write_model(path: Path, header: bytes, size: int) -> Path:
    path.write_bytes(header)
    _sizes[path.name] = size
    return path


class ScanTreeTest(unittest.TestCase):
    """Udawany dysk z modelami z kilku programów."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _sizes.clear()
        r = self.root

        # Ollama: manifest + blob (GGUF bez rozszerzenia)
        store = r / "ollama"
        (store / "blobs").mkdir(parents=True)
        digest = "sha256:" + "ab" * 32
        write_model(store / "blobs" / ("sha256-" + "ab" * 32), llama_like(name="Meta Llama 3.1 8B"),
                    4_920_000_000)
        manifest = store / "manifests" / "registry.ollama.ai" / "library" / "llama3.1" / "8b"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": 4_920_000_000},
            {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:" + "cd" * 32,
             "size": 1400}]}))
        hf_manifest = store / "manifests" / "hf.co" / "bartowski" / "Qwen-GGUF" / "Q4_K_M"
        hf_manifest.parent.mkdir(parents=True)
        hf_manifest.write_text(json.dumps({"layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "ee" * 32,
             "size": 5}]}))                                       # blobu brak – pomijamy

        # LM Studio: zwykły GGUF, plik dzielony (2 części), projektor obrazu
        lm = r / "lmstudio" / "bartowski" / "Gemma-GGUF"
        lm.mkdir(parents=True)
        write_model(lm / "gemma-Q4_K_M.gguf", llama_like(name="Gemma Test"), 2_500_000_000)
        write_model(lm / "big-Q4_K_M-00001-of-00002.gguf", llama_like(name="Big Split"), 40 * GB)
        write_model(lm / "big-Q4_K_M-00002-of-00002.gguf", b"GGUF", 30 * GB)
        write_model(lm / "mmproj-gemma-f16.gguf",
                    gguf_bytes([("general.architecture", "clip", "str")], []), 800_000_000)
        write_model(lm / "tiny.gguf", llama_like(name="Tiny"), 10_000_000)     # za mały

        # Pamięć podręczna Hugging Face: repozytorium transformers
        snap = r / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / "abc123"
        safetensors(snap / "model.safetensors", ["model.embed_tokens.weight", "model.layers.0.mlp.weight"],
                    1_500_000_000)
        (snap / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 28, "num_attention_heads": 16,
            "num_key_value_heads": 8, "hidden_size": 1024, "head_dim": 128, "vocab_size": 151936,
            "torch_dtype": "bfloat16", "tie_word_embeddings": True}))

        # Stable Diffusion WebUI
        sd = r / "stable-diffusion-webui" / "models"
        safetensors(sd / "Stable-diffusion" / "sdxl_base.safetensors",
                    ["model.diffusion_model.input_blocks.0.weight", "conditioner.x"], 6_900_000_000)
        safetensors(sd / "Lora" / "style.safetensors", ["lora_unet_down_blocks_0.alpha"], 150_000_000)
        (sd / "Stable-diffusion" / "old.ckpt").write_bytes(b"\x80\x02pickle")
        _sizes["old.ckpt"] = 4_200_000_000
        diff = r / "diffusers" / "sd15"
        (diff / "unet").mkdir(parents=True)
        (diff / "model_index.json").write_text("{}")
        safetensors(diff / "unet" / "diffusion_pytorch_model.safetensors", ["down_blocks.0.x"],
                    3_400_000_000)

        # Pomijane – przypadki z dysku autora (29.10.2026):
        # Forge/SD.Next trzymają „szkielety” diffusers – same JSON-y, bez wag,
        skeleton = r / "forge" / "backend" / "huggingface" / "stabilityai" / "sdxl"
        (skeleton / "unet").mkdir(parents=True)
        (skeleton / "model_index.json").write_text("{}")
        (skeleton / "unet" / "config.json").write_text("{}")
        # pamięć shaderów gry (*.bin) to nie model,
        steam = r / "SteamLibrary" / "steamapps" / "shadercache" / "steamapp_merged_shader_cache.bin"
        steam.parent.mkdir(parents=True)
        steam.write_bytes(b"x")
        _sizes[steam.name] = 120_000_000
        # Stability Matrix: Data/Models/StableDiffusion (bez myślnika).
        sm = r / "Data" / "Models" / "StableDiffusion" / "dreamshaper.ckpt"
        sm.parent.mkdir(parents=True)
        sm.write_bytes(b"\x80\x02")
        _sizes[sm.name] = 2_100_000_000

        # Przerwane pobrania (dysk autora: 6 × „Unconfirmed …smdownload”, 14,8 GB)
        for path, size in ((sm.parent / "Unconfirmed 5348838.smdownload", 3_657_792_256),
                           (store / "blobs" / ("sha256-" + "77" * 32 + "-partial"), 900_000_000),
                           (snap.parent.parent / "blobs" / ("aa" * 32 + ".incomplete"), 700_000_000)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            _sizes[path.name] = size

        # Ukryty folder i zwykłe śmieci – pomijane
        safetensors(r / "lmstudio" / ".trash" / "hidden.safetensors", ["model.layers.0.x"], 2 * GB)

        patch = fake_sizes(_sizes)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self):
        report = Scanner().scan_folder(self.root)
        return report, {m.name: m for m in report.models}

    def test_everything_found_once(self):
        report, by_name = self.scan()
        self.assertEqual(sorted(by_name), sorted([
            "llama3.1:8b", "Gemma Test", "Big Split", "mmproj-gemma-f16",
            "Qwen/Qwen3-0.6B", "sdxl_base", "style", "old", "sd15", "dreamshaper",
            "Unconfirmed 5348838.smdownload", "sha256-" + "77" * 32 + "-partial",
            "Qwen/Qwen3-0.6B · " + "aa" * 32 + ".incomplete"]))
        self.assertEqual(len(report.models), len(by_name))

    def test_ollama_names_from_manifests(self):
        _, m = self.scan()
        llama = m["llama3.1:8b"]
        self.assertEqual((llama.app, llama.kind, llama.quant), ("Ollama", Kind.TEXT, "Q4_K_M"))
        self.assertEqual(llama.target, "ollama:llama3.1:8b")
        self.assertEqual(llama.size, 4_920_001_400)                    # wszystkie warstwy

    def test_gguf_split_and_projector(self):
        _, m = self.scan()
        self.assertEqual(m["Big Split"].size, 70 * GB)
        projector = next(x for x in m.values() if x.note == "scan.note.projector")
        self.assertEqual(projector.kind, Kind.ADDON)
        self.assertFalse(projector.checkable)
        self.assertTrue(m["Gemma Test"].target.endswith("gemma-Q4_K_M.gguf"))

    def test_hf_cache_repo_is_one_checkable_entry(self):
        _, m = self.scan()
        qwen = m["Qwen/Qwen3-0.6B"]
        self.assertEqual((qwen.fmt, qwen.kind, qwen.app), ("transformers", Kind.TEXT, "Hugging Face"))
        model = load_model(qwen.target, FakeHttp())
        self.assertEqual(model.name, "Qwen/Qwen3-0.6B")
        self.assertEqual(model.arch.n_layers, 28)
        self.assertEqual(model.arch.params_total, 750_000_000)         # 1,5 GB / 2 bajty
        self.assertEqual(model.spec("Q4_K_M").quant, "Q4_K_M")

    def test_image_models_lora_and_pickles(self):
        _, m = self.scan()
        self.assertEqual(m["sdxl_base"].kind, Kind.IMAGE)
        self.assertEqual((m["style"].kind, m["style"].note), (Kind.ADDON, "scan.note.lora"))
        self.assertEqual((m["old"].kind, m["old"].fmt), (Kind.IMAGE, "ckpt"))   # z nazwy folderu
        self.assertEqual((m["sd15"].kind, m["sd15"].fmt), (Kind.IMAGE, "diffusers"))
        self.assertEqual(m["sd15"].size, 3_400_000_002)
        self.assertFalse(any(x.checkable for x in (m["sdxl_base"], m["style"], m["old"], m["sd15"])))

    def test_embedding_model_is_labelled(self):
        # Dysk autora: Open WebUI trzyma all-MiniLM-L6-v2 do wyszukiwania w dokumentach.
        repo = self.root / "webui_data" / "models--sentence-transformers--all-MiniLM-L6-v2"
        snap = repo / "snapshots" / "1110a2"
        safetensors(snap / "minilm.safetensors", ["embeddings.word_embeddings.weight"], 90_000_000)
        (snap / "config.json").write_text(json.dumps({"architectures": ["BertModel"]}))
        (snap / "modules.json").write_text("[]")
        patch = fake_sizes(_sizes)
        patch.start()
        self.addCleanup(patch.stop)
        _, m = self.scan()
        mini = m["sentence-transformers/all-MiniLM-L6-v2"]
        self.assertEqual((mini.kind, mini.note, mini.checkable), (Kind.OTHER, "scan.note.embedding", False))

    def test_unfinished_downloads(self):
        report, m = self.scan()
        partials = [x for x in report.models if x.note == "scan.note.partial"]
        self.assertEqual(sorted(x.app for x in partials), ["Hugging Face", "Ollama", "folder"])
        self.assertFalse(any(x.checkable for x in partials))
        self.assertEqual(report.partial_size, 3_657_792_256 + 900_000_000 + 700_000_000)

    def test_pickle_kind_from_folder_only(self):
        _, m = self.scan()
        self.assertEqual((m["dreamshaper"].kind, m["dreamshaper"].fmt), (Kind.IMAGE, "ckpt"))
        self.assertNotIn("steamapp_merged_shader_cache", m)
        self.assertNotIn("sdxl", m)                             # szkielet bez wag

    def test_second_copy_is_marked(self):
        other = self.root / "backup" / "ollama"
        (other / "blobs").mkdir(parents=True)
        write_model(other / "blobs" / ("sha256-" + "99" * 32), llama_like(name="Meta Llama 3.1 8B"),
                    4_920_000_000)
        manifest = other / "manifests" / "registry.ollama.ai" / "library" / "llama3.1" / "8b"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model",
                                                    "digest": "sha256:" + "99" * 32, "size": 4_920_001_400}]}))
        patch = fake_sizes(_sizes)
        patch.start()
        self.addCleanup(patch.stop)
        report, _ = self.scan()
        copies = [x for x in report.with_copies() if x.name == "llama3.1:8b"]
        self.assertEqual([x.note for x in copies], ["scan.note.copy", "scan.note.copy"])   # obie

    def test_same_file_twice_counts_once(self):
        link = self.root / "copy-of-gemma.gguf"
        try:
            os.link(self.root / "lmstudio" / "bartowski" / "Gemma-GGUF" / "gemma-Q4_K_M.gguf", link)
        except OSError:
            self.skipTest("brak dowiązań twardych")
        report, _ = self.scan()
        self.assertEqual(sum(1 for x in report.models if x.name == "Gemma Test"), 1)

    def test_stop(self):
        scanner = Scanner()
        scanner.stop.set()
        report = scanner.scan_folder(self.root)
        self.assertTrue(report.cancelled)
        self.assertEqual(report.models, [])

    def test_ollama_fallback_when_not_running(self):
        class Down(FakeHttp):
            def get_json(self, url):
                raise SourceError("source.network", url)
        with mock.patch("aibobr.scan.locations.known_locations",
                        return_value=[("Ollama", self.root / "ollama")]):
            model = load_model("ollama:llama3.1:8b", Down())
            self.assertEqual(model.name, "llama3.1:8b")
            self.assertEqual(model.origin, "ollama:llama3.1:8b")
            with self.assertRaises(SourceError) as e:
                load_model("ollama:qwen3:8b", Down())
            self.assertEqual(e.exception.code, "ollama.not_running")
        self.assertIsNotNone(ollama_store.find_blob("llama3.1:8b", [self.root / "ollama"]))

    def test_cli_scan_folder(self):
        set_language("pl")
        self.addCleanup(set_language, "en")
        c = main.build_app(ROOT / "tests" / "fake_hardware" / "real_laptop_rtx5050_linux.json",
                           user_dir=self.root / "cfg", cache_dir=self.root / "cache")
        out = io.StringIO()
        args = main.build_parser().parse_args(["--scan", str(self.root)])
        code = Cli(c, out=out, palette=PLAIN).run(args)
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Znaleziono 13 modeli", text)
        self.assertIn("Przerwane pobrania zajmują 4,9 GB", text)
        self.assertIn(f"{self.root / 'ollama'}  (ollama:llama3.1:8b)", text)   # folder, nie sha256
        self.assertIn("ollama:llama3.1:8b", text)
        self.assertIn("obraz", text)
        self.assertLess(text.index("llama3.1:8b"), text.index("sdxl_base"))   # tekstowe najpierw


class FilesTest(unittest.TestCase):
    def test_kind_from_keys(self):
        self.assertEqual(kind_from_keys(["double_blocks.0.img_attn.qkv.weight"])[0], Kind.IMAGE)   # Flux
        self.assertEqual(kind_from_keys(["encoder.down.0.x", "decoder.up.0.x"]), (Kind.ADDON, "scan.note.vae"))
        self.assertEqual(kind_from_keys(["model.layers.0.self_attn.q_proj.weight"])[0], Kind.TEXT)
        self.assertEqual(kind_from_keys(["something.else"])[0], Kind.OTHER)

    def test_manifest_names(self):
        root = Path("/m/manifests")
        name = lambda *p: ollama_store.model_name(root.joinpath(*p), root)   # noqa: E731
        self.assertEqual(name("registry.ollama.ai", "library", "qwen3", "8b"), "qwen3:8b")
        self.assertEqual(name("registry.ollama.ai", "user", "model", "latest"), "user/model:latest")
        self.assertEqual(name("hf.co", "bartowski", "X-GGUF", "Q4_K_M"), "hf.co/bartowski/X-GGUF:Q4_K_M")


class LocationsTest(unittest.TestCase):
    def test_only_existing_and_env_first(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / ".ollama" / "models").mkdir(parents=True)
            (home / ".lmstudio" / "models").mkdir(parents=True)
            custom = home / "my-hf"
            custom.mkdir()
            places = known_locations("linux", env={"HF_HUB_CACHE": str(custom)}, home=home)
            self.assertEqual(places, [("Ollama", home / ".ollama" / "models"),
                                      ("LM Studio", home / ".lmstudio" / "models"),
                                      ("Hugging Face", custom)])

    def test_windows_paths(self):
        with tempfile.TemporaryDirectory() as d:
            home, local = Path(d) / "user", Path(d) / "local"
            (local / "nomic.ai" / "GPT4All").mkdir(parents=True)
            (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
            places = known_locations("win32", env={"LOCALAPPDATA": str(local)}, home=home)
            self.assertEqual([a for a, _ in places], ["Hugging Face", "GPT4All"])


class ConsentTest(unittest.TestCase):
    def test_consent_is_saved(self):
        with tempfile.TemporaryDirectory() as d:
            c = main.build_app(None, user_dir=Path(d), cache_dir=Path(d) / "cache")
            self.assertFalse(c.scan_consent)
            c.set_scan_consent(True)
            again = main.build_app(None, user_dir=Path(d), cache_dir=Path(d) / "cache")
            self.assertTrue(again.scan_consent)


if __name__ == "__main__":
    unittest.main()
