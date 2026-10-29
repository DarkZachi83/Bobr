"""Modele obrazów (etap 10, SPEC §5a): części, rodzina, szczyt etapów, światła, pasek."""
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import main
from aibobr.calculator.verdict import evaluate
from aibobr.cli.bar import PLAIN
from aibobr.cli.cli import Cli, parse_resolution
from aibobr.domain.enums import FitLevel
from aibobr.domain.models import RunSettings
from aibobr.domain.units import gib
from aibobr.gui.bar_layout import image_bar_layout
from aibobr.i18n import set_language
from aibobr.knowledge.advisor import Advisor, image_label, render
from aibobr.knowledge.images import ImageCatalog
from aibobr.sources.base import SourceError
from aibobr.sources.cache import model_from_dict, model_to_dict
from aibobr.sources.factory import load_model
from aibobr.sources.image import load_diffusers_folder, load_safetensors_file
from tests.fake_http import FakeHttp
from tests.fixtures import QUANTS, gpu, pc
from tests.gguf_writer import gguf_bytes
from tests.sizes import fake_sizes

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ImageCatalog.load(ROOT / "data" / "image_models.json")
LAPTOP = ROOT / "tests" / "fake_hardware" / "real_laptop_rtx5050_linux.json"
G = 1024**3


def safetensors(path: Path, groups, dtype="F16") -> Path:
    header, off = {}, 0
    for name, nbytes in groups:
        header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [off, off + int(nbytes)]}
        off += int(nbytes)
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


SDXL = [("model.diffusion_model.input_blocks.0.weight", 4.78 * G),
        ("conditioner.embedders.0.transformer.x", 0.23 * G),
        ("conditioner.embedders.1.model.x", 1.29 * G),
        ("first_stage_model.decoder.x", 0.16 * G)]
SD15 = [("model.diffusion_model.x", 1.6 * G), ("cond_stage_model.transformer.text_model.x", 0.23 * G),
        ("first_stage_model.x", 0.16 * G)]
FLUX = [("double_blocks.0.img_attn.qkv.weight", 11 * G), ("single_blocks.0.x", 0.1 * G),
        ("img_in.weight", 0.01 * G)]


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        set_language("pl")
        self.addCleanup(set_language, "en")

    def spec(self, groups, name="m", dtype="F16"):
        path = safetensors(self.dir / f"{name}.safetensors", groups, dtype)
        return IMAGES.annotate(load_safetensors_file(path)).spec()


class SourcesTest(Tmp):
    def test_parts_from_header_are_exact(self):
        m = load_safetensors_file(safetensors(self.dir / "illu.safetensors", SDXL))
        parts = dict(m.arch.image_parts)
        self.assertEqual(parts["main"], int(4.78 * G))
        self.assertEqual(parts["encoders"], int(0.23 * G) + int(1.29 * G))
        self.assertEqual(parts["vae"], int(0.16 * G))
        self.assertEqual(m.spec().quant, "FP16")

    def test_families(self):
        self.assertEqual(self.spec(SDXL, "a").image.family, "sdxl")
        self.assertEqual(self.spec(SD15, "b").image.native, 512)
        sd2 = [("model.diffusion_model.x", G), ("cond_stage_model.model.transformer.x", G)]
        self.assertEqual(self.spec(sd2, "c").image.family, "sd2")
        flux = self.spec(FLUX, "d", dtype="F8_E4M3")
        self.assertEqual((flux.image.family, flux.quant), ("flux", "FP8"))
        self.assertEqual(flux.image.assumed, ("encoders", "vae"))           # osobne pliki – z tabeli
        self.assertEqual(self.spec([("joint_blocks.0.x", 4 * G)], "e").image.family, "sd3")
        unknown = self.spec([("model.diffusion_model.x", G)], "f").image
        self.assertIsNone(unknown.family)

    def test_addons_and_text_weights(self):
        lora = safetensors(self.dir / "lora.safetensors", [("lora_unet_down_blocks_0.alpha", 1)])
        with self.assertRaises(SourceError) as e:
            load_safetensors_file(lora)
        self.assertEqual(e.exception.code, "source.image_addon")
        text = safetensors(self.dir / "llm.safetensors", [("model.layers.0.mlp.weight", 1)])
        with self.assertRaises(SourceError) as e:
            load_safetensors_file(text)
        self.assertEqual(e.exception.code, "source.text_safetensors")

    def test_diffusers_folder_prefers_fp16(self):
        from unittest import mock
        from aibobr.sources import gguf_reader
        d = self.dir / "sd15"
        files = {("unet", "diffusion_pytorch_model.safetensors"): 3_400_000_000,
                 ("unet", "diffusion_pytorch_model.fp16.safetensors"): 1_700_000_000,
                 ("text_encoder", "model.fp16.safetensors"): 246_000_000,
                 ("vae", "diffusion_pytorch_model.fp16.safetensors"): 167_000_000}
        for folder, name in files:
            (d / folder).mkdir(parents=True, exist_ok=True)
            (d / folder / name).write_bytes(b"x")
        (d / "model_index.json").write_text(json.dumps({
            "_class_name": "StableDiffusionPipeline", "unet": ["d", "U"], "text_encoder": ["t", "C"],
            "vae": ["d", "A"], "scheduler": ["d", "S"]}))
        patch = mock.patch.object(gguf_reader, "file_size",
                                  lambda p: files.get((Path(p).parent.name, Path(p).name), 1))
        patch.start()
        self.addCleanup(patch.stop)
        m = load_diffusers_folder(d)
        self.assertEqual(dict(m.arch.image_parts),
                         {"main": 1_700_000_000, "encoders": 246_000_000, "vae": 167_000_000})  # fp16
        info = IMAGES.annotate(m).spec().image
        self.assertEqual((info.family, info.by), ("sd15", "index"))         # mały CLIP → SD 1.5
        self.assertEqual(load_model(str(d), FakeHttp()).arch.image_hint, "index:StableDiffusionPipeline")

    def test_gguf_flux(self):
        path = self.dir / "flux1-dev-Q4_K_S.gguf"
        path.write_bytes(gguf_bytes([("general.architecture", "flux", "str")], [("double_blocks.0.x", (8, 8))]))
        patch = fake_sizes({path.name: 6_800_000_000})
        patch.start()
        self.addCleanup(patch.stop)
        spec = IMAGES.annotate(load_model(str(path), FakeHttp())).spec()
        self.assertEqual(spec.image.family, "flux")
        v = evaluate(spec, pc(ram_gb=32, gpus=[gpu(vram_gb=12)]), RunSettings(), QUANTS)
        self.assertEqual(v.need.main, 6_800_000_000)                       # cały plik = część główna

    def test_cache_round_trip(self):
        m = load_safetensors_file(safetensors(self.dir / "x.safetensors", SDXL))
        again = model_from_dict(json.loads(json.dumps(model_to_dict(m))))
        self.assertEqual(again.arch.image_parts, m.arch.image_parts)
        self.assertIsNone(again.arch.image)


class VerdictTest(Tmp):
    def test_sdxl_on_authors_laptop(self):
        from aibobr.detection.profiles import load_profile
        spec, hw = self.spec(SDXL), load_profile(LAPTOP)
        v = evaluate(spec, hw, RunSettings(), QUANTS)                       # natywnie 1024×1024
        self.assertEqual((v.level, v.all_at_once, v.need.peak_stage), (FitLevel.GREEN, False, "sample"))
        self.assertEqual((v.need.width, v.need.height), (1024, 1024))
        self.assertLess(v.need.peak, v.need.all_at_once)                    # szczyt, nie suma
        big = evaluate(spec, hw, RunSettings(image_width=1536), QUANTS)
        self.assertEqual(big.level, FitLevel.YELLOW)                        # część główna z RAM

    def test_all_at_once_on_big_card(self):
        v = evaluate(self.spec(SDXL), pc(ram_gb=64, gpus=[gpu(vram_gb=24)]), RunSettings(), QUANTS)
        self.assertTrue(v.all_at_once)

    def test_flux_encoder_on_cpu_and_tiled_vae(self):
        hw = pc(ram_gb=32, gpus=[gpu(vram_gb=8)])                           # 7 GB użytkowe
        spec = self.spec([("double_blocks.0.x", 5.5 * G)], "flux-small")
        v = evaluate(spec, hw, RunSettings(image_width=2048), QUANTS)
        self.assertTrue(v.need.encoders_on_cpu)                             # T5 9 GB > karta
        self.assertTrue(v.need.vae_tiled)                                   # 4 MPx – kafelkami
        self.assertNotIn("encode", v.need.gpu_stages)

    def test_red_when_ram_too_small(self):
        v = evaluate(self.spec(FLUX, "f16"), pc(ram_gb=8, gpus=[gpu(vram_gb=6)]), RunSettings(), QUANTS)
        self.assertEqual(v.level, FitLevel.RED)
        self.assertGreater(v.shortfall, 0)

    def test_cpu_only(self):
        v = evaluate(self.spec(SD15), pc(ram_gb=16), RunSettings(), QUANTS)
        self.assertEqual(v.level, FitLevel.YELLOW)

    def test_batch_multiplies_compute_only(self):
        one = evaluate(self.spec(SDXL), pc(gpus=[gpu(vram_gb=24)]), RunSettings(), QUANTS).need
        two = evaluate(self.spec(SDXL), pc(gpus=[gpu(vram_gb=24)]), RunSettings(image_batch=2), QUANTS).need
        self.assertEqual(two.main, one.main)
        self.assertAlmostEqual(two.main_compute, 2 * one.main_compute, delta=2)


class AdvisorTest(Tmp):
    def explain(self, spec, hw, settings=RunSettings()):
        from aibobr.sources.base import ArchInfo, LoadedModel, Variant
        from aibobr.domain.enums import ModelSource
        model = LoadedModel("m", ModelSource.GGUF_FILE, ArchInfo(), (Variant(spec.quant, spec.weights_bytes),))
        return Advisor(QUANTS).explain(model, spec, hw, settings)

    def test_lower_resolution_suggested(self):
        v, e = self.explain(self.spec(SDXL), pc(gpus=[gpu(vram_gb=8)]), RunSettings(image_width=1536))
        self.assertEqual(e.summary.key, "verdict.image_yellow")
        self.assertEqual(render(e.actions[0]), "rozdzielczość 1024×1024 zmieści się w całości w karcie,")
        self.assertIn("note.img_estimate", [n.key for n in e.notes])

    def test_stages_listed_and_largest_named(self):
        v, e = self.explain(self.spec(SDXL), pc(gpus=[gpu(vram_gb=8)]))
        self.assertEqual(e.summary.key, "verdict.image_green_stages")
        self.assertIn("rysowanie", render(e.need[0]))
        self.assertEqual([x.key for x in e.need[1:4]], ["expl.img.encode", "expl.img.sample", "expl.img.decode"])

    def test_spare_bigger_images(self):
        v, e = self.explain(self.spec(SD15), pc(gpus=[gpu(vram_gb=12)]))
        self.assertEqual(e.actions[0].key, "expl.spare.img_res")

    def test_flux_notes(self):
        v, e = self.explain(self.spec(FLUX, "flux"), pc(ram_gb=64, gpus=[gpu(vram_gb=10)]))   # T5 9,1 GB > 9 GB
        keys = [n.key for n in e.notes]
        self.assertIn("note.img_assumed", keys)
        self.assertIn("note.img_encoders_cpu", keys)

    def test_label(self):
        self.assertEqual(image_label(self.spec(SDXL), RunSettings(image_width=1280, image_height=768, image_batch=2)),
                         "model obrazów SDXL · 1280×768 · 2 obrazów naraz")


class CliAndBarTest(Tmp):
    def test_resolution_parsing(self):
        self.assertEqual(parse_resolution("1024"), (1024, 1024))
        self.assertEqual(parse_resolution("1280x768"), (1280, 768))
        self.assertEqual(parse_resolution("1280×768"), (1280, 768))
        with self.assertRaises(ValueError):
            parse_resolution("duży")

    def test_cli_three_rows(self):
        path = safetensors(self.dir / "realism.safetensors", SDXL)
        c = main.build_app(LAPTOP, user_dir=self.dir / "cfg", cache_dir=self.dir / "cache")
        out = io.StringIO()
        Cli(c, out=out, palette=PLAIN).run(main.build_parser().parse_args([str(path), "--explain"]))
        text = out.getvalue()
        self.assertIn("realism · FP16 · model obrazów SDXL · 1024×1024", text)
        for row in ("System [", "Szczyt [", "Naraz  ["):
            self.assertIn(row, text)
        self.assertIn("Najwięcej pamięci potrzebuje rysowanie", text)
        out = io.StringIO()
        Cli(c, out=out, palette=PLAIN).run(main.build_parser().parse_args([str(path), "--resolution", "1536"]))
        self.assertIn("1536×1536", out.getvalue())
        self.assertIn("Pójdzie, ale wolniej", out.getvalue())

    def test_bar_layout(self):
        v = evaluate(self.spec(SDXL), pc(gpus=[gpu(vram_gb=8)]), RunSettings(), QUANTS)
        lay = image_bar_layout(v, 0, 1000)
        self.assertEqual([s.kind for s in lay.peak], ["main", "compute", "overhead"])
        self.assertEqual([s.kind for s in lay.at_once], ["main", "compute", "overhead", "encoders", "vae"])
        self.assertLess(lay.peak[-1].x1, lay.at_once[-1].x1)
        self.assertLessEqual(lay.peak[-1].x1, lay.markers[0])               # szczyt w karcie


if __name__ == "__main__":
    unittest.main()
