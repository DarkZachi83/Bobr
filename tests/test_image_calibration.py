"""Pomiar modeli obrazów w Forge-neo (31.10.2026) – test regresyjny współczynników.

Dane: tests/calibration/forge_rtx5050_2026-10-31.json (szczyty etapów z przebiegu w czasie)."""
import json
import unittest
from pathlib import Path

from aibobr.calculator.image import image_need
from aibobr.domain.enums import ModelSource
from aibobr.domain.models import ImageInfo, ModelSpec, RunSettings
from aibobr.domain.units import GIB, gib
from aibobr.knowledge.images import ImageCatalog

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "tests" / "calibration" / "forge_rtx5050_2026-10-31.json").read_text(encoding="utf-8"))
IMAGES = ImageCatalog.load(ROOT / "data" / "image_models.json")


def spec(family: str) -> ModelSpec:
    fam = next(f for f in IMAGES.families if f.id == family)
    parts = DATA["parts_gib"][family]
    info = ImageInfo(family=fam.id, name=fam.name, by="keys",
                     parts=tuple((k, gib(v)) for k, v in parts.items()), assumed=(),
                     native=fam.native, main_gb_per_mpx=fam.main_gb_per_mpx,
                     vae_gb_per_mpx=fam.vae_gb_per_mpx, overhead_gb=fam.overhead_gb, measured=fam.measured)
    return ModelSpec(name=family, source=ModelSource.GGUF_FILE, quant="FP16", image=info)


def need(family, side):
    return image_need(spec(family), RunSettings(image_width=side))


class ForgeCalibrationTest(unittest.TestCase):
    def test_families_marked_measured(self):
        self.assertTrue(spec("sdxl").image.measured)
        self.assertTrue(spec("sd15").image.measured)

    def test_sdxl_sampling_stage_matches_clean_runs(self):
        # Na czysto w karcie: UNet + obliczenia + kontekst CUDA + VAE (Forge go nie zwalnia).
        for run in (r for r in DATA["runs"] if r["family"] == "sdxl" and r["clean"]):
            sample = need("sdxl", run["side"]).stages["sample"] / GIB
            with self.subTest(side=run["side"]):
                self.assertAlmostEqual(sample, run["sample"], delta=0.12)

    def test_vae_decode_compute_per_megapixel(self):
        # Obliczenia dekodowania = szczyt − to, co w tej chwili leży w karcie.
        fam = {f.id: f for f in IMAGES.families}
        for run in DATA["runs"]:
            if not run["decode"] or run.get("decode_unet_resident") == 0.0:
                continue                                              # 1280 – osobny test etapu
            p = DATA["parts_gib"][run["family"]]
            # SD 1.5: wszystko w karcie; SDXL: część UNet + VAE + kontekst CUDA (ok. 0,3)
            resident = run.get("all_resident") or (run["decode_unet_resident"] + p["vae"] + 0.3)
            per_mpx = (run["decode"] - resident) / (run["side"] ** 2 / 1e6)
            with self.subTest(run=f'{run["family"]} {run["side"]}'):
                self.assertLess(abs(per_mpx - fam[run["family"]].vae_gb_per_mpx), 0.35)

    def test_sd15_all_at_once_is_safe_and_close(self):
        # Gdy miejsca jest dość, Forge trzyma wszystko – „Naraz” ma to pokryć z małym zapasem.
        for run in (r for r in DATA["runs"] if r["family"] == "sd15"):
            n = need("sd15", run["side"])
            with self.subTest(side=run["side"]):
                self.assertGreaterEqual(n.all_at_once / GIB, run["decode"])
                self.assertLess(n.all_at_once / GIB - run["decode"], 0.4)

    def test_sdxl_1280_decode_stage(self):
        # Przy 1280 dekodowanie (UNet zwolniony) to ten sam rząd co próbkowanie.
        n = need("sdxl", 1280)
        run = next(r for r in DATA["runs"] if r["family"] == "sdxl" and r["side"] == 1280)
        self.assertAlmostEqual(n.stages["decode"] / GIB, run["decode"], delta=0.15)


if __name__ == "__main__":
    unittest.main()
