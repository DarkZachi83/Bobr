import unittest
from dataclasses import replace

from aibobr.domain.enums import FitLevel, ModelSource
from aibobr.domain.models import RunSettings
from aibobr.domain.units import gib
from aibobr.i18n import has_key, set_language
from aibobr.knowledge.advisor import Advisor, Line, all_keys, render, render_list
from aibobr.sources.base import ArchInfo, LoadedModel, Variant
from tests.fixtures import (
    DESKTOP_3060, LLAMA_31_8B, LLAMA_33_70B, QUANTS, QWEN3_30B_A3B, QWEN_25_32B, gpu, pc,
)


def loaded(spec, sizes_gib: dict) -> LoadedModel:
    """LoadedModel z architekturą fixture'a i wariantami o podanych rozmiarach (GiB)."""
    arch = ArchInfo(params_total=spec.params_total, n_layers=spec.n_layers,
                    n_kv_heads=spec.n_kv_heads, head_dim=spec.head_dim,
                    max_context=spec.max_context, n_experts=spec.n_experts,
                    n_experts_used=spec.n_experts_used, params_active=spec.params_active)
    return LoadedModel(spec.name, ModelSource.HUGGINGFACE, arch,
                       tuple(Variant(q, gib(s)) for q, s in sizes_gib.items()),
                       preferred_quant="Q4_K_M")


# Rozmiary jak w repozytorium bartowski (zmierzone u autora w etapie 2).
LLAMA = loaded(LLAMA_31_8B, {
    "F32": 29.92, "Q8_0": 7.95, "Q6_K_L": 6.38, "Q6_K": 6.14, "Q5_K_L": 5.64, "Q5_K_M": 5.34,
    "Q5_K_S": 5.21, "Q4_K_L": 4.95, "Q4_K_M": 4.58, "Q3_K_XL": 4.45, "Q4_K_S": 4.37,
    "IQ4_NL": 4.36, "IQ4_XS": 4.14, "Q3_K_L": 4.03, "Q3_K_M": 3.74, "IQ3_M": 3.52,
    "Q2_K_L": 3.44, "Q3_K_S": 3.41, "IQ3_XS": 3.28, "Q2_K": 2.96, "IQ2_M": 2.75})
QWEN32 = loaded(QWEN_25_32B, {"Q8_0": 32.4, "Q6_K": 25.0, "Q5_K_M": 21.7, "Q4_K_M": 18.49,
                              "Q3_K_M": 14.8, "Q2_K": 11.5})
LLAMA70 = loaded(LLAMA_33_70B, {"Q8_0": 69.8, "Q4_K_M": 39.6, "Q3_K_M": 31.9, "Q2_K": 24.6})
MOE = loaded(QWEN3_30B_A3B, {"Q4_K_M": 17.3})
AUTO = RunSettings()


def explain(model, hw, settings=AUTO, quant=None, catalog=()):
    adv = Advisor(QUANTS, catalog)
    return adv.explain(model, model.spec(quant), hw, settings)


def keys(lines):
    return [line.key for line in lines]


class GreenTest(unittest.TestCase):
    def test_spare_context_and_better_quant(self):
        v, e = explain(LLAMA, DESKTOP_3060)
        self.assertEqual(v.level, FitLevel.GREEN)
        self.assertEqual(e.actions_title, "expl.spare.title")
        ctx, quant = e.actions
        self.assertEqual(ctx, Line("expl.spare.context", {"ctx": 32768}))   # 64k już nie
        self.assertEqual(quant.fields["quant"], "Q8_0")                    # F32 się nie mieści

    def test_no_spare_suggestion_that_would_itself_be_tight(self):
        # Karta 8 GB autora: Q5_K_M zmieściłoby się z 0,16 GB zapasu – tego nie podpowiadamy.
        v, e = explain(LLAMA, pc(gpus=[gpu(vram_gb=8)]))
        self.assertTrue(v.low_margin)
        self.assertEqual(keys(e.actions), ["expl.spare.none"])
        self.assertEqual(keys(e.notes), ["note.low_margin"])


class YellowTest(unittest.TestCase):
    def test_mostly_in_vram_has_gentle_wording(self):
        _, e = explain(LLAMA, pc(gpus=[gpu(vram_gb=8)]), replace(AUTO, context_tokens=16384))
        self.assertEqual(e.summary.key, "verdict.yellow_mostly")
        self.assertIn("expl.pc.slow_little", keys(e.computer))
        self.assertEqual(keys(e.actions), ["expl.act.quant", "expl.act.context", "expl.act.kv"])
        # Q3_K_XL ma większy plik niż Q4_K_S, ale to klasa Q3 – podpowiadamy Q4_K_S
        # (błąd wyłapany na laptopie autora).
        self.assertEqual(e.actions[0].fields["quant"], "Q4_K_S")
        self.assertEqual(e.actions[1].fields["ctx"], 8192)

    def test_32b_on_12gb_card_needs_smaller_model(self):
        _, e = explain(QWEN32, DESKTOP_3060)
        self.assertEqual(e.summary.key, "verdict.yellow")
        self.assertIn("expl.pc.slow", keys(e.computer))
        self.assertEqual(keys(e.actions), ["expl.act.nothing"])
        self.assertEqual(e.actions[0].fields["quant"], "Q3_K_M")       # Q2_K poniżej progu jakości

    def test_catalog_suggests_biggest_model_that_fits(self):
        small = replace(LLAMA_31_8B, name="Llama 3.1 8B")
        mid = replace(QWEN_25_32B, name="Qwen 2.5 14B", params_total=14_770_000_000,
                      weights_bytes=gib(7.5), n_layers=48)
        _, e = explain(QWEN32, DESKTOP_3060, catalog=[small, mid])
        self.assertEqual(e.actions[-1], Line("expl.act.model", {"name": "Qwen 2.5 14B", "quant": "Q4_K_M"}))

    def test_catalog_skips_tight_fit_and_same_model(self):
        small = replace(LLAMA_31_8B, name="Llama 3.1 8B")
        tight = replace(QWEN_25_32B, name="Qwen 2.5 14B", params_total=14_770_000_000,
                        weights_bytes=gib(8.4), n_layers=48)          # zapas ~0,6 GB
        same = replace(QWEN_25_32B, weights_bytes=gib(1))             # ta sama nazwa
        _, e = explain(QWEN32, DESKTOP_3060, catalog=[small, tight, same])
        self.assertEqual(e.actions[-1].fields["name"], "Llama 3.1 8B")

    def test_lowest_acceptable_quant_not_q2(self):
        # Karta 6 GB: mieszczą się Q3_K_S (3,5 bita), IQ3_XS (3,3) i warianty Q2 –
        # podpowiadamy najlepszą jakość nie gorszą niż ok. 3,4 bita.
        _, e = explain(LLAMA, pc(gpus=[gpu(vram_gb=6)]))
        self.assertEqual(keys(e.actions), ["expl.act.quant"])
        self.assertEqual(e.actions[0].fields["quant"], "Q3_K_S")
        self.assertNotIn("Q2", str(e.actions))

    def test_combo_when_no_single_change_helps(self):
        _, e = explain(LLAMA, pc(gpus=[gpu(vram_gb=5.5)]), replace(AUTO, context_tokens=16384))
        self.assertEqual(keys(e.actions), ["expl.act.combo"])

    def test_moe(self):
        v, e = explain(MOE, DESKTOP_3060)
        self.assertEqual(e.summary.key, "verdict.yellow_moe")
        self.assertIn("expl.pc.slow_moe", keys(e.computer))
        # „Używalnie” + „potrzebny mniejszy model” sobie przeczyły (laptop autora).
        self.assertEqual(keys(e.actions), ["expl.act.moe_offload"])

    def test_quality_class_comes_from_name_not_file_size(self):
        from aibobr.knowledge.advisor import _bits_per_weight
        bits = {q: _bits_per_weight(LLAMA.spec(q), QUANTS) for q in
                ("Q3_K_XL", "Q4_K_S", "Q4_K_L", "Q4_K_M", "Q6_K_L", "Q2_K_L")}
        self.assertLess(bits["Q3_K_XL"], bits["Q4_K_S"])
        self.assertEqual(bits["Q4_K_L"], bits["Q4_K_M"])
        self.assertEqual(bits["Q6_K_L"], QUANTS.bits("Q6_K"))
        self.assertLess(bits["Q2_K_L"], 3.4)

    def test_cpu_only(self):
        hw = pc(ram_gb=16, gpus=[gpu("Intel UHD", vram_gb=0.1, integrated=True)])
        _, e = explain(LLAMA, hw)
        self.assertEqual(e.summary.key, "verdict.yellow_cpu")
        self.assertEqual(keys(e.computer)[:2], ["expl.pc.no_gpu", "expl.pc.all_in_ram"])
        self.assertEqual(keys(e.actions), ["expl.act.cpu_only"])

    def test_ram_busy_now(self):
        hw = replace(DESKTOP_3060, ram_used=gib(25))
        _, e = explain(QWEN32, hw)
        self.assertIn("expl.pc.ram_busy", keys(e.computer))


class RedTest(unittest.TestCase):
    def test_falls_back_to_slow_options(self):
        v, e = explain(LLAMA70, DESKTOP_3060)
        self.assertEqual(v.level, FitLevel.RED)
        self.assertEqual(keys(e.actions), ["expl.act.quant", "expl.act.context"])
        self.assertEqual(e.actions[0].fields["quant"], "Q3_K_M")
        self.assertEqual(str(e.actions[0].fields["where"]), "expl.where.ram")
        self.assertEqual(e.actions[1].fields["ctx"], 2048)

    def test_hopeless(self):
        _, e = explain(LLAMA70, pc(ram_gb=16, gpus=[gpu(vram_gb=8)]))
        self.assertEqual(keys(e.actions), ["expl.act.nothing_red"])


class RenderingTest(unittest.TestCase):
    def tearDown(self):
        set_language("en")

    def test_polish_and_english_numbers(self):
        line = Line("expl.need.title", {"total": __import__("aibobr.knowledge.advisor",
                    fromlist=["Size"]).Size(gib(22.4))})
        self.assertEqual(render(line, "pl"), "Model potrzebuje 22,4 GB:")
        self.assertEqual(render(line, "en"), "The model needs 22.4 GB:")

    def test_last_item_ends_with_full_stop(self):
        _, e = explain(LLAMA, pc(gpus=[gpu(vram_gb=8)]), replace(AUTO, context_tokens=16384))
        texts = render_list(e.actions, "pl")
        self.assertTrue(texts[0].endswith(","))
        self.assertTrue(texts[-1].endswith("."))

    def test_every_key_produced_by_scenarios_is_translated(self):
        scenarios = [
            (LLAMA, DESKTOP_3060, AUTO), (LLAMA, pc(gpus=[gpu(vram_gb=8)]), AUTO),
            (LLAMA, pc(gpus=[gpu(vram_gb=8)]), replace(AUTO, context_tokens=16384)),
            (QWEN32, DESKTOP_3060, AUTO), (LLAMA, pc(gpus=[gpu(vram_gb=6)]), AUTO),
            (MOE, DESKTOP_3060, AUTO), (LLAMA70, DESKTOP_3060, AUTO),
            (LLAMA70, pc(ram_gb=16, gpus=[gpu(vram_gb=8)]), AUTO),
            (LLAMA, pc(ram_gb=16), AUTO), (LLAMA70, pc(ram_gb=16), AUTO),
            (QWEN32, replace(DESKTOP_3060, ram_used=gib(25)), replace(AUTO, context_tokens=65536)),
        ]
        for model, hw, settings in scenarios:
            _, e = explain(model, hw, settings)
            for key in all_keys(e):
                self.assertTrue(has_key(key), key)
                for line in (*e.need, *e.computer, *e.actions, *e.notes, e.summary):
                    for lang in ("pl", "en"):
                        self.assertNotIn("{", render(line, lang), line.key)


if __name__ == "__main__":
    unittest.main()
