import unittest
import urllib.request

from aibobr.sources.base import SourceError
from aibobr.sources.http import RemoteFile, _StripAuthOnHostChange
from aibobr.sources.huggingface import HF, HfRef, HuggingFaceSource, arch_from_config, parse_reference
from tests.fake_http import FakeHttp
from tests.gguf_writer import llama_like

REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
TREE = f"{HF}/api/models/{REPO}/tree/main?recursive=true"


def resolve(path, repo=REPO):
    return f"{HF}/{repo}/resolve/main/{path}"


def lfs(path, size):
    return {"type": "file", "path": path, "size": 135, "lfs": {"size": size}}


GGUF_TREE = [
    {"type": "file", "path": "README.md", "size": 5000},
    lfs("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", 8_540_000_000),
    lfs("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 4_920_000_000),
    lfs("Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf", 4_450_000_000),
    lfs("mmproj-model-f16.gguf", 600_000_000),
    {"type": "directory", "path": "F16"},
    lfs("F16/Meta-Llama-3.1-8B-Instruct-F16-00001-of-00002.gguf", 9_000_000_000),
    lfs("F16/Meta-Llama-3.1-8B-Instruct-F16-00002-of-00002.gguf", 7_070_000_000),
]


def gguf_fake(**kw):
    header = llama_like(name="Meta Llama 3.1 8B Instruct")
    files = {resolve(e["path"]): header for e in GGUF_TREE if e.get("type") == "file"}
    return FakeHttp(json={TREE: GGUF_TREE}, files=files, **kw)


class ParseReferenceTest(unittest.TestCase):
    def test_forms(self):
        cases = {
            "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct": HfRef("Qwen/Qwen2.5-7B-Instruct"),
            "huggingface.co/Qwen/Qwen2.5-7B-Instruct/": HfRef("Qwen/Qwen2.5-7B-Instruct"),
            "https://www.huggingface.co/Qwen/Qwen2.5-7B-Instruct?x=1#y": HfRef("Qwen/Qwen2.5-7B-Instruct"),
            "Qwen/Qwen2.5-7B-Instruct": HfRef("Qwen/Qwen2.5-7B-Instruct"),
            f"hf.co/{REPO}:q4_k_m": HfRef(REPO, quant="Q4_K_M"),
            f"https://huggingface.co/{REPO}/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf":
                HfRef(REPO, "main", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
            f"https://huggingface.co/{REPO}/tree/dev": HfRef(REPO, "dev"),
        }
        for text, ref in cases.items():
            self.assertEqual(parse_reference(text), ref, text)

    def test_not_models(self):
        for text in ("https://huggingface.co/datasets/a/b", "https://huggingface.co/spaces/a/b",
                     "https://github.com/a/b", "just text", "a/b/c", "https://huggingface.co/Qwen"):
            self.assertIsNone(parse_reference(text), text)


class GgufRepoTest(unittest.TestCase):
    def test_variants_and_architecture(self):
        http = gguf_fake()
        m = HuggingFaceSource(http).load(HfRef(REPO))
        self.assertEqual(m.name, "Meta Llama 3.1 8B Instruct")
        self.assertEqual([v.quant for v in m.variants], ["F16", "Q8_0", "Q4_K_M", "IQ4_XS"])
        self.assertEqual(m.variant("F16").size, 16_070_000_000)       # suma dwóch części
        self.assertEqual(len(m.variant("F16").files), 2)
        self.assertEqual(m.arch.params_total, 8_030_261_248)
        spec = m.spec()                                                 # domyślnie Q4_K_M
        self.assertEqual((spec.quant, spec.weights_bytes, spec.n_kv_heads),
                         ("Q4_K_M", 4_920_000_000, 8))

    def test_reads_only_the_header(self):
        http = gguf_fake()
        HuggingFaceSource(http).load(HfRef(REPO))
        ranges = [u for u in http.calls if "/resolve/" in u]
        self.assertLessEqual(len(ranges), 3)        # ~2,3 MB nagłówka / kawałki 1 MB
        self.assertTrue(all(u.endswith("Q4_K_M.gguf") for u in ranges))

    def test_mmproj_is_not_a_variant(self):
        m = HuggingFaceSource(gguf_fake()).load(HfRef(REPO))
        self.assertFalse(any("mmproj" in f for v in m.variants for f in v.files))

    def test_quant_from_link(self):
        src = HuggingFaceSource(gguf_fake())
        self.assertEqual(src.load(HfRef(REPO, quant="Q8_0")).spec().quant, "Q8_0")
        m = src.load(HfRef(REPO, path="Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"))
        self.assertEqual(m.spec().quant, "IQ4_XS")

    def test_gated_file(self):
        http = gguf_fake(denied={resolve("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")})
        with self.assertRaises(SourceError) as ctx:
            HuggingFaceSource(http).load(HfRef(REPO))
        self.assertEqual(ctx.exception.code, "source.gated")

    def test_missing_repo(self):
        with self.assertRaises(SourceError) as ctx:
            HuggingFaceSource(FakeHttp()).load(HfRef("nobody/nothing"))
        self.assertEqual(ctx.exception.code, "source.not_found")

    def test_incomplete_split_variant_is_dropped(self):
        tree = [e for e in GGUF_TREE if "00002-of" not in e.get("path", "")]
        http = gguf_fake()
        http.json[TREE] = tree
        m = HuggingFaceSource(http).load(HfRef(REPO))
        self.assertNotIn("F16", [v.quant for v in m.variants])


class TransformersRepoTest(unittest.TestCase):
    MIXTRAL = {
        "architectures": ["MixtralForCausalLM"], "torch_dtype": "bfloat16",
        "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
        "hidden_size": 4096, "intermediate_size": 14336, "max_position_embeddings": 32768,
        "num_local_experts": 8, "num_experts_per_tok": 2,
    }

    def test_config_and_safetensors(self):
        repo = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        tree = [{"type": "file", "path": "config.json", "size": 700}] + [
            lfs(f"model-{i:05d}-of-00019.safetensors", 4_900_000_000) for i in range(1, 20)]
        http = FakeHttp(json={
            f"{HF}/api/models/{repo}/tree/main?recursive=true": tree,
            resolve("config.json", repo): self.MIXTRAL,
        })
        m = HuggingFaceSource(http).load(HfRef(repo))
        a = m.arch
        self.assertEqual(a.params_total, 19 * 4_900_000_000 // 2)          # bf16 = 2 B
        self.assertEqual((a.n_layers, a.n_kv_heads, a.head_dim), (32, 8, 128))
        self.assertEqual((a.n_experts, a.n_experts_used), (8, 2))
        self.assertAlmostEqual(a.params_active / 1e9, 12.9, delta=0.3)   # Mixtral: ~12,9B aktywnych
        spec = m.spec()
        self.assertEqual(spec.quant, "Q4_K_M")
        self.assertIsNone(spec.weights_bytes)                               # rozmiar z tabeli

    def test_multimodal_text_config_wins(self):
        cfg = {"architectures": ["Gemma3ForConditionalGeneration"],
               "text_config": {"num_hidden_layers": 48, "num_attention_heads": 16,
                               "num_key_value_heads": 8, "head_dim": 256, "hidden_size": 3840}}
        a = arch_from_config(cfg)
        self.assertEqual((a.n_layers, a.n_kv_heads, a.head_dim), (48, 8, 256))

    def test_prequantized_repo_has_no_param_count(self):
        cfg = dict(self.MIXTRAL, quantization_config={"quant_method": "awq"})
        self.assertIsNone(arch_from_config(cfg, []).params_total)


class HttpPiecesTest(unittest.TestCase):
    def test_remote_file_reads_across_chunks(self):
        data = bytes(range(256)) * 40
        http = FakeHttp(files={"u": data})
        f = RemoteFile(http, "u", chunk=1000)
        self.assertEqual(f.read(10), data[:10])
        f.skip(2500)
        self.assertEqual(f.read(1500), data[2510:4010])
        with self.assertRaises(SourceError):
            f.skip(100_000)
            f.read(1)

    def test_token_is_not_sent_to_cdn(self):
        handler = _StripAuthOnHostChange()
        req = urllib.request.Request(f"{HF}/x/resolve/main/f.gguf",
                                     headers={"Authorization": "Bearer hf_secret"})
        cdn = handler.redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.hf.co/f")
        same = handler.redirect_request(req, None, 302, "Found", {}, f"{HF}/x/resolve/abc/f.gguf")
        self.assertFalse(cdn.has_header("Authorization"))
        self.assertTrue(same.has_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
