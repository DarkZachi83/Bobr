import unittest

from aibobr.bench.ollama import Generation, OllamaBench
from aibobr.diagnostics import bos
from aibobr.diagnostics.bos import BosCheck, BosProbe

CHATML = "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n"


class FakeOllama:
    """Udawana Ollama: liczy tokeny jak tokenizer – tekst tokenu początku (jeśli specjalny) to 1 token,
    „a” to 1 token, a sama Ollama dodaje token początku, gdy auto=True. Pamięć promptu, gdy model nie zwolniony."""

    def __init__(self, expects=True, bos_text="<s>", template=CHATML, auto=False, special=True,
                 verbose_only=False, info=True, release_ok=True, single=True, cache_once=False):
        self.expects, self.bos_text, self.template, self.auto = expects, bos_text, template, auto
        self.special, self.verbose_only, self.info, self.release_ok = special, verbose_only, info, release_ok
        self.single, self.cache_once = single, cache_once
        self.loaded, self.calls, self.prompts = False, [], []

    def digest(self, model):
        return "sha256:abc"

    def show(self, model, verbose=False):
        self.calls.append(("show", verbose))
        if not self.info:
            return {}
        info = {"general.architecture": "llama", "tokenizer.ggml.bos_token_id": 1}
        if self.expects is not None:
            info["tokenizer.ggml.add_bos_token"] = self.expects
        if not self.verbose_only or verbose:
            info["tokenizer.ggml.tokens"] = ["<unk>", self.bos_text, "a"]
        return {"template": self.template, "model_info": info}

    def unload(self, model):
        if self.release_ok:
            self.loaded = False

    def is_loaded(self, model):
        return self.loaded

    def generate(self, model, prompt, num_ctx, num_predict, raw=False, **kw):
        assert raw is True
        self.prompts.append(prompt)
        cached = self.loaded or self.cache_once
        self.cache_once = False
        self.loaded = True
        n = 1 if self.auto else 0
        rest = prompt
        if rest.startswith(self.bos_text):
            n += 1 if self.special else len(self.bos_text)
            rest = rest[len(self.bos_text):]
        n += 1 if self.single else 2
        return Generation(n, 0.01, 1, 0.01, 0.5, "x", prompt_cached=n if cached else 0)


def probe(fake):
    t = [0.0]
    return BosProbe(fake, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s), unload_timeout=1)


class Verdicts(unittest.TestCase):
    def test_minitron_missing(self):
        """Przypadek z eksperymentu: szablon Ollamy bez <s>, Ollama sama nie dodaje."""
        c = probe(FakeOllama(expects=True, auto=False)).run("hf.co/minitron")
        self.assertEqual((c.status, c.without, c.with_text, c.effective, c.template_has), ("missing", 0, 1, 0, False))
        self.assertTrue(c.warning)
        self.assertEqual(c.bos_text, "<s>")

    def test_engine_adds_it_itself(self):
        """Szablon bez tokenu, ale Ollama dodaje go sama (np. Gemma z oficjalnej biblioteki) – w porządku."""
        c = probe(FakeOllama(expects=True, bos_text="<bos>", auto=True)).run("gemma3")
        self.assertEqual((c.status, c.effective), ("ok", 1))

    def test_template_has_it(self):
        c = probe(FakeOllama(template="<s>" + CHATML, auto=False)).run("m")
        self.assertEqual((c.status, c.template_has, c.effective), ("ok", True, 1))
        c = probe(FakeOllama(template="{{ bos_token }}" + CHATML, auto=False)).run("m")
        self.assertEqual(c.status, "ok")

    def test_double(self):
        c = probe(FakeOllama(template="<s>" + CHATML, auto=True)).run("m")
        self.assertEqual((c.status, c.with_text, c.effective), ("double", 2, 2))
        self.assertTrue(c.warning)

    def test_model_without_bos(self):
        c = probe(FakeOllama(expects=False, bos_text="<|endoftext|>", auto=False)).run("qwen")
        self.assertEqual((c.status, c.effective), ("not_used", 0))
        self.assertFalse(c.warning)

    def test_unknown_expectation(self):
        self.assertEqual(probe(FakeOllama(expects=None, auto=True)).run("m").status, "ok")
        c = probe(FakeOllama(expects=None, auto=False)).run("m")
        self.assertEqual((c.status, c.effective), ("unknown", 0))
        self.assertFalse(c.warning)

    def test_verbose_only_when_needed(self):
        fake = FakeOllama(verbose_only=True)
        c = probe(fake).run("m")
        self.assertEqual((c.status, c.bos_text), ("missing", "<s>"))
        self.assertEqual(fake.calls, [("show", False), ("show", True)])
        fake = FakeOllama(verbose_only=True, expects=False)
        probe(fake).run("m")
        self.assertEqual(fake.calls, [("show", False)])          # model bez tokenu początku – słownik niepotrzebny


class Unknowns(unittest.TestCase):
    def test_no_model_info(self):
        c = probe(FakeOllama(info=False)).run("m")
        self.assertEqual((c.status, c.reason), ("unknown", "no_model_info"))

    def test_not_released(self):
        fake = FakeOllama(release_ok=False)
        fake.loaded = True
        c = probe(fake).run("m")
        self.assertEqual((c.status, c.reason), ("unknown", "not_released"))
        self.assertEqual(fake.prompts, [])

    def test_bos_text_not_special(self):
        c = probe(FakeOllama(special=False)).run("m")
        self.assertEqual((c.status, c.reason), ("unknown", "bos_not_special"))

    def test_probe_not_single_token(self):
        c = probe(FakeOllama(single=False, auto=True)).run("m")
        self.assertEqual((c.status, c.reason), ("unknown", "probe_not_single_token"))

    def test_cached_prompt_retried_once(self):
        fake = FakeOllama(cache_once=True)
        c = probe(fake).run("m")
        self.assertEqual(c.status, "missing")
        self.assertEqual(fake.prompts, ["a", "a", "<s>a"])

    def test_released_after_probe(self):
        fake = FakeOllama()
        probe(fake).run("m")
        self.assertFalse(fake.loaded)


class Helpers(unittest.TestCase):
    def test_template_markers(self):
        self.assertTrue(bos.template_has_bos("{{ bos_token }}{% for m in messages %}", None))
        self.assertTrue(bos.template_has_bos("<|begin_of_text|>{{ .Prompt }}", None))
        self.assertFalse(bos.template_has_bos(CHATML, "<s>"))
        self.assertIsNone(bos.template_has_bos("", "<s>"))

    def test_expectation_parsing(self):
        self.assertEqual(bos.model_expectation({"tokenizer.ggml.add_bos_token": "true",
                                                "tokenizer.ggml.bos_token_id": 2}), (True, 2))
        self.assertEqual(bos.model_expectation({}), (None, None))

    def test_round_trip(self):
        c = probe(FakeOllama()).run("m")
        self.assertEqual(BosCheck.from_dict(c.as_dict()), c)


class Client(unittest.TestCase):
    def test_raw_show_digest(self):
        class Http:
            def __init__(self):
                self.sent = []

            def post_json(self, url, payload, timeout=None):
                self.sent.append((url, payload))
                if url.endswith("/api/show"):
                    return {"template": "t", "model_info": {}}
                return {"prompt_eval_count": 2, "response": "x"}

            def get_json(self, url):
                return {"models": [{"name": "m:latest", "digest": "abc"}]}
        http = Http()
        b = OllamaBench("http://h", http=http)
        b.generate("m", "a", 512, 1, raw=True)
        self.assertIs(http.sent[-1][1]["raw"], True)
        b.generate("m", "a", 512, 1)
        self.assertNotIn("raw", http.sent[-1][1])
        self.assertEqual(b.show("m", verbose=True)["template"], "t")
        self.assertEqual(http.sent[-1][1], {"model": "m", "verbose": True})
        self.assertEqual((b.digest("m:latest"), b.digest("x")), ("abc", None))


if __name__ == "__main__":
    unittest.main()
