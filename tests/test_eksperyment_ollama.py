"""tools/eksperyment_ollama.py na udawanej Ollamie: efekt logprobs i brak tokenu początku w szablonie."""
import csv
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eksperyment_ollama as eo  # noqa: E402

MODEL = eo.MINITRON
TEMPLATE = "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n"


class FakeOllama:
    """Jak zaobserwowano: pamięć promptu zmienia odpowiedź tylko bez logprobs; z <s> inna odpowiedź."""

    def __init__(self):
        self.loaded, self.last = set(), None
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/version":
                    return self._json({"version": "0.99-test"})
                if self.path == "/api/tags":
                    return self._json({"models": [{"name": MODEL}]})
                if self.path == "/api/ps":
                    return self._json({"models": [{"name": m} for m in state.loaded]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    return self._json({"template": TEMPLATE, "modelfile": "FROM /x"})
                if req.get("keep_alive") == 0 and "prompt" not in req:
                    state.loaded.discard(req["model"])
                    state.last = None
                    return self._json({"done": True})
                prompt = req["prompt"]
                hit = req["model"] in state.loaded and state.last == prompt
                state.loaded.add(req["model"])
                state.last = prompt
                bos = req.get("raw") and prompt.startswith("<s>")
                if bos:
                    text = "Zażółć gęślą jaźń"
                elif hit or req.get("logprobs"):
                    text = "Zazółć gęślą jaźń"
                else:
                    text = "Zazólć żółtym jeszcze raz."
                out = {"response": text, "done": True, "prompt_eval_count": 46 if bos else 45,
                       "prompt_eval_cached_count": 44 if hit else 0}
                if req.get("logprobs"):
                    out["logprobs"] = [{"token": text.split()[0], "logprob": -0.5, "top_logprobs": []}]
                self._json(out)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


class OllamaExperimentTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.addCleanup(self.fake.close)
        self.old_pairs = eo.PAIRS
        eo.PAIRS = [(MODEL, ["pl_diacritics_restore"])]
        self.addCleanup(setattr, eo, "PAIRS", self.old_pairs)

    def exp(self):
        out = Path(tempfile.mkdtemp())
        e = eo.Experiment(out, self.fake.host, repeats=2, unload_timeout=1, echo=lambda s: None)
        e.tests = {k: v for k, v in e.tests.items() if k == "pl_diacritics_restore"}
        e.cfg = {k: v for k, v in e.cfg.items() if k == "pl_diacritics_restore"}
        return out, e

    def test_logprobs_effect(self):
        out, e = self.exp()
        result = e.run("logprobs")
        s = result["summary"][0]
        self.assertEqual((s["O0_jak_O1_bez"], s["O0_jak_O1_z"]), ("NIE", "tak"))
        self.assertEqual((s["O0_bez_jak_O0_z"], s["O1_bez_jak_O1_z"]), ("NIE", "tak"))
        self.assertEqual(s["O0_bez_najczestsza"], "2/2")
        w = result["manifest"]["wnioski"]
        self.assertEqual((w["pamiec_zmienia_odpowiedz_bez_logprobs"], w["pamiec_zmienia_odpowiedz_z_logprobs"],
                          w["logprobs_zmienia_odpowiedz_od_zera"]), (1, 0, 1))
        rows = read_csv(out / "proby.csv")
        self.assertEqual(len(rows), 2 * 2 * 2)
        self.assertTrue(all(r["wliczona"] == "tak" for r in rows))
        events = [json.loads(x) for x in (out / "zdarzenia.jsonl").read_text(encoding="utf-8").splitlines()]
        reqs = [ev["request"] for ev in events if ev["type"] == "attempt"]
        self.assertEqual(sum("logprobs" in r for r in reqs), 4)
        self.assertTrue(all(r["options"]["seed"] == 42 and r["options"]["num_ctx"] == 2048 for r in reqs))

    def test_bos_variants(self):
        out, e = self.exp()
        result = e.run("bos")
        s = result["summary"][0]
        self.assertEqual(s["raw_jak_szablon"], "tak")                     # odtworzenie szablonu się zgadza
        self.assertEqual(s["bos_zmienia_odpowiedz"], "TAK")
        self.assertEqual((s["raw_tokeny"], s["raw+bos_tokeny"]), (45, 46))
        self.assertEqual((s["raw_zaliczone"], s["raw+bos_zaliczone"]), ("0/2", "2/2"))
        m = result["manifest"]
        self.assertIs(m["szablon_ollamy_ma_bos"], False)
        self.assertEqual(m["wnioski"]["zaliczone_raw_bos"], "2/2")
        events = [json.loads(x) for x in (out / "zdarzenia.jsonl").read_text(encoding="utf-8").splitlines()]
        prompts = [ev["prompt"] for ev in events if ev["type"] == "attempt"]
        self.assertTrue(any(p.startswith("<s><|im_start|>user\n") for p in prompts))
        self.assertTrue(all(ev["request"].get("raw") in (None, True) and "logprobs" not in ev["request"]
                            for ev in events if ev["type"] == "attempt"))
        rows = read_csv(out / "proby.csv")
        self.assertTrue(all(r["model_zwolniony"] == "tak" for r in rows))   # każda próba od zera


if __name__ == "__main__":
    unittest.main()
