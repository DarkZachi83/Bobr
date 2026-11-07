"""tools/eksperyment_llamacpp.py: udawana Ollama (HTTP) i udawany llama-server (osobny proces)."""
import csv
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eksperyment_llamacpp as el  # noqa: E402

MODEL = "fake:7b"
FAKE_LLAMA = [sys.executable, str(Path(__file__).resolve().parent / "fake_llama_server.py")]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeOllama:
    """/api/tags, /api/show (FROM pliku GGUF), /api/ps, /api/generate z pamięcią promptu."""

    def __init__(self, gguf: Path, share=1.0, prompt_n=20, logprobs=True):
        self.loaded, self.last = set(), None
        self.prompt_n, self.logprobs = prompt_n, logprobs
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
                    return self._json({"models": [{"name": m, "size": 100, "size_vram": int(100 * share)}
                                                  for m in state.loaded]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    return self._json({"modelfile": f"FROM {gguf}\nTEMPLATE x", "details": {"family": "fake"},
                                       "template": "{{- range .Messages }}<|im_start|>{{ .Role }}",
                                       "model_info": {"fake.block_count": 32}})
                if req.get("keep_alive") == 0 and "prompt" not in req:
                    state.loaded.discard(req["model"])
                    state.last = None
                    return self._json({"done": True})
                hit = req["model"] in state.loaded and state.last == req["prompt"]
                state.loaded.add(req["model"])
                state.last = req["prompt"]
                text = "Zażółć gęślą jaźń" if hit else "Zazolc gesla jazn"
                out = {"response": text, "done": True, "prompt_eval_count": state.prompt_n,
                       "prompt_eval_cached_count": state.prompt_n - 1 if hit else 0}
                if state.logprobs and req.get("logprobs"):
                    first = ("Zażółć", -0.88, "Zazolc", -0.89) if hit else ("Zazolc", -0.88, "Zażółć", -0.89)
                    out["logprobs"] = [{"token": first[0], "logprob": first[1],
                                        "top_logprobs": [{"token": first[0], "logprob": first[1]},
                                                         {"token": first[2], "logprob": first[3]}]}]
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


class LlamaExperimentTest(unittest.TestCase):
    def run_exp(self, share=1.0, models=None, **fake_kw):
        d = Path(tempfile.mkdtemp())
        gguf = d / "model.gguf"
        gguf.write_bytes(b"GGUF")
        fake = FakeOllama(gguf, share, **fake_kw)
        self.addCleanup(fake.close)
        out = d / "out"
        exp = el.Experiment(out, FAKE_LLAMA, fake.host, port=free_port(), repeats=3, models=models,
                            scope="pelny", unload_timeout=1, start_timeout=30, echo=lambda s: None)
        exp.tests = {k: v for k, v in exp.tests.items() if k == "pl_diacritics_restore"}
        exp.cfg = {k: v for k, v in exp.cfg.items() if k == "pl_diacritics_restore"}
        return out, exp.run()

    def test_cache_switch_separates_answers(self):
        out, result = self.run_exp()
        s = result["summary"][0]
        self.assertEqual((s["ollama_O0_jak_O1"], s["llama_E_jak_F"]), ("NIE", "NIE"))    # pamięć zmienia odpowiedź
        self.assertEqual((s["llama_F_najczestsza"], s["llama_E_najczestsza"]), ("3/3", "3/3"))
        self.assertEqual(s["llama_F_jak_pierwsze_zapytanie"], "tak")                   # F = „od zera”
        self.assertEqual((s["llama_F_jak_ollama_O0"], s["llama_E_jak_ollama_O1"]), ("tak", "tak"))
        self.assertEqual(s["pominiete"], 0)
        w = result["manifest"]["wnioski"]
        self.assertEqual((w["llama_pamiec_zmienia_odpowiedz"], w["llama_F_stabilne"]), (1, 1))

    def test_files_conditions_and_manifest(self):
        out, result = self.run_exp()
        rows = read_csv(out / "proby.csv")
        self.assertEqual(len(rows), 2 + 1 + 3 + 3)                     # O0, O1, przygotowawcza E, 3×E, 3×F
        f = [r for r in rows if r["wariant"] == "F"]
        self.assertTrue(all(r["przetworzone_tokeny"] == r["tokeny_promptu"] and r["wliczona"] == "tak" for r in f))
        e = [r for r in rows if r["wariant"] == "E" and r["proba"] != "przygotowawcza"]
        self.assertTrue(all(int(r["przetworzone_tokeny"]) < int(r["tokeny_promptu"]) for r in e))
        o0 = next(r for r in rows if r["wariant"] == "O0")
        self.assertEqual((o0["model_zwolniony"], o0["tokeny_z_pamieci"]), ("tak", "0"))
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["status"], "zakończony")
        self.assertEqual(m["llama_server"], "version: 9999 (fake)")
        self.assertEqual(m["modele"][MODEL]["ngl"], 999)
        self.assertIn("-np", m["modele"][MODEL]["llama_server_args"])
        events = [json.loads(x) for x in (out / "zdarzenia.jsonl").read_text(encoding="utf-8").splitlines()]
        raw = [e for e in events if e["type"] == "llama_raw"]
        self.assertTrue(all(e["request"]["temperature"] == 0 and e["request"]["seed"] == 42 for e in raw))
        self.assertEqual({e["request"]["cache_prompt"] for e in raw}, {True, False})

    def test_token_counts_and_template_bos(self):
        out, result = self.run_exp(prompt_n=6)
        events = [json.loads(x) for x in (out / "zdarzenia.jsonl").read_text(encoding="utf-8").splitlines()]
        tpl = next(e for e in events if e["type"] == "template")
        s = result["summary"][0]
        self.assertEqual(s["tokeny_llama"], len(tpl["token_ids"]))
        self.assertIn(s["tokeny_jak_ollama"], ("tak", "NIE"))
        m = result["manifest"]["modele"][MODEL]
        self.assertEqual((m["szablon_gguf_ma_bos"], m["szablon_ollamy_ma_bos"]), (True, False))
        self.assertEqual(el.has_bos("{{- range .Messages }}<|im_start|>{{ .Role }}"), False)
        self.assertIsNone(el.has_bos(None))

    def test_first_divergence_shows_near_tie(self):
        out, result = self.run_exp()
        s = result["summary"][0]
        self.assertEqual(s["llama_rozbieznosc_pozycja"], 0)
        self.assertEqual(s["llama_rozbieznosc_E_top2"], "'Zażółć' 0.4120 | 'Zazolc' 0.4090")
        self.assertEqual(s["llama_rozbieznosc_F_top2"], "'Zazolc' 0.4130 | 'Zażółć' 0.4080")
        self.assertEqual(s["ollama_rozbieznosc_pozycja"], 0)
        self.assertTrue(result["manifest"]["ollama_logprobs"])
        self.assertEqual(result["manifest"]["wnioski"]["llama_rozbieznosci_z_tokenami"], 1)

    def test_old_ollama_without_logprobs_is_null(self):
        out, result = self.run_exp(logprobs=False)
        s = result["summary"][0]
        self.assertEqual((s["ollama_rozbieznosc_pozycja"], s["ollama_rozbieznosc_O0_top2"]), ("—", "—"))
        self.assertFalse(result["manifest"]["ollama_logprobs"])
        self.assertEqual(s["llama_rozbieznosc_pozycja"], 0)            # llama.cpp dalej ma dane

    def test_partial_offload_and_retry_when_out_of_memory(self):
        self.assertEqual(el.ngl_for(0.5, 32), 15)
        self.assertEqual(el.ngl_for(1.0, 32), 999)
        self.assertEqual(el.ngl_for(None, None), 999)
        os.environ["FAKE_FAIL_NGL"] = "999"
        self.addCleanup(os.environ.pop, "FAKE_FAIL_NGL", None)
        out, result = self.run_exp()
        m = result["manifest"]["modele"][MODEL]
        self.assertEqual((m["status"], m["ngl"]), ("zmierzony", 28))       # 999 nie wszedł → 32 − 4
        events = [json.loads(x) for x in (out / "zdarzenia.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["type"] == "server_failed" and "out of memory" in e["log_tail"] for e in events))


if __name__ == "__main__":
    unittest.main()
