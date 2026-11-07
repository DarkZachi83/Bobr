"""tools/eksperyment_cache.py na udawanej Ollamie (prawdziwy serwer HTTP na localhost).

Udawana Ollama odpowiada inaczej, gdy prompt jest już w pamięci (ten sam prompt drugi raz
z rzędu na wczytanym modelu) – dokładnie hipoteza, którą eksperyment ma sprawdzić.
"""
import csv
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eksperyment_cache as ec  # noqa: E402

MODEL = "fake:7b"


class FakeOllama:
    def __init__(self, cached_field=True, stuck=False):
        self.loaded, self.last_prompt = set(), None
        self.cached_field, self.stuck = cached_field, stuck
        self.ps_after_unload = 0
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
                    return self._json({"models": [{"name": MODEL, "digest": "abc", "size": 1}]})
                if self.path == "/api/ps":
                    if state.ps_after_unload:               # zwalnianie trwa chwilę
                        state.ps_after_unload -= 1
                        return self._json({"models": [{"name": MODEL}]})
                    return self._json({"models": [{"name": m} for m in state.loaded]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if req.get("keep_alive") == 0 and "prompt" not in req:
                    if not state.stuck:
                        state.loaded.discard(req["model"])
                        state.last_prompt = None
                        state.ps_after_unload = 1
                    return self._json({"done": True})
                cached = req["model"] in state.loaded and state.last_prompt == req["prompt"]
                state.loaded.add(req["model"])
                state.last_prompt = req["prompt"]
                text = "Zażółć gęślą jaźń" if cached else "Zazolc gesla jazn"
                final = {"done": True, "prompt_eval_count": 20, "eval_count": 3, "eval_duration": 10}
                if state.cached_field:
                    final["prompt_eval_cached_count"] = 19 if cached else 0
                if not req.get("stream"):
                    return self._json({"response": text, **final})
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                words = text.split(" ")
                for i, w in enumerate(words):
                    piece = w + (" " if i < len(words) - 1 else "")
                    self.wfile.write((json.dumps({"response": piece, "done": False}) + "\n").encode())
                self.wfile.write((json.dumps({"response": "", **final}) + "\n").encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


class ExperimentTest(unittest.TestCase):
    def run_exp(self, **fake):
        self.fake = FakeOllama(**fake)
        self.addCleanup(self.fake.close)
        out = Path(tempfile.mkdtemp())
        exp = ec.Experiment(self.fake.host, out, repeats=3, unload_timeout=0.3,
                            plan=[(MODEL, ["pl_diacritics_restore"])], sleep=lambda s: None,
                            echo=lambda s: None)
        result = exp.run()
        return out, result

    def test_variants_separate_cache_effect(self):
        out, result = self.run_exp()
        rows = {s["wariant"]: s for s in result["summary"]}
        # A, B: po przebiegu przygotowawczym każda próba trafia w pamięć promptu
        self.assertEqual(rows["A"]["zgodne_z_proba_1"], "3/3")
        self.assertEqual(rows["A"]["zaliczone"], "3/3")
        self.assertEqual(rows["B"]["najczestsza_jak_w_A"], "tak")
        # C, D: model zwolniony – za każdym razem odpowiedź „na zimno”
        self.assertEqual(rows["C"]["zaliczone"], "0/3")
        self.assertEqual(rows["D"]["najczestsza_jak_w_A"], "NIE")
        self.assertEqual(rows["C"]["prompt_cached_max"], 0)
        self.assertEqual(rows["A"]["prompt_cached_min"], 19)
        self.assertTrue(all(s["problemy_odbioru"] == 0 for s in result["summary"]))

    def test_files_and_raw_stream(self):
        out, result = self.run_exp()
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["manifest.json", "podsumowanie.csv", "proby.csv", "zdarzenia.jsonl"])
        rows = read_csv(out / "proby.csv")
        self.assertEqual(len(rows), 4 * 3 + 2)                    # + przygotowawcze w A i B
        warm = [r for r in rows if r["proba"] == "przygotowawcza"]
        self.assertEqual({r["wariant"] for r in warm}, {"A", "B"})
        self.assertTrue(all(r["wliczona"] == "nie" for r in warm))
        streamed = [r for r in rows if r["strumien"] == "tak"]
        self.assertTrue(all(r["kawalki_strumienia"] == "3" == r["kawalki_w_bobrze"] for r in streamed))
        self.assertTrue(all(r["tekst_bobra_jak_surowy"] == "tak" and r["done"] == "tak" for r in rows))
        self.assertEqual([r["proba"] for r in rows if r["wariant"] == "C"], ["1/3", "2/3", "3/3"])
        events = [json.loads(line) for line in (out / "zdarzenia.jsonl").read_text(encoding="utf-8").splitlines()]
        attempt = next(e for e in events if e["type"] == "attempt" and e["stream"])
        self.assertEqual(attempt["request"]["options"]["seed"], 42)
        self.assertGreaterEqual(len(attempt["raw"]), 4)           # 3 kawałki + komunikat końcowy
        self.assertTrue(any(e["type"] == "release" and e["confirmed"] and e["checks"] >= 2 for e in events))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "zakończony")
        self.assertEqual(manifest["ollama"], "0.99-test")
        self.assertEqual(len(manifest["proby"]), len(rows))

    def test_missing_cached_field_is_null_not_zero(self):
        out, result = self.run_exp(cached_field=False)
        rows = read_csv(out / "proby.csv")
        self.assertTrue(all(r["prompt_eval_cached_count"] == "null" for r in rows))
        self.assertTrue(all(s["prompt_cached_max"] == "null" for s in result["summary"]))

    def test_model_not_released_is_excluded(self):
        out, result = self.run_exp(stuck=True)
        rows = {s["wariant"]: s for s in result["summary"]}
        self.assertEqual(rows["C"]["proby_wliczone"], 0)
        self.assertEqual(rows["C"]["pominiete"], 3)
        self.assertEqual(rows["C"]["werdykt_wiekszosciowy"], "—")
        tries = [r for r in read_csv(out / "proby.csv") if r["wariant"] == "D"]
        self.assertTrue(all(r["model_zwolniony"] == "NIE" and r["warunki_spelnione"] == "nie" for r in tries))


if __name__ == "__main__":
    unittest.main()
