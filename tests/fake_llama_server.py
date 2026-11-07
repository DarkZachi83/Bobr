"""Udawany llama-server do testów tools/eksperyment_llamacpp.py.

Pamięć promptu jak w hipotezie: ten sam prompt drugi raz z cache_prompt: true → przelicza tylko
ostatni token (prompt_n = 1) i odpowiada inaczej; cache_prompt: false → zawsze od zera.
Argumenty jak prawdziwy: -m PLIK --port N … ; --version wypisuje wersję. FAKE_FAIL_NGL=999 –
nie startuje przy -ngl 999 (udawany brak pamięci karty).
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
if "--version" in args:
    print("version: 9999 (fake)")
    sys.exit(0)
port = int(args[args.index("--port") + 1])
if os.environ.get("FAKE_FAIL_NGL") and args[args.index("-ngl") + 1] == os.environ["FAKE_FAIL_NGL"]:
    print("cudaMalloc failed: out of memory", flush=True)
    sys.exit(1)
state = {"last": None}


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
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/props":
            return self._json({"build_info": "fake", "total_slots": 1,
                                "chat_template": "{{bos_token}}<user>{{ content }}<assistant>"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":                # jak szablon Minitrona: „<s>” na początku
            return self._json({"prompt": "<s><user>" + req["messages"][0]["content"] + "<assistant>"})
        if self.path == "/tokenize":                      # serwer i tak dodaje BOS (id 1)
            words = req["content"].replace("<s>", " <s> ").split()
            toks = [{"id": 1, "piece": "<s>"}] + [{"id": 1 if w == "<s>" else 100 + i, "piece": w}
                                                  for i, w in enumerate(words)]
            return self._json({"tokens": toks})
        prompt = req["prompt"]
        n = 1 + len(prompt.replace("<s>", " <s> ").split())
        hit = req.get("cache_prompt") and state["last"] == prompt
        state["last"] = prompt
        text = "Zażółć gęślą jaźń" if hit else "Zazolc gesla jazn"
        # niemal remis na pierwszym tokenie: z pamięci wygrywa „Zażółć”, od zera „Zazolc”
        first = [("Zażółć", 0.4120), ("Zazolc", 0.4090)] if hit else [("Zazolc", 0.4130), ("Zażółć", 0.4080)]
        probs = [{"id": 7, "token": first[0][0], "logprob": __import__("math").log(first[0][1]),
                  "top_logprobs": [{"token": t, "logprob": __import__("math").log(p)} for t, p in first]}]
        probs += [{"id": 8 + i, "token": " " + w, "logprob": -0.01, "top_logprobs": []}
                  for i, w in enumerate(text.split()[1:])] if req.get("n_probs") else []
        self._json({"content": text, "tokens_evaluated": n, "tokens_cached": n - 1 if hit else 0,
                    "timings": {"prompt_n": 1 if hit else n, "cache_n": n - 1 if hit else 0},
                    **({"completion_probabilities": probs} if req.get("n_probs") else {})})


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
