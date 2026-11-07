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
            return self._json({"build_info": "fake", "total_slots": 1})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            return self._json({"prompt": "<user>" + req["messages"][0]["content"] + "<assistant>"})
        prompt = req["prompt"]
        n = len(prompt.split())
        hit = req.get("cache_prompt") and state["last"] == prompt
        state["last"] = prompt
        text = "Zażółć gęślą jaźń" if hit else "Zazolc gesla jazn"
        self._json({"content": text, "tokens_evaluated": n, "tokens_cached": n - 1 if hit else 0,
                    "timings": {"prompt_n": 1 if hit else n, "cache_n": n - 1 if hit else 0}})


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
