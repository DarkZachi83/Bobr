"""Eksperyment kontrolny 2: ta sama pamięć promptu w llama.cpp, sterowana jednym przełącznikiem.

    python tools/eksperyment_llamacpp.py --dry-run                 # plan, ścieżki modeli, wersje
    python tools/eksperyment_llamacpp.py --zakres pary             # 10 par z eksperymentu A–D (kilkanaście min)
    python tools/eksperyment_llamacpp.py                           # wszystkie modele z Ollamy × 15 zadań

Skąd ten skrypt: w eksperymencie A–D (tools/eksperyment_cache.py) odpowiedź w Ollamie zależała od
tego, czy prompt był w pamięci. Uwaga z r/LocalLLaMA (jacek2023): „spróbuj llama.cpp”. W llama-server
pamięć promptu da się wyłączyć w pojedynczym zapytaniu (cache_prompt: false) – bez zwalniania modelu.
Jeśli hipoteza jest prawdziwa, F (bez pamięci) daje zawsze odpowiedź „od zera”, a E – „z pamięci”.

Dla każdego modelu, po kolei (nigdy dwa modele naraz):
  Ollama     O0  model zwolniony (potwierdzone w /api/ps), zadanie liczone od zera       – 1 próba
             O1  to samo zadanie od razu drugi raz (prompt z pamięci)                    – 1 próba
  llama.cpp  E   cache_prompt: true  – przebieg przygotowawczy, potem N prób
             F   cache_prompt: false – N prób; każda musi przeliczyć cały prompt
Ten sam plik GGUF (ścieżka z Ollamy: /api/show), ten sam prompt, temperatura 0, seed 42, kontekst
2048, myślenie wyłączone. llama-server: --parallel 1, stały rozmiar batcha (-b, -ub), warstwy na
karcie (-ngl) tak jak w Ollamie (z udziału w karcie), zapisane w manifeście.

Warunki próby sprawdzamy z danych serwera, nie zakładamy: w F timings.prompt_n musi być równe
długości promptu (nic z pamięci), w E – mniejsze (prompt z pamięci). Próba niespełniająca warunku
jest oznaczona i nie wchodzi do podsumowania. Brak pola w odpowiedzi to null, nigdy 0.

Porównanie Ollama ↔ llama.cpp jest pomocnicze: szablony czatu i wersje silnika mogą się różnić,
więc inna treść między serwerami nie jest jeszcze wynikiem. Wynikiem jest porównanie E ↔ F
(ten sam serwer, ten sam stan modelu, zmienia się tylko pamięć promptu) i O0 ↔ O1.

Tokeny (0.8.2, pytanie z r/LocalLLaMA o logits):
- llama.cpp: tokeny wejściowe z /tokenize (id i tekst), a przy każdej odpowiedzi n_probs: 5 –
  dla każdego wygenerowanego tokenu 5 najbardziej prawdopodobnych kandydatów (prawdopodobieństwa
  po softmaksie, nie surowe logits). Zdublowany token początku (szablon dokleja „<s>”, a serwer
  dodaje własny) jest usuwany przed wysłaniem; kolumna tokeny_jak_ollama mówi, czy wejście ma tyle
  tokenów co w Ollamie – tylko wtedy porównanie między serwerami ma sens.
- Ollama: logprobs i top_logprobs w /api/generate, jeśli wersja je zwraca (inaczej null). Sekwencji
  tokenów wejściowych Ollama nie udostępnia – tylko ich liczbę.
- Pierwsza rozbieżność: dla par, w których odpowiedź z pamięci i od zera się różni, pozycja
  pierwszego innego tokenu i dwa najbardziej prawdopodobne tokeny w obu stanach – czy to był
  niemal remis.

Wyniki: zdarzenia.jsonl, proby.csv, podsumowanie.csv, manifest.json, llama-server-<model>.log.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aibobr import __version__                                    # noqa: E402
from aibobr.bench.ollama import OllamaBench                         # noqa: E402
from aibobr.bench.quality import CONTEXT, evaluate                  # noqa: E402
from aibobr.sources.http import HttpClient                          # noqa: E402
from eksperyment_cache import PLAN as PAIRS, load_tests, now_iso, null, release, sha, write_csv  # noqa: E402

OLLAMA = "http://localhost:11434"
PORT = 8089
BATCH = 512
START_TIMEOUT = 300.0          # s – wczytanie dużego modelu z dysku na wolniejszym komputerze
SUITES_ORDER = ("pl_jezyk_v1", "en_format_v1")
N_PROBS = 5


# ---------------------------------------------------------------------- tokeny i prawdopodobieństwa
def _p(entry: dict) -> Optional[float]:
    if entry.get("logprob") is not None:
        return math.exp(float(entry["logprob"]))
    if entry.get("prob") is not None:
        return float(entry["prob"])
    return None


def llama_tokens(response: dict) -> Optional[list[dict]]:
    """completion_probabilities z llama-server → [{token, id, p, top: [(token, p), …]}]. Obsługuje
    nowy format (logprob, top_logprobs) i stary (content, probs: tok_str/prob)."""
    probs = response.get("completion_probabilities")
    if not probs:
        return None
    out = []
    for e in probs:
        top = e.get("top_logprobs") or e.get("top_probs") or e.get("probs") or []
        out.append({"token": e.get("token", e.get("content")), "id": e.get("id"), "p": _p(e),
                    "top": [(t.get("token", t.get("tok_str")), _p(t)) for t in top]})
    return out


def ollama_tokens(response: dict) -> Optional[list[dict]]:
    """logprobs z Ollamy (nowsze wersje) → ten sam kształt co llama_tokens; None, gdy brak pola."""
    lp = response.get("logprobs")
    if not lp:
        return None
    return [{"token": e.get("token"), "id": e.get("id"), "p": _p(e),
             "top": [(t.get("token"), _p(t)) for t in (e.get("top_logprobs") or [])]} for e in lp]


def first_divergence(a: Optional[list[dict]], b: Optional[list[dict]]) -> Optional[dict]:
    """Pierwsza pozycja, na której wygenerowane tokeny się różnią, i dwa najbardziej
    prawdopodobne kandydaty w obu stanach. None – brak danych albo brak różnicy."""
    if not a or not b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x["token"] != y["token"]:
            return {"pozycja": i, "a": x, "b": y}
    if len(a) != len(b):
        i = min(len(a), len(b))
        return {"pozycja": i, "a": a[i] if i < len(a) else None, "b": b[i] if i < len(b) else None}
    return None


def top2(entry: Optional[dict]) -> str:
    """„'tok1' 0.4120 | 'tok2' 0.4090” – czy był niemal remis."""
    if not entry:
        return "—"
    top = [(t, p) for t, p in entry.get("top", []) if p is not None][:2]
    if not top:
        return f"{entry.get('token')!r} {entry['p']:.4f}" if entry.get("p") is not None else repr(entry.get("token"))
    return " | ".join(f"{t!r} {p:.4f}" for t, p in top)


def tokenize(srv, text: str) -> tuple[list, list]:
    """/tokenize z tokenem początku tak jak /completion: (identyfikatory, teksty tokenów)."""
    r = srv.request("POST", "/tokenize", {"content": text, "add_special": True, "with_pieces": True})
    ids, pieces = [], []
    for t in r.get("tokens") or []:
        if isinstance(t, dict):
            ids.append(t.get("id"))
            pieces.append(t.get("piece"))
        else:
            ids.append(t)
            pieces.append(None)
    return ids, pieces


def strip_double_bos(srv, formatted: str) -> tuple[str, list, list, bool]:
    """Szablon czatu bywa z „<s>” na początku, a /completion i tak dodaje token początku – model
    widzi go dwa razy (Minitron: 65 tokenów w llama.cpp, 64 w Ollamie). Usuwamy tekst z szablonu."""
    ids, pieces = tokenize(srv, formatted)
    if len(ids) >= 2 and ids[0] == ids[1] and pieces[0] and formatted.startswith(pieces[0]):
        formatted = formatted[len(pieces[0]):]
        ids, pieces = tokenize(srv, formatted)
        return formatted, ids, pieces, True
    return formatted, ids, pieces, False


# ---------------------------------------------------------------------- llama-server
class LlamaServer:
    """Uruchamia llama-server dla jednego modelu i rozmawia z nim przez HTTP (bez proxy – localhost)."""

    def __init__(self, binary, model_path: str, port: int, ngl: int, batch: int, ubatch: int,
                 log_path: Path, seed: int = 42):
        prefix = [binary] if isinstance(binary, str) else list(binary)      # lista – np. [python, skrypt]
        self.args = prefix + ["-m", model_path, "--host", "127.0.0.1", "--port", str(port), "-c", str(CONTEXT),
                     "-np", "1", "-b", str(batch), "-ub", str(ubatch), "-ngl", str(ngl), "--seed", str(seed)]
        self.url = f"http://127.0.0.1:{port}"
        self.ngl = ngl
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method: str, path: str, payload: Optional[dict] = None, timeout: float = 900) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with self._opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def start(self, timeout: float = START_TIMEOUT, sleep: Callable[[float], None] = time.sleep) -> bool:
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.args, stdout=self._log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.proc.poll() is not None:                  # zakończył się – np. brak pamięci karty
                return False
            try:
                if self.request("GET", "/health", timeout=5).get("status") == "ok":
                    return True
            except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, ValueError):
                pass
            sleep(0.5)
        return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if getattr(self, "_log", None):
            self._log.close()

    def log_tail(self, n: int = 15) -> str:
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""


def llama_version(binary) -> Optional[str]:
    try:
        cmd = ([binary] if isinstance(binary, str) else list(binary)) + ["--version"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout + out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr) else None
    except (OSError, subprocess.SubprocessError):
        return None


def gguf_of(http: HttpClient, host: str, model: str) -> tuple[Optional[str], Optional[int], dict]:
    """(ścieżka pliku GGUF, liczba warstw, surowe /api/show) – pierwszy FROM z Modelfile to model
    (kolejne to np. projektor obrazu)."""
    show = http.post_json(f"{host}/api/show", {"model": model})
    path = None
    for line in str(show.get("modelfile") or "").splitlines():
        if line.startswith("FROM "):
            path = line[5:].strip()
            break
    info = show.get("model_info") or {}
    blocks = next((int(v) for k, v in info.items() if k.endswith(".block_count")), None)
    return path, blocks, {"details": show.get("details"), "parameters": show.get("parameters")}


def readable(path: Optional[str]) -> bool:
    """Plik modelu musi dać się odczytać (blob Ollamy bywa w katalogu użytkownika ollama)."""
    try:
        return bool(path) and Path(path).is_file() and os.access(path, os.R_OK)
    except OSError:
        return False


def ngl_for(share: Optional[float], blocks: Optional[int]) -> int:
    """Warstwy na karcie jak w Ollamie: cały model w karcie → 999; część → udział × warstwy (o 1 mniej)."""
    if share is None or share >= 0.999 or not blocks:
        return 999
    return max(0, math.floor(share * (blocks + 1)) - 1)


# ---------------------------------------------------------------------- eksperyment
class Experiment:
    def __init__(self, out: Path, binary: str, ollama: str = OLLAMA, port: int = PORT, repeats: int = 5,
                 batch: int = BATCH, ubatch: int = BATCH, models: Optional[list[str]] = None, scope: str = "pelny",
                 unload_timeout: float = 30.0, start_timeout: float = START_TIMEOUT,
                 sleep: Callable[[float], None] = time.sleep, echo: Callable[[str], None] = print,
                 root: Optional[Path] = None):
        self.out, self.binary, self.port, self.repeats = out, binary, port, repeats
        self.batch, self.ubatch, self.scope = batch, ubatch, scope
        self.unload_timeout, self.start_timeout, self.sleep, self.echo = unload_timeout, start_timeout, sleep, echo
        self.host = ollama.rstrip("/")
        self.http = HttpClient(timeout=900)
        self.bench = OllamaBench(self.host, http=self.http)
        self.tests, self.cfg = load_tests(root or Path(__file__).resolve().parent.parent)
        self.models = models
        self.rows: list[dict] = []
        self.order = 0
        self._events = None
        self.servers: dict = {}
        self.tokens: dict = {}                 # (model, zadanie, wariant, próba) → tokeny odpowiedzi
        self.prompt_tokens: dict = {}          # (model, zadanie) → {"llama": n, "ollama": n}
        self.ollama_logprobs: Optional[bool] = None

    def event(self, kind: str, **data) -> None:
        self._events.write(json.dumps({"type": kind, "time": now_iso(), **data}, ensure_ascii=False) + "\n")
        self._events.flush()

    # -------------------------------------------------- plan
    def plan(self, available: list[str]) -> list[tuple[str, list[str]]]:
        if self.scope == "pary":
            plan = [(m, t) for m, t in PAIRS if m in available]
        else:
            ids = [tid for sid in SUITES_ORDER for tid, (s, *_rest) in self.cfg.items() if s == sid]
            plan = [(m, ids) for m in available]
        if self.models:
            plan = [(m, t) for m, t in plan if m in self.models]
        return plan

    # -------------------------------------------------- jedna próba
    def row(self, server: str, model: str, test_id: str, variant: str, label: str, counted: bool, ok: bool,
            text: str, prompt_tokens: Optional[int], cached: Optional[int], processed: Optional[int],
            seconds: float, error: str = "", released: str = "—", ngl: Any = "—") -> dict:
        self.order += 1
        v = evaluate(text, self.tests[test_id]["evaluator"]) if not error else None
        r = {"kolejnosc": self.order, "czas": now_iso(), "serwer": server, "model": model, "zadanie": test_id,
             "wariant": variant, "proba": label, "wliczona": "tak" if counted and ok and not error else "nie",
             "warunki_spelnione": "tak" if ok and not error else "nie",
             "zaliczona": "—" if v is None else ("tak" if v.passed else "nie"),
             "powod": "" if v is None or v.passed else v.reason, "blad": error,
             "tokeny_promptu": null(prompt_tokens), "tokeny_z_pamieci": null(cached),
             "przetworzone_tokeny": null(processed), "czas_s": round(seconds, 3),
             "model_zwolniony": released, "ngl": ngl, "skrot_odpowiedzi": sha(text), "odpowiedz": text}
        self.rows.append(r)
        self.event("attempt", **{k: v for k, v in r.items()})
        return r

    def ollama_attempt(self, model: str, test_id: str, variant: str, released: Optional[tuple]) -> dict:
        test = self.tests[test_id]
        _, max_tokens, temperature, seed = self.cfg[test_id]
        t0 = time.monotonic()
        # Jak OllamaBench.generate (te same opcje), plus logprobs – Bobr ich nie prosi.
        payload = {"model": model, "prompt": test["prompt"], "stream": False, "think": False, "keep_alive": "5m",
                   "logprobs": True, "top_logprobs": N_PROBS,
                   "options": {"num_ctx": CONTEXT, "num_predict": max_tokens, "temperature": temperature,
                               "seed": seed}}
        try:
            r = self.http.post_json(f"{self.host}/api/generate", payload, timeout=900)
        except Exception as e:
            return self.row("ollama", model, test_id, variant, "1/1", True, False, "", None, None, None,
                            time.monotonic() - t0, f"{type(e).__name__}: {e}")
        toks = ollama_tokens(r)
        if self.ollama_logprobs is None or toks:
            self.ollama_logprobs = toks is not None
        self.tokens[(model, test_id, variant, "1/1")] = toks
        self.prompt_tokens.setdefault((model, test_id), {})["ollama"] = r.get("prompt_eval_count")
        self.event("ollama_raw", model=model, test=test_id, variant=variant,
                   request={k: v for k, v in payload.items() if k != "prompt"},
                   response={k: v for k, v in r.items() if k not in ("context",)})

        class _G:                                       # jak Generation – reszta kodu bez zmian
            text = str(r.get("response") or "")
            prompt_tokens = int(r.get("prompt_eval_count") or 0)
            prompt_cached = None if r.get("prompt_eval_cached_count") is None else int(r["prompt_eval_cached_count"])
        g = _G
        cached = g.prompt_cached
        if variant == "O0":                           # od zera: zwolnienie potwierdzone i nic z pamięci
            ok = bool(released and released[0]) and not cached
        else:                                         # z pamięci: musiało coś przyjść z pamięci
            ok = cached is None or cached > 0
        processed = None if cached is None else g.prompt_tokens - cached
        return self.row("ollama", model, test_id, variant, "1/1", True, ok, g.text, g.prompt_tokens, cached,
                        processed, time.monotonic() - t0,
                        released="—" if released is None else ("tak" if released[0] else "NIE"))

    def llama_attempt(self, srv: LlamaServer, model: str, test_id: str, variant: str, label: str,
                      counted: bool, formatted: str) -> dict:
        _, max_tokens, temperature, seed = self.cfg[test_id]
        cache = variant == "E"
        payload = {"prompt": formatted, "n_predict": max_tokens, "temperature": temperature, "seed": seed,
                   "cache_prompt": cache, "stream": False, "n_probs": N_PROBS}
        t0 = time.monotonic()
        try:
            r = srv.request("POST", "/completion", payload)
        except Exception as e:
            return self.row("llama.cpp", model, test_id, variant, label, counted, False, "", None, None, None,
                            time.monotonic() - t0, f"{type(e).__name__}: {e}")
        timings = r.get("timings") or {}
        prompt_tokens = r.get("tokens_evaluated")
        processed = timings.get("prompt_n")
        cached = timings.get("cache_n", r.get("tokens_cached"))
        if processed is None or prompt_tokens is None:
            ok = True                                  # serwer nie podaje – zapisujemy null, bez oceny warunku
        elif cache:
            ok = processed < prompt_tokens             # E: prompt z pamięci
        else:
            ok = processed >= prompt_tokens            # F: cały prompt liczony od nowa
        row = self.row("llama.cpp", model, test_id, variant, label, counted, ok, str(r.get("content") or ""),
                       prompt_tokens, cached, processed, time.monotonic() - t0, ngl=srv.ngl)
        self.tokens[(model, test_id, variant, label)] = llama_tokens(r)
        self.event("llama_raw", order=row["kolejnosc"], request=payload, response=r)
        return row

    # -------------------------------------------------- model
    def run_model(self, model: str, task_ids: list[str], manifest: dict) -> None:
        path, blocks, show = gguf_of(self.http, self.host, model)
        entry = {"gguf": path, "warstwy": blocks, "show": show}
        manifest["modele"][model] = entry
        if not readable(path):
            self.echo(f"Pomijam {model}: nie mogę odczytać pliku GGUF ({path}) – brak pliku albo uprawnień.")
            entry["status"] = "brak dostępu do pliku GGUF"
            return
        # --- Ollama: O0 (od zera) i O1 (z pamięci) dla każdego zadania
        share = None
        for test_id in task_ids:
            self.echo(f"{model} · {test_id} · Ollama O0/O1")
            released = release(self.bench, self.http, self.host, model, self.unload_timeout, self.sleep)
            self.ollama_attempt(model, test_id, "O0", released)
            if share is None:
                try:
                    share = self.bench.gpu_share(model)
                except Exception:
                    share = None
            self.ollama_attempt(model, test_id, "O1", None)
        entry["udzial_w_karcie_ollama"] = share
        freed = release(self.bench, self.http, self.host, model, self.unload_timeout, self.sleep)
        entry["ollama_zwolniona_przed_llama"] = freed[0]
        # --- llama.cpp: E i F
        ngl = ngl_for(share, blocks)
        srv = None
        for attempt in range(3):                       # brak pamięci karty → mniej warstw
            log = self.out / f"llama-server-{sha(model)}-{attempt}.log"
            srv = LlamaServer(self.binary, path, self.port, ngl, self.batch, self.ubatch, log)
            self.echo(f"{model} · llama-server, -ngl {ngl}")
            if srv.start(self.start_timeout, self.sleep):
                break
            tail = srv.log_tail()
            srv.stop()
            self.event("server_failed", model=model, ngl=ngl, log_tail=tail)
            self.echo(f"  ! llama-server nie wystartował (-ngl {ngl}). Koniec dziennika:\n{tail}")
            srv = None
            ngl = max(0, (ngl if ngl != 999 else (blocks or 40)) - 4)
        entry.update({"ngl": ngl, "llama_server_args": srv.args if srv else None})
        if srv is None:
            entry["status"] = "llama-server nie wystartował"
            return
        try:
            props = srv.request("GET", "/props", timeout=10)
            entry["llama_props"] = {k: props.get(k) for k in ("build_info", "default_generation_settings",
                                                              "total_slots", "chat_template") if k in props}
        except Exception:
            pass
        try:
            for test_id in task_ids:
                prompt = self.tests[test_id]["prompt"]
                try:                                   # szablon czatu z pliku GGUF – ten sam dla E i F
                    formatted = srv.request("POST", "/apply-template", {
                        "messages": [{"role": "user", "content": prompt}],
                        "chat_template_kwargs": {"enable_thinking": False}}).get("prompt")
                except Exception as e:
                    self.echo(f"  ! /apply-template nie działa ({e}) – potrzebny nowszy llama-server.")
                    entry["status"] = "brak /apply-template"
                    return
                formatted, ids, pieces, stripped = strip_double_bos(srv, formatted)
                self.prompt_tokens.setdefault((model, test_id), {})["llama"] = len(ids)
                self.event("template", model=model, test=test_id, formatted=formatted, bos_stripped=stripped,
                           token_ids=ids, token_pieces=pieces)
                self.echo(f"{model} · {test_id} · llama.cpp E/F")
                self.llama_attempt(srv, model, test_id, "E", "przygotowawcza", False, formatted)
                for n in range(1, self.repeats + 1):
                    self.llama_attempt(srv, model, test_id, "E", f"{n}/{self.repeats}", True, formatted)
                for n in range(1, self.repeats + 1):
                    self.llama_attempt(srv, model, test_id, "F", f"{n}/{self.repeats}", True, formatted)
            entry["status"] = "zmierzony"
        finally:
            srv.stop()

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        tags = self.http.get_json(f"{self.host}/api/tags")
        available = [m.get("name") or m.get("model") for m in (tags.get("models") or [])]
        plan = self.plan(available)
        manifest = {"narzedzie": "tools/eksperyment_llamacpp.py", "bobr": __version__, "start": now_iso(),
                    "system": platform.platform(), "python": sys.version.split()[0],
                    "ollama": self.bench.version(), "llama_server": llama_version(self.binary),
                    "llama_server_bin": self.binary, "zakres": self.scope, "kontekst": CONTEXT,
                    "batch": self.batch, "ubatch": self.ubatch, "parallel": 1, "powtorzenia": self.repeats,
                    "warianty": {"O0": "Ollama, model zwolniony, od zera", "O1": "Ollama, od razu drugi raz",
                                 "E": "llama.cpp cache_prompt true, po przebiegu przygotowawczym",
                                 "F": "llama.cpp cache_prompt false"},
                    "plan": [{"model": m, "zadania": t} for m, t in plan], "modele": {}, "status": "w toku"}
        with open(self.out / "zdarzenia.jsonl", "w", encoding="utf-8") as self._events:
            try:
                self.event("start", manifest=manifest)
                for model, task_ids in plan:
                    for other in available:            # jeden model naraz – karta wolna
                        if other != model:
                            try:
                                if self.bench.is_loaded(other):
                                    release(self.bench, self.http, self.host, other, self.unload_timeout, self.sleep)
                            except Exception:
                                pass
                    self.run_model(model, task_ids, manifest)
                manifest["status"] = "zakończony"
            except KeyboardInterrupt:
                manifest["status"] = "przerwany (Ctrl+C)"
                self.echo("\nPrzerwano – zapisuję to, co już zmierzone.")
            finally:
                manifest["koniec"] = now_iso()
                self.event("end", status=manifest["status"])
        manifest["ollama_logprobs"] = self.ollama_logprobs
        summary = summarize(self.rows, self.tokens, self.prompt_tokens)
        write_csv(self.out / "proby.csv", self.rows)
        write_csv(self.out / "podsumowanie.csv", summary)
        manifest["wnioski"] = conclusions(summary)
        manifest["pliki"] = ["zdarzenia.jsonl", "proby.csv", "podsumowanie.csv", "manifest.json"]
        (self.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"manifest": manifest, "summary": summary}


# ---------------------------------------------------------------------- podsumowanie
def _modal(rows: list[dict]) -> tuple[Optional[str], int, int]:
    texts = [r["odpowiedz"] for r in rows]
    if not texts:
        return None, 0, 0
    text, count = Counter(texts).most_common(1)[0]
    return text, count, len(set(texts))


def _same(a: Optional[str], b: Optional[str]) -> str:
    return "—" if a is None or b is None else ("tak" if a == b else "NIE")


def summarize(rows: list[dict], tokens: Optional[dict] = None, prompt_tokens: Optional[dict] = None) -> list[dict]:
    tokens, prompt_tokens = tokens or {}, prompt_tokens or {}
    pairs: dict = {}
    for r in rows:
        pairs.setdefault((r["model"], r["zadanie"]), []).append(r)
    out = []
    for (model, test_id), rs in pairs.items():
        def pick(variant, counted=True):
            return [r for r in rs if r["wariant"] == variant and (r["wliczona"] == "tak") == counted
                    and (counted or r["proba"] == "przygotowawcza")]
        o0, o1, e, f = pick("O0"), pick("O1"), pick("E"), pick("F")
        warm = pick("E", counted=False)
        e_text, e_n, e_dist = _modal(e)
        f_text, f_n, f_dist = _modal(f)
        cold = warm[0]["odpowiedz"] if warm else None
        o0_text = o0[0]["odpowiedz"] if o0 else None
        o1_text = o1[0]["odpowiedz"] if o1 else None
        # Pierwsza rozbieżność: pierwsza wliczona próba E kontra pierwsza F; O0 kontra O1.
        def toks(variant, group):
            return tokens.get((model, test_id, variant, group[0]["proba"])) if group else None
        div_l = first_divergence(toks("E", e), toks("F", f)) if e_text != f_text else None
        div_o = first_divergence(toks("O0", o0), toks("O1", o1)) if o0_text != o1_text else None
        n = prompt_tokens.get((model, test_id), {})
        same_n = "—" if n.get("llama") is None or n.get("ollama") is None else (
            "tak" if n["llama"] == n["ollama"] else "NIE")
        out.append({
            "model": model, "zadanie": test_id,
            "ollama_O0_zaliczona": o0[0]["zaliczona"] if o0 else "—",
            "ollama_O1_zaliczona": o1[0]["zaliczona"] if o1 else "—",
            "ollama_O0_jak_O1": _same(o0_text, o1_text),
            "llama_E_najczestsza": f"{e_n}/{len(e)}" if e else "—", "llama_E_roznych": e_dist,
            "llama_E_zaliczone": f"{sum(r['zaliczona'] == 'tak' for r in e)}/{len(e)}" if e else "—",
            "llama_F_najczestsza": f"{f_n}/{len(f)}" if f else "—", "llama_F_roznych": f_dist,
            "llama_F_zaliczone": f"{sum(r['zaliczona'] == 'tak' for r in f)}/{len(f)}" if f else "—",
            "llama_E_jak_F": _same(e_text, f_text),
            "llama_F_jak_pierwsze_zapytanie": _same(f_text, cold),
            "llama_F_jak_ollama_O0": _same(f_text, o0_text),
            "llama_E_jak_ollama_O1": _same(e_text, o1_text),
            "tokeny_llama": null(n.get("llama")), "tokeny_ollama": null(n.get("ollama")), "tokeny_jak_ollama": same_n,
            "llama_rozbieznosc_pozycja": "—" if not div_l else div_l["pozycja"],
            "llama_rozbieznosc_E_top2": top2(div_l["a"]) if div_l else "—",
            "llama_rozbieznosc_F_top2": top2(div_l["b"]) if div_l else "—",
            "ollama_rozbieznosc_pozycja": "—" if not div_o else div_o["pozycja"],
            "ollama_rozbieznosc_O0_top2": top2(div_o["a"]) if div_o else "—",
            "ollama_rozbieznosc_O1_top2": top2(div_o["b"]) if div_o else "—",
            "pominiete": sum(r["wliczona"] == "nie" and r["proba"] != "przygotowawcza" for r in rs),
            "skrot_E": sha(e_text) if e_text is not None else "—",
            "skrot_F": sha(f_text) if f_text is not None else "—",
        })
    return out


def conclusions(summary: list[dict]) -> dict:
    def count(key, value="NIE"):
        return sum(s[key] == value for s in summary)
    measured = [s for s in summary if s["llama_E_jak_F"] != "—"]
    return {
        "par": len(summary),
        "par_zmierzonych_w_llama": len(measured),
        "ollama_pamiec_zmienia_odpowiedz": count("ollama_O0_jak_O1"),
        "llama_pamiec_zmienia_odpowiedz": count("llama_E_jak_F"),
        "llama_F_stabilne": sum(s["llama_F_roznych"] == 1 for s in measured),
        "llama_E_stabilne": sum(s["llama_E_roznych"] == 1 for s in measured),
        "llama_F_jak_pierwsze_zapytanie": count("llama_F_jak_pierwsze_zapytanie", "tak"),
        "llama_rozbieznosci_z_tokenami": sum(s["llama_rozbieznosc_pozycja"] != "—" for s in summary),
        "ollama_rozbieznosci_z_tokenami": sum(s["ollama_rozbieznosc_pozycja"] != "—" for s in summary),
        "par_z_ta_sama_liczba_tokenow": count("tokeny_jak_ollama", "tak"),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llama-server", default=shutil.which("llama-server") or "llama-server",
                    help="ścieżka do llama-server (domyślnie z PATH)")
    ap.add_argument("--ollama", default=OLLAMA)
    ap.add_argument("--zakres", choices=("pelny", "pary"), default="pelny",
                    help="pelny: wszystkie modele z Ollamy × 15 zadań; pary: 10 par z eksperymentu A–D")
    ap.add_argument("--models", nargs="*", help="tylko te modele (nazwy jak w ollama list)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--ubatch", type=int, default=BATCH)
    ap.add_argument("--out", type=Path, help="folder wyników (domyślnie eksperyment-llamacpp-RRRRMMDD-GGMM)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or Path(f"eksperyment-llamacpp-{datetime.datetime.now():%Y%m%d-%H%M}")
    exp = Experiment(out, args.llama_server, args.ollama, args.port, args.repeats, args.batch, args.ubatch,
                     args.models, args.zakres)
    version = llama_version(args.llama_server)
    if not version:
        print(f"Nie mogę uruchomić llama-server ({args.llama_server}). Podaj ścieżkę: --llama-server /ścieżka")
        return 1
    try:
        tags = exp.http.get_json(f"{exp.host}/api/tags")
        print(f"Ollama {exp.bench.version()} · llama-server: {version}")
    except Exception as e:
        print(f"Nie mogę połączyć się z Ollamą pod {exp.host}: {e}")
        return 1
    available = [m.get("name") or m.get("model") for m in (tags.get("models") or [])]
    plan = exp.plan(available)
    tasks = sum(len(t) for _, t in plan)
    print(f"Plan ({args.zakres}): {len(plan)} modeli, {tasks} par model–zadanie; na parę: Ollama 2 próby, "
          f"llama.cpp {1 + 2 * args.repeats} zapytań → razem {tasks * (3 + 2 * args.repeats)} zapytań.")
    for model, task_ids in plan:
        path, blocks, _ = gguf_of(exp.http, exp.host, model)
        ok = "✓" if readable(path) else "✗ brak pliku albo uprawnień"
        print(f"  {ok}  {model}  ({len(task_ids)} zadań, warstw: {blocks})  {path}")
    if args.dry_run:
        return 0
    result = exp.run()
    print("\nWnioski (liczby par):")
    for k, v in result["manifest"]["wnioski"].items():
        print(f"  {k}: {v}")
    print(f"\nStatus: {result['manifest']['status']}. Wyniki: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
