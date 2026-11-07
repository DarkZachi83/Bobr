"""Eksperyment kontrolny: strumieniowanie i pamięć promptu (cache) w testach jakości.

    python tools/eksperyment_cache.py --dry-run      # sam plan, bez zapytań
    python tools/eksperyment_cache.py                # cały eksperyment (ok. 15–25 min)

Skąd ten skrypt: w benchmarku 0.8 (5 powtórzeń) w 21 ze 165 zadań wyszło „1/5 identycznych”
i ani razu 2, 3 ani 4 z 5. Pierwszy przebieg różni się od kolejnych dwiema rzeczami naraz:
idzie strumieniem (podgląd na żywo) i trafia na model bez zapamiętanego promptu. Tu te dwa
warunki rozdzielamy:

    A  bez strumienia  model załadowany, przebieg przygotowawczy, potem 5 prób
    B  strumień        model załadowany, przebieg przygotowawczy, potem 5 prób
    C  bez strumienia  przed każdą próbą model zwolniony z pamięci (potwierdzone w /api/ps)
    D  strumień        przed każdą próbą model zwolniony z pamięci (potwierdzone w /api/ps)

Zapytania idą przez kod Bobra (OllamaBench.generate + HttpClient) – sprawdzamy ten sam
sposób odbioru co w programie. Klient zapisuje przy tym surowe bajty każdej linii odpowiedzi,
więc złożenie strumienia przez Bobra da się porównać z niezależnym złożeniem z surowych danych.

Wynik w osobnym folderze (program i baza danych Bobra się nie zmieniają):
    zdarzenia.jsonl     wszystko po kolei: zapytania, surowe linie, odpowiedzi, zwalnianie
    proby.csv           każda próba osobno (także przygotowawcze – oznaczone, niewliczane)
    podsumowanie.csv    każda para model–zadanie–wariant: zaliczenia, zgodność, cache
    manifest.json       konfiguracja, wersje, plan i to, czy każda próba spełniła warunki

Zasady zapisu: brak pola w odpowiedzi Ollamy to null („null” w CSV), nigdy 0. Próba w C/D,
w której modelu nie udało się zwolnić w limicie czasu, jest oznaczona jako niespełniająca
warunków wariantu i nie wchodzi do podsumowania (licznik „pominięte”).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aibobr import __version__                                    # noqa: E402
from aibobr.bench.ollama import OllamaBench, TIMEOUT                # noqa: E402
from aibobr.bench.quality import CONTEXT, evaluate                  # noqa: E402
from aibobr.sources.http import HttpClient                          # noqa: E402

HOST = "http://localhost:11434"
SUITES = ("pl_jezyk_v1", "en_format_v1")
VARIANTS = {                      # wariant: (strumień, model zwalniany przed każdą próbą)
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
    "D": (True, True),
}
# Zatwierdzony plan: 3 modele, 10 par model–zadanie. Zadania z rozbieżnością „1/5” w benchmarku
# 0.8 oraz pl_masculine_personal jako kontrola (wszędzie 5/5). gemma2:9b Q4_K_M – model
# kontrolny (15/15 identycznych w benchmarku).
PLAN = [
    ("hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M",
     ["pl_henrykowska", "pl_diacritics_restore", "json_hardware", "pl_masculine_personal"]),
    ("gemma2:9b",
     ["pl_diacritics_restore", "pl_henrykowska", "py_fibonacci", "pl_masculine_personal"]),
    ("gemma2:9b-instruct-q4_K_M",
     ["pl_diacritics_restore", "pl_masculine_personal"]),
]
STATS = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_cached_count",
         "prompt_eval_duration", "eval_count", "eval_duration", "done_reason")


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def null(value: Any) -> Any:
    """Do CSV: brak wartości to „null”, nie pusta komórka ani 0."""
    return "null" if value is None else value


# ---------------------------------------------------------------------- klient z zapisem
class _Tee:
    """Odpowiedź urllib, która przy czytaniu zapisuje surowe bajty (linia po linii albo całość)."""

    def __init__(self, resp, sink: list):
        self._resp, self._sink = resp, sink

    def __enter__(self):
        self._resp.__enter__()
        return self

    def __exit__(self, *exc):
        return self._resp.__exit__(*exc)

    def __iter__(self):
        for line in self._resp:
            self._sink.append((time.monotonic(), line))
            yield line

    def read(self, *args):
        data = self._resp.read(*args)
        self._sink.append((time.monotonic(), data))
        return data

    def __getattr__(self, name):
        return getattr(self._resp, name)


class RecordingHttp(HttpClient):
    """HttpClient Bobra bez zmian w logice – tylko zapamiętuje ciało zapytania i surową odpowiedź."""

    def __init__(self, timeout: float = TIMEOUT):
        super().__init__(timeout=timeout)
        self.request: Optional[dict] = None
        self.raw: list = []

    def reset(self) -> None:
        self.request, self.raw = None, []

    def _request(self, url, headers=None, data=None, timeout=None):
        self.request = {"url": url, "body": json.loads(data.decode("utf-8")) if data else None,
                        "t": time.monotonic()}
        return _Tee(super()._request(url, headers, data, timeout), self.raw)


# ---------------------------------------------------------------------- analiza surowej odpowiedzi
def inspect_raw(raw: list, stream: bool) -> dict:
    """Niezależnie od Bobra: parsowanie każdej linii, obecność done, tekst złożony z surowych danych."""
    lines, errors, pieces, final = [], 0, [], None
    for t, data in raw:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            errors += 1
            lines.append({"t": t, "raw": text, "parsed": False})
            continue
        lines.append({"t": t, "raw": text, "parsed": True})
        if stream:
            if obj.get("response"):
                pieces.append(obj["response"])
            if obj.get("done"):
                final = obj
        else:
            final = obj
            pieces.append(str(obj.get("response") or ""))
    return {"lines": lines, "parse_errors": errors, "done_seen": bool(final and final.get("done")),
            "text": "".join(pieces), "final": final, "chunks": len(pieces)}


# ---------------------------------------------------------------------- Ollama
def loaded(http: HttpClient, host: str, model: str) -> bool:
    data = http.get_json(f"{host}/api/ps")
    return any(m.get("name") == model or m.get("model") == model
               for m in (data.get("models") or []) if isinstance(data, dict))


def release(bench: OllamaBench, http: HttpClient, host: str, model: str, timeout: float,
            sleep: Callable[[float], None] = time.sleep) -> tuple[bool, float, int]:
    """keep_alive 0, potem /api/ps co 0,5 s aż modelu nie ma albo minie limit.
    (potwierdzone, sekundy czekania, liczba sprawdzeń)"""
    start = time.monotonic()
    bench.unload(model)
    checks = 0
    while True:
        checks += 1
        if not loaded(http, host, model):
            return True, time.monotonic() - start, checks
        if time.monotonic() - start >= timeout:
            return False, time.monotonic() - start, checks
        sleep(0.5)


def load_tests(root: Path) -> tuple[dict, dict]:
    """{id zadania: test}, {id zadania: (id pakietu, max_tokens, temperatura, seed)}"""
    tests, cfg = {}, {}
    for sid in SUITES:
        suite = json.loads((root / "data" / "benchmarks" / f"{sid}.json").read_text(encoding="utf-8"))
        gen = suite.get("default_generation_config", {})
        for test in suite["tests"]:
            tests[test["id"]] = test
            cfg[test["id"]] = (sid, int(gen.get("max_tokens", 256)), float(gen.get("temperature", 0)),
                               int(gen.get("seed", 42)))
    return tests, cfg


# ---------------------------------------------------------------------- eksperyment
class Experiment:
    def __init__(self, host: str, out: Path, repeats: int = 5, unload_timeout: float = 30.0,
                 plan=PLAN, http: Optional[RecordingHttp] = None, root: Optional[Path] = None,
                 sleep: Callable[[float], None] = time.sleep, echo: Callable[[str], None] = print):
        self.host, self.out, self.repeats = host.rstrip("/"), out, repeats
        self.unload_timeout, self.plan, self.sleep, self.echo = unload_timeout, plan, sleep, echo
        self.http = http or RecordingHttp()
        self.bench = OllamaBench(self.host, http=self.http)
        self.tests, self.cfg = load_tests(root or Path(__file__).resolve().parent.parent)
        self.rows: list[dict] = []
        self.order = 0
        self._events = None

    # -------------------------------------------------- zapis
    def event(self, kind: str, **data) -> None:
        self._events.write(json.dumps({"type": kind, "time": now_iso(), **data}, ensure_ascii=False) + "\n")
        self._events.flush()

    # -------------------------------------------------- jedna próba
    def attempt(self, model: str, test_id: str, variant: str, n: int, warmup: bool = False,
                released: Optional[tuple] = None) -> dict:
        stream, _ = VARIANTS[variant]
        test = self.tests[test_id]
        sid, max_tokens, temperature, seed = self.cfg[test_id]
        self.order += 1
        self.http.reset()
        received: list[str] = []
        started = now_iso()
        t0 = time.monotonic()
        error = None
        try:
            g = self.bench.generate(model, test["prompt"], CONTEXT, max_tokens, temperature=temperature,
                                    seed=seed, on_chunk=received.append if stream else None)
            bobr_text = g.text
        except Exception as e:                       # zapisujemy i idziemy dalej
            bobr_text, error = "", f"{type(e).__name__}: {e}"
        seconds = time.monotonic() - t0
        info = inspect_raw(self.http.raw, stream)
        final = info["final"] or {}
        verdict = evaluate(bobr_text, test["evaluator"]) if not error else None
        conditions = error is None and info["done_seen"] and (released is None or released[0])
        row = {
            "kolejnosc": self.order, "czas": started, "model": model, "zadanie": test_id, "pakiet": sid,
            "wariant": variant, "proba": "przygotowawcza" if warmup else f"{n}/{self.repeats}",
            "wliczona": "nie" if warmup else ("tak" if conditions else "nie"),
            "strumien": "tak" if stream else "nie",
            "model_zwolniony": "—" if released is None else ("tak" if released[0] else "NIE"),
            "czekanie_na_zwolnienie_s": "—" if released is None else round(released[1], 2),
            "warunki_spelnione": "tak" if conditions else "nie",
            "zaliczona": "—" if verdict is None else ("tak" if verdict.passed else "nie"),
            "powod": "" if verdict is None or verdict.passed else verdict.reason,
            "blad": error or "",
            **{k: null(final.get(k)) for k in STATS},
            "kawalki_strumienia": info["chunks"] if stream else "—",
            "kawalki_w_bobrze": len(received) if stream else "—",
            "bledy_parsowania": info["parse_errors"],
            "done": "tak" if info["done_seen"] else "nie",
            "tekst_bobra_jak_surowy": "tak" if bobr_text == info["text"] else "NIE",
            "czas_s": round(seconds, 3),
            "skrot_odpowiedzi": sha(bobr_text),
            "odpowiedz": bobr_text,
        }
        self.event("attempt", order=self.order, model=model, test=test_id, variant=variant,
                   attempt=None if warmup else n, warmup=warmup, stream=stream,
                   request=self.http.request["body"] if self.http.request else None,
                   raw=[{"dt": round(ln["t"] - t0, 4), "line": ln["raw"], "parsed": ln["parsed"]}
                        for ln in info["lines"]],
                   bobr_text=bobr_text, raw_text=info["text"], final=final, error=error,
                   released=None if released is None else {"confirmed": released[0],
                                                           "seconds": round(released[1], 3),
                                                           "checks": released[2]},
                   conditions_met=conditions, passed=None if verdict is None else verdict.passed)
        self.rows.append(row)
        return row

    def release(self, model: str) -> tuple:
        result = release(self.bench, self.http, self.host, model, self.unload_timeout, self.sleep)
        self.event("release", model=model, confirmed=result[0], seconds=round(result[1], 3), checks=result[2])
        if not result[0]:
            self.echo(f"  ! {model}: nie zwolniony po {self.unload_timeout:.0f} s – próba niewliczona")
        return result

    # -------------------------------------------------- całość
    def models_info(self) -> dict:
        data = self.http.get_json(f"{self.host}/api/tags")
        return {m.get("name") or m.get("model"): {"digest": m.get("digest"), "size": m.get("size"),
                                                  "details": m.get("details")}
                for m in (data.get("models") or []) if isinstance(data, dict)}

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        manifest = {"narzedzie": "tools/eksperyment_cache.py", "bobr": __version__, "start": now_iso(),
                    "host": self.host, "python": sys.version.split()[0], "system": platform.platform(),
                    "kontekst": CONTEXT, "think": False, "keep_alive": "5m", "powtorzenia": self.repeats,
                    "limit_zwalniania_s": self.unload_timeout,
                    "warianty": {v: {"strumien": s, "zwalnianie_przed_proba": u} for v, (s, u) in VARIANTS.items()},
                    "przebieg_przygotowawczy": "A i B: jedno zapytanie identyczne z próbami wariantu, niewliczane",
                    "kolejnosc": "dla każdej pary model–zadanie warianty A, B, C, D po kolei",
                    "plan": [{"model": m, "zadania": t} for m, t in self.plan], "status": "w toku"}
        with open(self.out / "zdarzenia.jsonl", "w", encoding="utf-8") as self._events:
            try:
                manifest["ollama"] = self.bench.version()
                info = self.models_info()
                manifest["modele"] = {m: info.get(m) for m, _ in self.plan}
                self.event("start", manifest=manifest)
                for model, task_ids in self.plan:
                    if model not in info:
                        self.echo(f"Pomijam {model}: brak w Ollamie (ollama pull {model}).")
                        self.event("skip_model", model=model)
                        continue
                    for other in [m for m, _ in self.plan if m != model]:
                        if other in info and loaded(self.http, self.host, other):
                            self.release(other)           # jeden model naraz
                    for test_id in task_ids:
                        self.pair(model, test_id)
                manifest["status"] = "zakończony"
            except KeyboardInterrupt:
                manifest["status"] = "przerwany (Ctrl+C)"
                self.echo("\nPrzerwano – zapisuję to, co już zmierzone.")
            finally:
                manifest["koniec"] = now_iso()
                manifest["proby"] = [{k: r[k] for k in ("kolejnosc", "model", "zadanie", "wariant", "proba",
                                                         "wliczona", "warunki_spelnione", "model_zwolniony")}
                                     for r in self.rows]
                self.event("end", status=manifest["status"])
        summary = summarize(self.rows)
        write_csv(self.out / "proby.csv", self.rows)
        write_csv(self.out / "podsumowanie.csv", summary)
        manifest["pliki"] = ["zdarzenia.jsonl", "proby.csv", "podsumowanie.csv", "manifest.json"]
        (self.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"manifest": manifest, "summary": summary}

    def pair(self, model: str, test_id: str) -> None:
        for variant, (stream, unload) in VARIANTS.items():
            self.echo(f"{model} · {test_id} · {variant}")
            if not unload:
                self.attempt(model, test_id, variant, 0, warmup=True)
                for n in range(1, self.repeats + 1):
                    self.attempt(model, test_id, variant, n)
            else:
                for n in range(1, self.repeats + 1):
                    self.attempt(model, test_id, variant, n, released=self.release(model))


# ---------------------------------------------------------------------- podsumowanie
def summarize(rows: list[dict]) -> list[dict]:
    groups: dict = {}
    for r in rows:
        if r["proba"] == "przygotowawcza":
            continue
        groups.setdefault((r["model"], r["zadanie"]), {}).setdefault(r["wariant"], []).append(r)
    out = []
    for (model, test_id), variants in groups.items():
        modal_a = None
        for variant in VARIANTS:
            tries = variants.get(variant, [])
            valid = [r for r in tries if r["wliczona"] == "tak"]
            texts = [r["odpowiedz"] for r in valid]
            common, count = Counter(texts).most_common(1)[0] if texts else (None, 0)
            if variant == "A":
                modal_a = common
            passes = sum(r["zaliczona"] == "tak" for r in valid)
            cached = [r["prompt_eval_cached_count"] for r in valid if r["prompt_eval_cached_count"] != "null"]
            out.append({
                "model": model, "zadanie": test_id, "wariant": variant,
                "strumien": "tak" if VARIANTS[variant][0] else "nie",
                "model_zwalniany": "tak" if VARIANTS[variant][1] else "nie",
                "proby_wliczone": len(valid), "pominiete": len(tries) - len(valid),
                "zaliczone": f"{passes}/{len(valid)}",
                "werdykt_wiekszosciowy": "—" if not valid else ("zaliczone" if passes * 2 > len(valid) else "niezaliczone"),
                "proba_1": "—" if not valid else valid[0]["zaliczona"],
                "zgodne_z_proba_1": f"{sum(t == texts[0] for t in texts)}/{len(texts)}" if texts else "—",
                "najczestsza_odpowiedz": f"{count}/{len(texts)}" if texts else "—",
                "roznych_odpowiedzi": len(set(texts)),
                "najczestsza_jak_w_A": "—" if variant == "A" or common is None or modal_a is None
                                       else ("tak" if common == modal_a else "NIE"),
                "skrot_najczestszej": sha(common) if common is not None else "—",
                "prompt_cached_min": min(cached) if cached else "null",
                "prompt_cached_max": max(cached) if cached else "null",
                "problemy_odbioru": sum(r["bledy_parsowania"] > 0 or r["done"] != "tak"
                                        or r["tekst_bobra_jak_surowy"] != "tak" for r in tries),
            })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    """Średnik i BOM – jak eksporty Bobra (Excel z polskimi ustawieniami)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def print_summary(summary: list[dict], echo=print) -> None:
    echo("\nmodel · zadanie: wariant = zgodne z próbą 1 / najczęstsza / zaliczone (pominięte)")
    pairs: dict = {}
    for s in summary:
        pairs.setdefault((s["model"], s["zadanie"]), []).append(
            f"{s['wariant']}={s['zgodne_z_proba_1']} / {s['najczestsza_odpowiedz']} / {s['zaliczone']}"
            + (f" ({s['pominiete']})" if s["pominiete"] else ""))
    for (model, test_id), cells in pairs.items():
        echo(f"{model.split('/')[-1]} · {test_id}:  " + "   ".join(cells))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--out", type=Path, help="folder wyników (domyślnie eksperyment-cache-RRRRMMDD-GGMM)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--unload-timeout", type=float, default=30.0, help="ile sekund czekać na zwolnienie modelu")
    ap.add_argument("--dry-run", action="store_true", help="pokaż plan i sprawdź modele, bez zapytań do modeli")
    args = ap.parse_args(argv)
    out = args.out or Path(f"eksperyment-cache-{datetime.datetime.now():%Y%m%d-%H%M}")
    exp = Experiment(args.host, out, repeats=args.repeats, unload_timeout=args.unload_timeout)
    pairs = sum(len(t) for _, t in PLAN)
    print(f"Plan: {len(PLAN)} modele, {pairs} par model–zadanie, warianty A–D × {args.repeats} prób "
          f"= {pairs * 4 * args.repeats} prób (+ {pairs * 2} przygotowawczych w A i B).")
    try:
        info = exp.models_info()
        print(f"Ollama {exp.bench.version()} pod {exp.host}")
    except Exception as e:
        print(f"Nie mogę połączyć się z Ollamą pod {exp.host}: {e}")
        return 1
    for model, tasks in PLAN:
        print(f"  {'✓' if model in info else '✗ brak'}  {model}: {', '.join(tasks)}")
    if args.dry_run:
        return 0
    result = exp.run()
    print_summary(result["summary"])
    print(f"\nStatus: {result['manifest']['status']}. Wyniki: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
