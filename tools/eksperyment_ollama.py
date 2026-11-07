"""Eksperymenty kontrolne w samej Ollamie: wpływ opcji logprobs i brak tokenu początku w szablonie.

    python tools/eksperyment_ollama.py logprobs --dry-run
    python tools/eksperyment_ollama.py logprobs            # 10 par z A–D, logprobs bez / z (ok. 5–10 min)
    python tools/eksperyment_ollama.py bos                 # Minitron, 15 zadań, z <s> / bez (ok. 8–12 min)

logprobs – skąd: w eksperymencie llama.cpp (7 listopada) Ollama zapytana z logprobs dała różnicę
„od zera / z pamięci” w 1 z 10 par, a dzień wcześniej bez logprobs w 6 z 10 – przy tej samej wersji.
Tu oba ustawienia są w jednym przebiegu, na zmianę, po N razy:
    O0  model zwolniony (potwierdzone w /api/ps), prompt od zera
    O1  to samo zadanie od razu drugi raz (prompt z pamięci)
dla logprobs: false i logprobs: true. Wynik: czy sama prośba o logprobs zmienia odpowiedź.

bos – skąd: szablon Ollamy dla Minitrona (hf.co/…) nie ma tokenu początku, a szablon w pliku GGUF
ma „{{bos_token}}” (sprawdzone w trybie raw: bez <s> 45 tokenów, z <s> 46). Warianty, każdy po
zwolnieniu modelu (prompt od zera), bez logprobs – tak jak benchmark Bobra:
    szablon   zwykłe zapytanie, szablon Ollamy
    raw       tryb raw, ten sam tekst co szablon Ollamy (kontrola: musi dać to samo co „szablon”)
    raw+bos   tryb raw, z <s> na początku – tak jak szablon z pliku GGUF
Wynik: ile zadań model zalicza z tokenem początku i bez.

Pliki: zdarzenia.jsonl (surowe zapytania i odpowiedzi), proby.csv, podsumowanie.csv, manifest.json.
Brak pola w odpowiedzi to null, nigdy 0. Próba bez potwierdzonego zwolnienia albo z pamięcią promptu
tam, gdzie miało jej nie być, nie wchodzi do podsumowania.
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aibobr import __version__                                    # noqa: E402
from aibobr.bench.ollama import OllamaBench                         # noqa: E402
from aibobr.bench.quality import CONTEXT, evaluate                  # noqa: E402
from aibobr.sources.http import HttpClient                          # noqa: E402
from eksperyment_cache import PLAN as PAIRS, load_tests, now_iso, null, release, sha, write_csv  # noqa: E402
from eksperyment_llamacpp import N_PROBS, first_divergence, has_bos, ollama_tokens, top2  # noqa: E402

OLLAMA = "http://localhost:11434"
MINITRON = "hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M"
SUITES_ORDER = ("pl_jezyk_v1", "en_format_v1")
CHATML = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


class Experiment:
    def __init__(self, out: Path, ollama: str = OLLAMA, repeats: int = 3, unload_timeout: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep, echo: Callable[[str], None] = print,
                 root: Optional[Path] = None):
        self.out, self.repeats, self.unload_timeout, self.sleep, self.echo = out, repeats, unload_timeout, sleep, echo
        self.host = ollama.rstrip("/")
        self.http = HttpClient(timeout=900)
        self.bench = OllamaBench(self.host, http=self.http)
        self.tests, self.cfg = load_tests(root or Path(__file__).resolve().parent.parent)
        self.rows: list[dict] = []
        self.tokens: dict = {}
        self.order = 0
        self._events = None

    def event(self, kind: str, **data) -> None:
        self._events.write(json.dumps({"type": kind, "time": now_iso(), **data}, ensure_ascii=False) + "\n")
        self._events.flush()

    def release(self, model: str) -> tuple:
        result = release(self.bench, self.http, self.host, model, self.unload_timeout, self.sleep)
        self.event("release", model=model, confirmed=result[0], seconds=round(result[1], 3))
        return result

    # -------------------------------------------------- jedno zapytanie
    def ask(self, model: str, test_id: str, variant: str, n: int, *, cold: bool, released: Optional[tuple],
            logprobs: bool = False, raw_prompt: Optional[str] = None) -> dict:
        test = self.tests[test_id]
        _, max_tokens, temperature, seed = self.cfg[test_id]
        payload = {"model": model, "stream": False, "keep_alive": "5m",
                   "options": {"num_ctx": CONTEXT, "num_predict": max_tokens, "temperature": temperature,
                               "seed": seed}}
        if raw_prompt is None:                      # jak OllamaBench.generate
            payload.update(prompt=test["prompt"], think=False)
        else:                                       # tryb raw: szablon pomijany, tekst wprost
            payload.update(prompt=raw_prompt, raw=True)
        if logprobs:
            payload.update(logprobs=True, top_logprobs=N_PROBS)
        self.order += 1
        t0 = time.monotonic()
        error, r = "", {}
        try:
            r = self.http.post_json(f"{self.host}/api/generate", payload, timeout=900)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        text = str(r.get("response") or "")
        cached = r.get("prompt_eval_cached_count")
        if error:
            ok = False
        elif cold:                                   # od zera: zwolnienie potwierdzone, nic z pamięci
            ok = bool(released and released[0]) and not cached
        else:                                        # z pamięci: coś musiało przyjść z pamięci
            ok = cached is None or cached > 0
        v = evaluate(text, test["evaluator"]) if not error else None
        row = {"kolejnosc": self.order, "czas": now_iso(), "model": model, "zadanie": test_id, "wariant": variant,
               "proba": f"{n}/{self.repeats}", "logprobs": "tak" if logprobs else "nie",
               "raw": "tak" if raw_prompt is not None else "nie",
               "wliczona": "tak" if ok else "nie", "zaliczona": "—" if v is None else ("tak" if v.passed else "nie"),
               "powod": "" if v is None or v.passed else v.reason, "blad": error,
               "tokeny_promptu": null(r.get("prompt_eval_count")), "tokeny_z_pamieci": null(cached),
               "model_zwolniony": "—" if released is None else ("tak" if released[0] else "NIE"),
               "czas_s": round(time.monotonic() - t0, 3), "skrot_odpowiedzi": sha(text), "odpowiedz": text}
        self.rows.append(row)
        self.tokens[(model, test_id, variant, n)] = ollama_tokens(r)
        self.event("attempt", row=row, request={k: v for k, v in payload.items() if k != "prompt"},
                   prompt=payload["prompt"], response={k: v for k, v in r.items() if k != "context"})
        return row

    # -------------------------------------------------- logprobs
    def run_logprobs(self, plan) -> None:
        for model, task_ids in plan:
            for test_id in task_ids:
                self.echo(f"{model} · {test_id} · logprobs bez/z")
                for n in range(1, self.repeats + 1):
                    for lp in (False, True):                 # na zmianę – ten sam stan komputera
                        tag = "z" if lp else "bez"
                        released = self.release(model)
                        self.ask(model, test_id, f"O0_{tag}", n, cold=True, released=released, logprobs=lp)
                        self.ask(model, test_id, f"O1_{tag}", n, cold=False, released=None, logprobs=lp)

    # -------------------------------------------------- bos
    def run_bos(self, model: str, task_ids: list[str]) -> None:
        for test_id in task_ids:
            prompt = self.tests[test_id]["prompt"]
            self.echo(f"{model} · {test_id} · szablon / raw / raw+bos")
            for n in range(1, self.repeats + 1):
                for variant, raw in (("szablon", None), ("raw", CHATML.format(prompt=prompt)),
                                     ("raw+bos", "<s>" + CHATML.format(prompt=prompt))):
                    released = self.release(model)
                    self.ask(model, test_id, variant, n, cold=True, released=released, raw_prompt=raw)

    # -------------------------------------------------- całość
    def run(self, kind: str, model: Optional[str] = None) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        tags = self.http.get_json(f"{self.host}/api/tags")
        available = [m.get("name") or m.get("model") for m in (tags.get("models") or [])]
        manifest = {"narzedzie": "tools/eksperyment_ollama.py", "eksperyment": kind, "bobr": __version__,
                    "start": now_iso(), "system": platform.platform(), "python": sys.version.split()[0],
                    "ollama": self.bench.version(), "kontekst": CONTEXT, "powtorzenia": self.repeats,
                    "status": "w toku"}
        if kind == "logprobs":
            plan = [(m, t) for m, t in PAIRS if m in available]
            manifest["plan"] = [{"model": m, "zadania": t} for m, t in plan]
            manifest["warianty"] = {"O0": "model zwolniony, od zera", "O1": "od razu drugi raz (z pamięci)",
                                    "bez/z": "logprobs false / true (top_logprobs 5)"}
        else:
            model = model or MINITRON
            plan = [tid for sid in SUITES_ORDER for tid, (s, *_r) in self.cfg.items() if s == sid]
            show = self.http.post_json(f"{self.host}/api/show", {"model": model}) if model in available else {}
            manifest.update(model=model, zadania=plan, szablon_ollamy=show.get("template"),
                            szablon_ollamy_ma_bos=has_bos(show.get("template")),
                            warianty={"szablon": "zwykłe zapytanie (szablon Ollamy)",
                                      "raw": "raw, ten sam tekst co szablon Ollamy (kontrola)",
                                      "raw+bos": "raw, z <s> na początku (jak szablon GGUF)"})
        with open(self.out / "zdarzenia.jsonl", "w", encoding="utf-8") as self._events:
            try:
                self.event("start", manifest=manifest)
                if kind == "logprobs":
                    self.run_logprobs(plan)
                elif model not in available:
                    self.echo(f"Brak modelu {model} w Ollamie.")
                    manifest["status"] = "brak modelu"
                elif "<|im_start|>" not in str(manifest.get("szablon_ollamy") or ""):
                    self.echo("Szablon Ollamy tego modelu to nie ChatML – test bos obsługuje tylko ChatML.")
                    manifest["status"] = "szablon nie ChatML"
                else:
                    self.run_bos(model, plan)
                if manifest["status"] == "w toku":
                    manifest["status"] = "zakończony"
            except KeyboardInterrupt:
                manifest["status"] = "przerwany (Ctrl+C)"
                self.echo("\nPrzerwano – zapisuję to, co już zmierzone.")
            finally:
                manifest["koniec"] = now_iso()
                self.event("end", status=manifest["status"])
        summary = summarize_logprobs(self.rows, self.tokens) if kind == "logprobs" else summarize_bos(self.rows)
        write_csv(self.out / "proby.csv", self.rows)
        write_csv(self.out / "podsumowanie.csv", summary)
        manifest["wnioski"] = (conclusions_logprobs if kind == "logprobs" else conclusions_bos)(summary)
        manifest["pliki"] = ["zdarzenia.jsonl", "proby.csv", "podsumowanie.csv", "manifest.json"]
        (self.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"manifest": manifest, "summary": summary}


# ---------------------------------------------------------------------- podsumowania
def _group(rows):
    out: dict = {}
    for r in rows:
        if r["wliczona"] == "tak":
            out.setdefault((r["model"], r["zadanie"]), {}).setdefault(r["wariant"], []).append(r)
    return out


def _modal(rs):
    texts = [r["odpowiedz"] for r in rs]
    if not texts:
        return None, "—", 0
    text, n = Counter(texts).most_common(1)[0]
    return text, f"{n}/{len(texts)}", len(set(texts))


def _same(a, b):
    return "—" if a is None or b is None else ("tak" if a == b else "NIE")


def _passed(rs):
    return f"{sum(r['zaliczona'] == 'tak' for r in rs)}/{len(rs)}" if rs else "—"


def summarize_logprobs(rows: list[dict], tokens: Optional[dict] = None) -> list[dict]:
    tokens = tokens or {}
    out = []
    for (model, test_id), v in _group(rows).items():
        row = {"model": model, "zadanie": test_id}
        modal = {}
        for tag in ("bez", "z"):
            for st in ("O0", "O1"):
                key = f"{st}_{tag}"
                text, share, distinct = _modal(v.get(key, []))
                modal[key] = text
                row.update({f"{key}_najczestsza": share, f"{key}_roznych": distinct,
                            f"{key}_zaliczone": _passed(v.get(key, []))})
            row[f"O0_jak_O1_{tag}"] = _same(modal[f"O0_{tag}"], modal[f"O1_{tag}"])
        row["O0_bez_jak_O0_z"] = _same(modal["O0_bez"], modal["O0_z"])
        row["O1_bez_jak_O1_z"] = _same(modal["O1_bez"], modal["O1_z"])
        div = None
        if modal["O0_z"] != modal["O1_z"]:                    # prawdopodobieństwa są tylko przy logprobs
            def first(key):
                rs = v.get(key, [])
                return tokens.get((model, test_id, key, int(rs[0]["proba"].split("/")[0]))) if rs else None
            div = first_divergence(first("O0_z"), first("O1_z"))
        row.update({"rozbieznosc_z_pozycja": "—" if not div else div["pozycja"],
                    "rozbieznosc_O0_z_top2": top2(div["a"]) if div else "—",
                    "rozbieznosc_O1_z_top2": top2(div["b"]) if div else "—"})
        out.append(row)
    return out


def conclusions_logprobs(summary: list[dict]) -> dict:
    def n(key, value="NIE"):
        return sum(s[key] == value for s in summary)
    return {"par": len(summary),
            "pamiec_zmienia_odpowiedz_bez_logprobs": n("O0_jak_O1_bez"),
            "pamiec_zmienia_odpowiedz_z_logprobs": n("O0_jak_O1_z"),
            "logprobs_zmienia_odpowiedz_od_zera": n("O0_bez_jak_O0_z"),
            "logprobs_zmienia_odpowiedz_z_pamieci": n("O1_bez_jak_O1_z"),
            "grup_niestabilnych": sum(s[k] not in (0, 1) for s in summary for k in s if k.endswith("_roznych"))}


def summarize_bos(rows: list[dict]) -> list[dict]:
    out = []
    for (model, test_id), v in _group(rows).items():
        row = {"model": model, "zadanie": test_id}
        modal = {}
        for variant in ("szablon", "raw", "raw+bos"):
            text, share, distinct = _modal(v.get(variant, []))
            modal[variant] = text
            tok = v.get(variant, [{}])[0].get("tokeny_promptu", "null") if v.get(variant) else "null"
            row.update({f"{variant}_zaliczone": _passed(v.get(variant, [])), f"{variant}_najczestsza": share,
                        f"{variant}_roznych": distinct, f"{variant}_tokeny": tok})
        row["raw_jak_szablon"] = _same(modal["raw"], modal["szablon"])         # kontrola odtworzenia
        row["bos_zmienia_odpowiedz"] = {"tak": "nie", "NIE": "TAK"}.get(_same(modal["raw+bos"], modal["raw"]), "—")
        row["odpowiedz_bez_bos"] = modal["raw"] or ""
        row["odpowiedz_z_bos"] = modal["raw+bos"] or ""
        out.append(row)
    return out


def conclusions_bos(summary: list[dict]) -> dict:
    def total(key):
        a = b = 0
        for s in summary:
            if "/" in str(s[key]):
                x, y = s[key].split("/")
                a, b = a + int(x), b + int(y)
        return f"{a}/{b}"
    return {"zadan": len(summary),
            "kontrola_raw_jak_szablon": sum(s["raw_jak_szablon"] == "tak" for s in summary),
            "bos_zmienia_odpowiedz": sum(s["bos_zmienia_odpowiedz"] == "TAK" for s in summary),
            "zaliczone_szablon": total("szablon_zaliczone"), "zaliczone_raw": total("raw_zaliczone"),
            "zaliczone_raw_bos": total("raw+bos_zaliczone")}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eksperyment", choices=("logprobs", "bos"))
    ap.add_argument("--ollama", default=OLLAMA)
    ap.add_argument("--model", help=f"bos: model do testu (domyślnie {MINITRON})")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or Path(f"eksperyment-ollama-{args.eksperyment}-{datetime.datetime.now():%Y%m%d-%H%M}")
    exp = Experiment(out, args.ollama, args.repeats)
    try:
        tags = exp.http.get_json(f"{exp.host}/api/tags")
        print(f"Ollama {exp.bench.version()} pod {exp.host}")
    except Exception as e:
        print(f"Nie mogę połączyć się z Ollamą pod {exp.host}: {e}")
        return 1
    available = [m.get("name") or m.get("model") for m in (tags.get("models") or [])]
    if args.eksperyment == "logprobs":
        plan = [(m, t) for m, t in PAIRS if m in available]
        pairs = sum(len(t) for _, t in plan)
        print(f"Plan: {pairs} par × {args.repeats} × (bez, z logprobs) × (O0, O1) = {pairs * args.repeats * 4} zapytań"
              f" + {pairs * args.repeats * 2} zwolnień modelu.")
        for m, t in PAIRS:
            print(f"  {'✓' if m in available else '✗ brak'}  {m}: {', '.join(t)}")
    else:
        model = args.model or MINITRON
        print(f"Plan: {model}, 15 zadań × {args.repeats} × (szablon, raw, raw+bos) = {15 * args.repeats * 3} zapytań,"
              " każde po zwolnieniu modelu.")
        print(f"  {'✓' if model in available else '✗ brak'}  {model}")
    if args.dry_run:
        return 0
    result = exp.run(args.eksperyment, args.model)
    print("\nWnioski:")
    for k, v in result["manifest"]["wnioski"].items():
        print(f"  {k}: {v}")
    print(f"\nStatus: {result['manifest']['status']}. Wyniki: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
