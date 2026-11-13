"""Kruchość promptu z terminala – diagnostyka 0.12, część 2 (bez okna).

    python tools/kruchosc.py gemma3:4b                                  # test ścisły po polsku
    python tools/kruchosc.py gemma3:4b --test humanistyczny_v1/pl --tasks H1 H5
    python tools/kruchosc.py hf.co/…Minitron…:Q4_K_M --test scisly_v1/en --fragile 0.15 --tie 0.03
    python tools/kruchosc.py gemma3:4b --test moj_test/ko                # własny test z edytora
    python tools/kruchosc.py hf.co/…Minitron…:Q4_K_M --prompt "Przywróć polskie znaki: Zazolc gesla jazn"

Każde zadanie: model zwolniony, odpowiedź z logprobs (dwóch najlepszych kandydatów przy każdym tokenie), te same
ustawienia co test jakości. Wynik: stabilne / kruche / remis, najwęższe miejsce i pierwszy prawie remis.
Etykieta liczy kroki w odpowiedzi i przed nią (miejsce odpowiedzi wskazuje kod oceny testu; w kodzie bez
docstringów i komentarzy); prawie remisy po odpowiedzi – osobno. Przy --prompt – cała wypowiedź.
Osobny przebieg – nie wpływa na wyniki benchmarku ani testu jakości. Niczego nie zmienia w Ollamie.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aibobr.bench.ollama import OllamaBench                                    # noqa: E402
from aibobr.diagnostics.fragility import FRAGILE, TIE, FragilityProbe, Thresholds, suite_tasks  # noqa: E402
from aibobr.quality import suites                                              # noqa: E402
from aibobr.quality.custom import CustomSuites                                 # noqa: E402

LABEL = {"stable": "stabilne", "fragile": "KRUCHE", "tie": "REMIS", "unsupported": "Ollama nie podaje logprobs",
         "not_run": "nie wykonano"}


def tok(t) -> str:
    return "‹koniec›" if t == "" else repr(t)          # pusty token – koniec odpowiedzi


def pair(p) -> str:
    if not p:
        return "—"
    (t1, p1), second = p
    return f"{tok(t1)} {p1:.3f}" + (f" | {tok(second[0])} {second[1]:.3f}" if second else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kruchość promptu: odstęp między pierwszym a drugim kandydatem.")
    ap.add_argument("model")
    ap.add_argument("--test", default="scisly_v1/pl", help="test/język, np. humanistyczny_v1/pl, moj_test/ko")
    ap.add_argument("--tasks", nargs="*", help="tylko te zadania, np. S1 S9")
    ap.add_argument("--prompt", help="dowolne polecenie zamiast testu (ocena całej wypowiedzi)")
    ap.add_argument("--max-tokens", type=int, default=256, help="limit tokenów dla --prompt")
    ap.add_argument("--steps", action="store_true", help="każdy krok odpowiedzi: token, dwóch kandydatów, odstęp")
    ap.add_argument("--fragile", type=float, default=FRAGILE, help=f"próg „kruche” (domyślnie {FRAGILE})")
    ap.add_argument("--tie", type=float, default=TIE, help=f"próg „remis” (domyślnie {TIE})")
    ap.add_argument("--ollama", default="http://localhost:11434")
    args = ap.parse_args(argv)
    if args.prompt:
        suite_id, lang, tasks = "--prompt", "-", [("P", args.prompt, args.max_tokens)]
    else:
        suite_id, _, lang = args.test.partition("/")
        data = suites.load(suite_id) if suite_id in suites.STANDARD else CustomSuites().load(suite_id)
        lang = lang or data["suite_info"]["languages"][0]
        tasks = [t for t in suite_tasks(data, lang) if not args.tasks or t[0] in args.tasks]
    thresholds = Thresholds(args.fragile, args.tie)
    probe = FragilityProbe(OllamaBench(args.ollama), clock=time.monotonic, sleep=time.sleep)
    print(f"== {args.model} – {suite_id} ({lang}), {len(tasks)} zadań, progi: kruche < {thresholds.fragile}, "
          f"remis < {thresholds.tie}")

    def report(e):
        if e["event"] != "f_task_done":
            return
        t = e["task"]
        scope = (f"  [odpowiedź: {t['answer_steps']} z {t['steps']} kroków]" if t["scope"] == "answer"
                 else "  [cała wypowiedź]" if t["steps"] else "")
        print(f"\n{t['task_id']:>4}  {LABEL.get(t['label'], t['label'])}{scope}"
              + (f"  ({t['reason']})" if t["reason"] else ""))
        if t["min_margin"] is not None:
            print(f"      najwęższe miejsce: krok {t['min_pos']}/{t['steps']}, odstęp {t['min_margin']:.3f}: "
                  f"{pair(t['min_pair'])}")
        if t["first_pos"] is not None:
            print(f"      pierwszy prawie remis: krok {t['first_pos']} po „…{t['first_context']}”: "
                  f"{pair(t['first_pair'])}  (kroków kruchych: {t['fragile_steps']}, remisów: {t['tie_steps']})")
        if t["after_fragile"]:
            print(f"      po odpowiedzi (bez wpływu na etykietę): kruchych {t['after_fragile']}, remisów {t['after_tie']}")
        if args.steps:
            for n, (token, p1, token2, p2) in enumerate(t.get("step_data") or []):
                margin = p1 - (p2 or 0.0)
                mark = " REMIS" if margin < thresholds.tie else " kruche" if margin < thresholds.fragile else ""
                second = f"{tok(token2)} {p2:.3f}" if token2 is not None else "–"
                print(f"      krok {n:>3}: {tok(token):<14} {p1:.3f} | {second:<20} odstęp {margin:.3f}{mark}")
        answer = " ".join(t["answer"].split())
        print(f"      odpowiedź: {answer[:100]}{'…' if len(answer) > 100 else ''}")
    run = probe.run(args.model, tasks, thresholds, report=report)
    c = run.counts()
    print(f"\nRazem: stabilne {c['stable']}, kruche {c['fragile']}, remis {c['tie']}"
          + (f", nie wykonano {c['not_run']}" if c["not_run"] else "")
          + (" – ta Ollama nie podaje logprobs (potrzebna nowsza wersja)" if c["unsupported"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
