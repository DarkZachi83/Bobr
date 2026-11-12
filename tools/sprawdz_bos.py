"""Token początku (BOS) w Ollamie z terminala – diagnostyka 0.12, część 1 (bez okna).

    python tools/sprawdz_bos.py                       # wszystkie modele z Ollamy
    python tools/sprawdz_bos.py hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M gemma3:4b

Każdy model: opis z pliku GGUF, szablon Ollamy i dwa krótkie zapytania w trybie raw, każde po zwolnieniu
modelu (dwa wczytania). Niczego nie zmienia w Ollamie.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aibobr.bench.ollama import OllamaBench                     # noqa: E402
from aibobr.diagnostics.bos import BosProbe                     # noqa: E402

LABEL = {"ok": "OK – jeden token początku", "missing": "BRAK – model oczekuje tokenu początku, nie dostaje go",
         "double": "PODWÓJNY – model dostaje dwa tokeny początku", "not_used": "OK – model nie używa tokenu początku",
         "unknown": "nie wiadomo"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Czy model w Ollamie dostaje token początku (BOS).")
    p.add_argument("models", nargs="*", help="nazwy modeli; bez nazw – wszystkie z /api/tags")
    p.add_argument("--ollama", default="http://localhost:11434")
    args = p.parse_args(argv)
    bench = OllamaBench(args.ollama)
    models = args.models or [m.get("name") for m in (bench.http.get_json(f"{bench.host}/api/tags") or {})
                             .get("models", []) if m.get("name")]
    probe = BosProbe(bench, clock=time.monotonic, sleep=time.sleep)
    for model in models:
        print(f"\n== {model}")
        c = probe.run(model)
        print(f"   wynik: {LABEL.get(c.status, c.status)}  ({c.reason})")
        source = {"model_info": "z Ollamy", "file": "z pliku modelu", None: "niedostępny"}[c.gguf_template_source]
        print(f"   model oczekuje: {c.expects}   (flaga add_bos_token: {c.add_bos}, szablon GGUF z bos_token: "
              f"{c.gguf_template_has} – {source})")
        print(f"   token: {c.bos_text!r}   w szablonie Ollamy: {c.template_has}")
        print(f"   Ollama dodaje sama: {c.without}   z tokenem w tekście: {c.with_text}   "
              f"zwykłe zapytanie: {c.effective}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
