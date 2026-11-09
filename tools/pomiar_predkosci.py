"""Benchmark prędkości 0.9 z terminala – sprawdzenie na prawdziwej Ollamie, zanim powstanie okno.

Zapisuje do OSOBNEJ bazy w folderze wyników (domyślnie pomiar-predkosci-RRRR-MM-DD_GGMM/), nie do
historii programu. Stabilizacja temperatury bez okna: po przekroczeniu czasu test rusza sam,
a wynik dostaje ostrzeżenie (decyzja autora 3a).

  python tools/pomiar_predkosci.py gemma2:9b
  python tools/pomiar_predkosci.py gemma2:9b qwen2.5:7b --runs 3 --contexts 1024 8192
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aibobr import __version__  # noqa: E402
from aibobr.app.bench_db import BenchDb  # noqa: E402
from aibobr.bench import speed  # noqa: E402
from aibobr.bench.ollama import OllamaBench  # noqa: E402
from aibobr.bench.session import SpeedBenchmark  # noqa: E402

LABEL = {"fast": "🟢 Szybko", "useful": "🔵 Użytecznie", "slow": "🟡 Wolno", "too_slow": "🔴 Zbyt wolno", None: "—"}


def num(value, digits=1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def printer(event: dict) -> None:
    kind = event["event"]
    if kind == "s_start":
        print(f"Ollama {event['version']} · modele: {', '.join(event['models'])} · przebiegi: {event['runs']} · "
              f"zadania: {event['tasks']} · długi kontekst: {event['contexts'] or '—'}")
    elif kind == "s_release":
        print(f"\n== {event['model']} – przebieg {event['run']}: zwolniony={event['released']}, "
              f"VRAM wrócił={event['recovered']}")
    elif kind == "s_cooling" and event["key"] == "bench.cooling.start":
        print(f"   stabilizacja temperatury (≤ {event['tolerance']} °C w {event['window']} s)…")
    elif kind == "s_cooled":
        c = event["cooling"]
        print(f"   stabilizacja: {c.status} po {c.seconds} s (CPU {c.cpu_temp_c} °C, GPU {c.gpu_temp_c} °C)")
    elif kind == "s_baseline":
        m = event["baseline"]["mean"]
        print(f"   linia bazowa: CPU {m['cpu_pct']}%, GPU {m['gpu_util_pct']}%, VRAM {m['vram_used_mb']} MB")
    elif kind == "s_task_done":
        r = event["result"]
        mark = r.error or ", ".join(r.flags) or "ok"
        tps = r.gen_tokens / (r.eval_ms / 1000) if r.counted_gen else None
        print(f"   {r.task_no:2}/{event['total']} {r.task_id:26} {num(tps)} tok/s  {r.gen_tokens or 0:4} tok  "
              f"TTFT {num(r.ttft_wall_ms, 0)} ms  [{mark}]")
    elif kind == "s_long_done":
        lc = event["result"]
        print(f"   kontekst {lc['num_ctx']}: wejście {lc.get('prompt_tokens')} tok, wygenerowane {lc.get('gen_tokens')}, "
              f"prompt {num(lc.get('pp_tps'))} tok/s, generowanie {num(lc.get('tg_tps'))} tok/s, "
              f"ważny={lc['valid']} {lc.get('invalid_reason') or ''}")
    elif kind == "s_run_done":
        r = event["result"]
        b = speed.basis(r)
        print(f"   WERDYKT: {LABEL[r['verdict']]} (przesądziło: {r['verdict_measure']})")
        if b:
            print(f"   generowanie {num(b['tps'])} tok/s z {b['tokens']} tokenów w {num(b['seconds'])} s · "
                  f"TTFT mediana {num(r['ttft_wall_median_ms'], 0)} ms, najgorszy {num(r['ttft_wall_max_ms'], 0)} ms")
        print(f"   prompt {num(r['pp_tps'])} tok/s · wczytanie {num(r['load_wall_ms'], 0)} ms · "
              f"narzut {num((r['overhead_ratio'] or 0) * 100)}% · błędy {r['tasks_error']} · "
              f"ostrzeżenia: {', '.join(r['warnings_json']) or '—'}")
    elif kind == "stopped":
        print("\nZatrzymano.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--contexts", type=int, nargs="*", default=[1024, 8192])
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--out", type=Path, default=Path(f"pomiar-predkosci-{datetime.now():%Y-%m-%d_%H%M}"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    db = BenchDb(a.out / "benchmarks.sqlite")
    root = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
    pack = speed.build_pack([json.loads((root / f"{s}.json").read_text(encoding="utf-8"))
                             for s in ("pl_jezyk_v1", "en_format_v1")])
    bench = OllamaBench(a.host)
    info = {"app_version": __version__, "runtime_version": bench.version()}
    stop = threading.Event()
    try:
        out = SpeedBenchmark(bench, pack, db=db, info=info).run(a.models, a.runs, a.contexts, printer, stop)
    except KeyboardInterrupt:
        stop.set()
        return 1
    runs = out["runs"]
    for model in a.models:
        rng = speed.spread([r for r in runs if r["model"] == model])
        if rng and a.runs > 1:
            print(f"\n{model}: {a.runs} przebiegi – generowanie {num(rng[0])}–{num(rng[1])} tok/s "
                  "(pomiar rozrzutu, nie dowód powtarzalności)")
    print(f"\nPojedynczy przebieg: próbka wydajności w tych warunkach, nie dowód stabilności.\nBaza: {a.out / 'benchmarks.sqlite'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
