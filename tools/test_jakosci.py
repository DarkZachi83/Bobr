"""Testy jakości 0.10 z terminala – sprawdzenie na prawdziwej Ollamie, zanim powstanie zakładka.

Zapisuje do OSOBNEJ bazy w folderze wyników (domyślnie test-jakosci-RRRR-MM-DD_GGMM/), nie do
historii programu. Kod Pythona tylko w zweryfikowanej piaskownicy (bubblewrap); bez niej –
zadania z kodem nieocenione.

  python tools/test_jakosci.py gemma2:9b
  python tools/test_jakosci.py gemma2:9b --attempts 3 --tests humanistyczny_v1/pl scisly_v1/pl
  python tools/test_jakosci.py gemma2:9b --tests moj_test/ko moj_test/en   # własny test z edytora
"""
from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aibobr import __version__  # noqa: E402
from aibobr.app.bench_db import BenchDb  # noqa: E402
from aibobr.bench.ollama import OllamaBench  # noqa: E402
from aibobr.quality.custom import CustomSuites  # noqa: E402
from aibobr.quality import score, suites  # noqa: E402
from aibobr.quality.runner import QualityRunner  # noqa: E402
from aibobr.quality.sandbox import Sandbox  # noqa: E402

ALL = ["humanistyczny_v1/pl", "scisly_v1/pl", "scisly_v1/en"]
MARK = {True: "✓", False: "✗", None: "–"}
LABEL = {"text": "w tych testach lepiej wypadł w zadaniach tekstowych",
         "science": "w tych testach lepiej wypadł w zadaniach ścisłych",
         "similar": "w tych testach wypadł podobnie w obu", "weak": "w tych testach wypadł słabo w obu",
         "none": "brak porównania"}


def frac(t: score.Tally) -> str:
    pct = f" ({t.percent:.0f}%)" if t.percent is not None else ""
    cover = f" · pokrycie {t.evaluated}/{t.total} – wynik częściowy" if t.partial else ""
    return f"{t.correct}/{t.evaluated} ocenionych{pct}{cover}"


def printer(e: dict) -> None:
    kind = e["event"]
    if kind == "q_start":
        print(f"Ollama {e['version']} · modele: {', '.join(e['models'])} · próby: {e['attempts']} · "
              f"testy: {', '.join(e['tests'])} · piaskownica: {e['sandbox']}")
    elif kind == "q_run":
        print(f"\n== {e['model']} – {e['name'].get('pl') or next(iter(e['name'].values()))} ({e['lang']})")
    elif kind == "q_task_done":
        t = e["result"]
        fmt = " · niezgodny format" if t.correct and t.format_compliant is False else ""
        diff = " · odpowiedzi różne" if t.differing else ""
        print(f"   {t.task_id:>4} {t.level:<6} {t.category:<16} {MARK[t.correct]} {t.reason:<9}{fmt}{diff} "
              f"| {t.answer.strip()[:60]!r}")
    elif kind == "q_run_done":
        s = e["result"]["summary"]
        print(f"   WYNIK: {frac(s.overall)}")
        print("   kategorie: " + " · ".join(f"{k} {v.correct}/{v.evaluated}" for k, v in s.by_category.items()))
        print("   poziomy:   " + " · ".join(f"{k} {v.correct}/{v.evaluated}" for k, v in s.by_level.items()))
        if s.format_noncompliant:
            print(f"   niezgodny format przy {s.format_noncompliant} zaliczonych zadaniach")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--attempts", type=int, choices=(1, 3), default=1)
    p.add_argument("--tests", nargs="*", default=ALL)
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--out", type=Path, default=Path(f"test-jakosci-{datetime.now():%Y-%m-%d_%H%M}"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    loaded, tests = {}, []
    for item in a.tests:
        suite_id, lang = item.split("/")
        if suite_id not in loaded:      # własny test – z folderu quality/ w katalogu ustawień (edytor 0.11)
            loaded[suite_id] = (suites.load(suite_id) if suite_id in suites.STANDARD
                                else CustomSuites().load(suite_id))
        tests.append((loaded[suite_id], lang))
    bench = OllamaBench(a.host)
    runner = QualityRunner(bench, Sandbox(), BenchDb(a.out / "benchmarks.sqlite"),
                           {"app_version": __version__, "runtime_version": bench.version()})
    stop = threading.Event()
    try:
        out = runner.run(a.models, tests, a.attempts, printer, stop)
    except KeyboardInterrupt:
        stop.set()
        return 1
    for model in a.models:
        runs = {(r["suite"], r["lang"]): r["summary"].overall for r in out["runs"] if r["model"] == model}
        lab = score.label(runs.get(("humanistyczny_v1", "pl")), runs.get(("scisly_v1", "pl")))
        print(f"\n{model}: {LABEL[lab.kind]}" + (" (wynik częściowy)" if lab.partial and lab.kind != "none" else ""))
        if ("scisly_v1", "pl") in runs and ("scisly_v1", "en") in runs:
            pl, en = runs[("scisly_v1", "pl")], runs[("scisly_v1", "en")]
            print(f"  ścisły: PL {pl.correct}/{pl.evaluated} · EN {en.correct}/{en.evaluated}")
    print(f"\nBaza: {a.out / 'benchmarks.sqlite'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
