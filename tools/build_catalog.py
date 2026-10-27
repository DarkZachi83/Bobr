"""Buduje data/models/catalog.json z listy repozytoriów (SPEC §9) – narzędzie autora.

    python tools/build_catalog.py                      # lista z data/models/catalog_sources.txt
    python tools/build_catalog.py --token hf_…         # gdyby któreś repo wymagało tokenu

Dla każdego repozytorium pobiera (jak program) listę plików z rozmiarami i nagłówek
jednego pliku GGUF – kilka MB, nie gigabajty. Repozytoria, których nie da się wczytać,
są wypisywane i pomijane; reszta trafia do katalogu posortowana od najmniejszego.
Po zbudowaniu: sprawdź wynik (git diff), zatwierdź i wypchnij – przycisk
„Sprawdź aktualizacje” w programie pobiera plik z gałęzi main.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aibobr.knowledge.catalog import FORMAT  # noqa: E402
from aibobr.sources.base import SourceError  # noqa: E402
from aibobr.sources.http import HttpClient  # noqa: E402
from aibobr.sources.huggingface import HfRef, HuggingFaceSource  # noqa: E402

SOURCES = ROOT / "data" / "models" / "catalog_sources.txt"
OUTPUT = ROOT / "data" / "models" / "catalog.json"


def read_sources(path: Path) -> list[str]:
    repos = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            repos.append(line)
    return repos


def entry(repo: str, source: HuggingFaceSource) -> dict:
    model = source.load(HfRef(repo))
    return {
        "name": model.name,
        "repo": repo,
        "arch": {k: v for k, v in asdict(model.arch).items() if v is not None},
        "variants": {v.quant: v.size for v in model.variants if v.size},
    }


def build(repos: list[str], source: HuggingFaceSource, today: str, log=print) -> tuple[dict, list]:
    models, failed = [], []
    for repo in repos:
        t0 = time.time()
        try:
            e = entry(repo, source)
        except SourceError as err:
            failed.append((repo, err.code))
            log(f"  ✗ {repo}: {err.code}")
            continue
        if not e["variants"]:
            failed.append((repo, "no_sizes"))
            log(f"  ✗ {repo}: brak plików GGUF z rozmiarami")
            continue
        models.append(e)
        params = e["arch"].get("params_total")
        log(f"  ✓ {repo}: {e['name']}, {params / 1e9:.1f}B, {len(e['variants'])} wariantów"
            f" ({time.time() - t0:.1f} s)" if params else f"  ✓ {repo}: {e['name']}")
    models.sort(key=lambda m: m["arch"].get("params_total") or 0)
    return {"format": FORMAT, "updated": today,
            "source": "tools/build_catalog.py ← data/models/catalog_sources.txt",
            "models": models}, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--token")
    ap.add_argument("--sources", type=Path, default=SOURCES)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    repos = read_sources(args.sources)
    print(f"Repozytoriów: {len(repos)}")
    catalog, failed = build(repos, HuggingFaceSource(HttpClient(token=args.token, timeout=60)),
                            datetime.date.today().isoformat())
    if not catalog["models"]:
        print("Żadne repozytorium się nie wczytało – catalog.json bez zmian.")
        return 1
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nZapisano {len(catalog['models'])} modeli do {args.output}"
          + (f"; pominięte: {len(failed)}" if failed else ""))
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
