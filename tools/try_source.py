"""Narzędzie deweloperskie: wczytaj model ze źródła i pokaż, co program z niego odczytał.

    python tools/try_source.py <link HF | hf.co/…:QUANT | ścieżka .gguf> [--token hf_…]

Nie jest częścią programu dla użytkownika (ten dostanie tryb --cli w etapie 4).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aibobr.calculator.memory import weights_bytes  # noqa: E402
from aibobr.calculator.quants import QuantTable  # noqa: E402
from aibobr.domain.units import to_gib  # noqa: E402
from aibobr.sources.base import SourceError  # noqa: E402
from aibobr.sources.factory import load_model  # noqa: E402
from aibobr.sources.http import HttpClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--token", help="token Hugging Face (modele gated)")
    args = ap.parse_args()
    t = time.time()
    try:
        m = load_model(args.model, HttpClient(token=args.token))
    except SourceError as e:
        print(f"BŁĄD {e.code}: {e.detail}")
        return 1
    a = m.arch
    print(f"{m.name}  [{m.source.value}]  ({time.time() - t:.1f} s)")
    print(f"  architektura: {a.architecture}  parametry: {a.params_total and a.params_total / 1e9:.2f}B"
          if a.params_total else f"  architektura: {a.architecture}  parametry: ?")
    print(f"  warstwy {a.n_layers} · głowice KV {a.n_kv_heads} · wymiar głowicy {a.head_dim}"
          f" · kontekst maks. {a.max_context}")
    if a.n_experts:
        act = f"{a.params_active / 1e9:.2f}B" if a.params_active else "?"
        print(f"  MoE: {a.n_experts} ekspertów, {a.n_experts_used} aktywnych, aktywne parametry {act}")
    quants = QuantTable.load(Path(__file__).resolve().parent.parent / "data" / "quants.json")
    print("  warianty (GB = 1024³ B, jak w programie):")
    default = m.default_variant().quant
    for v in m.variants:
        if v.size:
            size = f"{to_gib(v.size):6.2f} GB"
        else:
            try:
                size = f"~{to_gib(weights_bytes(m.spec(v.quant), quants)[0]):5.2f} GB (szacunek)"
            except Exception:
                size = "     ? GB"
        parts = f"  [{len(v.files)} części]" if len(v.files) > 1 else ""
        mark = "  ← domyślny" if v.quant == default else ""
        print(f"    {v.quant:10} {size}{parts}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
