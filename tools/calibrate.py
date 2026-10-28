"""Kalibracja (SPEC §15, etap 8): przewidywania AI Bobra kontra prawdziwe zużycie pamięci.

    python tools/calibrate.py                          # wszystkie modele z `ollama list`
    python tools/calibrate.py llama3.1:8b gemma3:4b    # wybrane
    python tools/calibrate.py --contexts 2048,8192,16384

Dla każdego modelu i kontekstu: zwalnia pamięć, ładuje model do Ollamy (pusty prompt –
nic nie jest generowane), odczytuje z Ollamy zajętą pamięć (łącznie i w VRAM) oraz z
nvidia-smi faktyczny przyrost VRAM, i porównuje z tym, co przewiduje program dla karty
(wagi bez tablicy słownika, która zostaje w RAM, + pamięć rozmowy, bez narzutu).
Różnica to właśnie narzut, który kalibrujemy.

Wynik: tabela na ekranie i plik calibration-RRRRMMDD-GGMM.json – wyślij go autorom.
Nic nie jest pobierane z internetu; modele muszą być już w Ollamie.
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aibobr import __version__  # noqa: E402
from aibobr.calculator.memory import memory_need  # noqa: E402
from aibobr.calculator.quants import QuantTable  # noqa: E402
from aibobr.detection.base import CommandRunner  # noqa: E402
from aibobr.domain.models import RunSettings  # noqa: E402
from aibobr.sources.base import SourceError  # noqa: E402
from aibobr.sources.http import HttpClient  # noqa: E402
from aibobr.sources.ollama import DEFAULT_HOST, OllamaSource  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIB = 1024**2
GIB = 1024**3
DEFAULT_CONTEXTS = (2048, 8192, 16384)


def nvidia_used_mib(run: Callable = CommandRunner()) -> Optional[int]:
    """Zajęty VRAM wszystkich kart NVIDIA (MiB) albo None, gdy nie ma nvidia-smi."""
    out = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if not out:
        return None
    try:
        return sum(int(float(x)) for x in out.split())
    except ValueError:
        return None


@dataclass
class Measurement:
    model: str
    quant: str
    architecture: Optional[str]
    context: int
    file_bytes: Optional[int]
    predicted_weights: int
    predicted_kv: int
    kv_sliding: bool
    is_moe: bool
    embd_in_ram: int = 0                      # tablica słownika – zostaje w RAM, nie w karcie
    ollama_size: Optional[int] = None         # /api/ps: cała pamięć modelu
    ollama_vram: Optional[int] = None         # /api/ps: część w VRAM
    ollama_context: Optional[int] = None      # /api/ps: kontekst, który Ollama naprawdę dała
    nvidia_delta: Optional[int] = None        # przyrost zajętego VRAM wg nvidia-smi
    error: Optional[str] = None

    @property
    def predicted(self) -> int:
        """Przewidywanie dla karty (to widzi nvidia-smi)."""
        return self.predicted_weights + self.predicted_kv - self.embd_in_ram

    @property
    def fully_in_vram(self) -> bool:
        return bool(self.ollama_size and self.ollama_vram and self.ollama_vram >= self.ollama_size)

    @property
    def measured(self) -> Optional[int]:
        """Najlepszy pomiar: nvidia-smi, gdy całość w VRAM; inaczej wycena Ollamy."""
        if self.fully_in_vram and self.nvidia_delta:
            return self.nvidia_delta
        return self.ollama_size

    @property
    def overhead(self) -> Optional[int]:
        return None if self.measured is None else self.measured - self.predicted


class Calibrator:
    def __init__(self, ollama: OllamaSource, quants: QuantTable,
                 nvidia: Callable[[], Optional[int]] = nvidia_used_mib,
                 sleep: Callable[[float], None] = time.sleep, log=print):
        self.ollama, self.quants, self.nvidia = ollama, quants, nvidia
        self.sleep, self.log = sleep, log

    def measure(self, name: str, ctx: int) -> Measurement:
        model = self.ollama.load(name)
        spec = model.spec()
        need = memory_need(spec, RunSettings(context_tokens=ctx, overhead=0), self.quants)
        m = Measurement(model=name, quant=spec.quant, architecture=model.arch.architecture,
                        context=ctx, file_bytes=spec.weights_bytes,
                        predicted_weights=need.weights_total, predicted_kv=need.kv_cache,
                        kv_sliding=need.kv_sliding, is_moe=spec.is_moe,
                        embd_in_ram=need.embd_in_ram)
        try:
            self.ollama.unload_all()
            self.sleep(3)                                     # sterownik oddaje pamięć z opóźnieniem
            before = self.nvidia()
            self.ollama.load_into_memory(name, ctx)
            self.sleep(2)
            after = self.nvidia()
            entry = next((r for r in self.ollama.running()
                          if r.get("name") == name or r.get("model") == name), None)
            if entry:
                m.ollama_size = int(entry.get("size") or 0) or None
                m.ollama_vram = int(entry.get("size_vram") or 0)
                m.ollama_context = entry.get("context_length") or None
            if before is not None and after is not None:
                m.nvidia_delta = max(0, after - before) * MIB
        except SourceError as e:
            m.error = f"{e.code}: {e.detail}"
        finally:
            try:
                self.ollama.unload(name)
            except SourceError:
                pass
        return m

    def run(self, names: list[str], contexts: list[int]) -> list[Measurement]:
        results = []
        for name in names:
            try:
                limit = self.ollama.load(name).arch.max_context
            except SourceError as e:
                hint = f" – pobierz: ollama pull {name}" if e.code == "ollama.not_found" else ""
                self.log(f"  ✗ {name}: {'nie ma w Ollamie' if hint else e.code}{hint}")
                continue
            for ctx in contexts:
                if limit and ctx > limit:
                    continue
                m = self.measure(name, ctx)
                results.append(m)
                self.log(row(m))
        return results


def gb(value: Optional[int]) -> str:
    return "     –" if value is None else f"{value / GIB:6.2f}"


HEADER = (f"  {'model':28} {'kontekst':>8} {'przewid.':>8} {'Ollama':>7} {'w VRAM':>7}"
          f" {'nvidia':>7} {'narzut':>7}  uwagi")


def row(m: Measurement) -> str:
    notes = []
    if m.kv_sliding:
        notes.append("okno przesuwne")
    if m.is_moe:
        notes.append("MoE")
    if m.ollama_size and not m.fully_in_vram:
        notes.append("część w RAM")
    if m.ollama_context and m.ollama_context != m.context:
        notes.append(f"Ollama dała kontekst {m.ollama_context}")
    if m.error:
        notes.append(m.error)
    return (f"  {m.model[:28]:28} {m.context:>8} {gb(m.predicted)} {gb(m.ollama_size)}"
            f" {gb(m.ollama_vram)} {gb(m.nvidia_delta)} {gb(m.overhead)}  {', '.join(notes)}")


def summary(results: list[Measurement]) -> dict:
    full = [m.overhead for m in results if m.overhead is not None and m.fully_in_vram]
    every = [m.overhead for m in results if m.overhead is not None]
    pick = full or every
    if not pick:
        return {}
    return {"measurements": len(pick), "fully_in_vram": bool(full),
            "overhead_median_gb": round(statistics.median(pick) / GIB, 3),
            "overhead_min_gb": round(min(pick) / GIB, 3),
            "overhead_max_gb": round(max(pick) / GIB, 3)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*", help="nazwy z `ollama list` (domyślnie wszystkie)")
    ap.add_argument("--contexts", default=",".join(map(str, DEFAULT_CONTEXTS)))
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]

    ollama = OllamaSource(HttpClient(timeout=30), args.host)
    try:
        version = ollama.version()
        names = args.models or [m["name"] for m in ollama.installed()
                                if "embed" not in m.get("name", "")]
    except SourceError as e:
        print(f"Ollama nie odpowiada pod {args.host} ({e.code}). Uruchom ją i spróbuj ponownie.")
        return 1
    if not names:
        print("Ollama nie ma żadnego modelu. Pobierz np.: ollama pull llama3.2:3b")
        return 1
    print(f"Ollama {version} · modeli: {len(names)} · konteksty: {contexts}")
    if nvidia_used_mib() is None:
        print("Brak nvidia-smi – mierzę tylko według Ollamy.")
    print(HEADER)

    quants = QuantTable.load(ROOT / "data" / "quants.json")
    results = Calibrator(ollama, quants).run(names, contexts)
    report = {"aibobr": __version__, "ollama": version, "date": datetime.datetime.now().isoformat(
        timespec="minutes"), "system": platform.platform(), "summary": summary(results),
        "results": [{**asdict(m), "overhead": m.overhead, "fully_in_vram": m.fully_in_vram}
                    for m in results]}
    out = args.output or Path(f"calibration-{datetime.datetime.now():%Y%m%d-%H%M}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    s = report["summary"]
    if s:
        print(f"\nNarzut (pomiar − przewidywanie): mediana {s['overhead_median_gb']} GB, "
              f"od {s['overhead_min_gb']} do {s['overhead_max_gb']} GB "
              f"({s['measurements']} pomiarów{', całość w VRAM' if s['fully_in_vram'] else ''}).")
    print(f"Zapisano {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
