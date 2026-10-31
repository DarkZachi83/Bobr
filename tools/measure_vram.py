"""Pomiar VRAM dowolnego programu (etap 9): szczyt zużycia każdego procesu osobno.

    python tools/measure_vram.py
    python tools/measure_vram.py --model orpheus-3b-0.1-ft-q4_k_m.gguf --context 8192

1. Zapisuje, ile karty zajmują procesy przed startem.
2. Prosi: uruchom model (np. llama-server + skrypt z dekoderem), wygeneruj nagranie, naciśnij Enter.
3. W tym czasie co pół sekundy pyta nvidia-smi o pamięć KAŻDEGO procesu – osobno widać
   serwer z modelem językowym i proces dekodera dźwięku.
4. Pokazuje szczyty, przebieg w czasie z etapami („płaskowyżami”, np. UNet → VAE)
   i zapisuje measure-RRRRMMDD-GGMM.json z całym przebiegiem (do testu regresyjnego).

Z --model dopisuje przewidywanie AI Bóbra dla tej samej długości kontekstu (bez narzutu –
różnica to narzut runtime'u). Działa z każdym programem: llama.cpp, LM Studio, Python,
ComfyUI… – także dla modeli obrazów w etapie 10. Tylko karty NVIDIA (nvidia-smi).
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aibobr.detection.base import CommandRunner  # noqa: E402

MIB = 1024**2
GIB = 1024**3
QUERY_APPS = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
              "--format=csv,noheader,nounits"]
QUERY_TOTAL = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]


def parse_apps(text: Optional[str]) -> dict[int, tuple[str, Optional[int]]]:
    """'1234, /usr/bin/llama-server, 2950' → {1234: ('/usr/bin/llama-server', 2950)}.

    Na części laptopów i w WSL sterownik podaje „[N/A]” zamiast pamięci – wtedy None."""
    apps = {}
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        mem = parts[-1]
        apps[int(parts[0])] = (", ".join(parts[1:-1]), int(mem) if mem.isdigit() else None)
    return apps


def parse_total(text: Optional[str]) -> Optional[int]:
    try:
        return sum(int(float(x)) for x in (text or "").split()) if text else None
    except ValueError:
        return None


def command_line(pid: int) -> Optional[str]:
    """Pełne polecenie procesu (Linux) – „python gguf_orpheus.py” mówi więcej niż „python”."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode(errors="replace").strip() or None


@dataclass
class Process:
    pid: int
    name: str
    command: Optional[str] = None
    peak_mib: Optional[int] = None
    before_mib: Optional[int] = None         # już działał przed startem pomiaru


@dataclass
class Result:
    label: str
    seconds: float
    total_before_mib: Optional[int]
    total_peak_mib: Optional[int]
    processes: list[Process] = field(default_factory=list)
    prediction: Optional[dict] = None

    @property
    def total_delta_mib(self) -> Optional[int]:
        if self.total_before_mib is None or self.total_peak_mib is None:
            return None
        return self.total_peak_mib - self.total_before_mib

    def new_processes(self) -> list[Process]:
        """Procesy, które zaczęły używać karty w trakcie pomiaru, od największego."""
        new = [p for p in self.processes if p.before_mib is None and p.peak_mib]
        return sorted(new, key=lambda p: p.peak_mib or 0, reverse=True)

    def grown_processes(self, min_mib: int = 50) -> list[Process]:
        """Procesy działające już przed pomiarem, które w trakcie zajęły wyraźnie więcej
        (np. Forge albo ComfyUI uruchomione wcześniej – liczy się ich szczyt przy obrazie)."""
        grown = [p for p in self.processes if p.before_mib is not None and p.peak_mib
                 and p.peak_mib - p.before_mib >= min_mib]
        return sorted(grown, key=lambda p: p.peak_mib or 0, reverse=True)


class Sampler:
    def __init__(self, run: Callable = CommandRunner(), interval: float = 0.5,
                 cmdline: Callable[[int], Optional[str]] = command_line,
                 clock: Callable[[], float] = time.monotonic):
        self.run, self.interval, self.cmdline, self.clock = run, interval, cmdline, clock
        self.stop = threading.Event()
        self.processes: dict[int, Process] = {}
        self.total_peak: Optional[int] = None
        self.samples = 0
        self.timeline: list[tuple[float, dict[int, int]]] = []   # (sekundy, {pid: MiB})
        self._t0: Optional[float] = None

    def baseline(self) -> tuple[Optional[int], dict]:
        apps = parse_apps(self.run(QUERY_APPS))
        for pid, (name, mem) in apps.items():
            self.processes[pid] = Process(pid, name, self.cmdline(pid), mem, mem)
        return parse_total(self.run(QUERY_TOTAL)), apps

    def sample(self) -> None:
        self.samples += 1
        total = parse_total(self.run(QUERY_TOTAL))
        if total is not None:
            self.total_peak = max(self.total_peak or 0, total)
        now = self.clock()
        if self._t0 is None:
            self._t0 = now
        point: dict[int, int] = {}
        for pid, (name, mem) in parse_apps(self.run(QUERY_APPS)).items():
            p = self.processes.get(pid)
            if p is None:
                p = self.processes[pid] = Process(pid, name, self.cmdline(pid))
            if mem is not None:
                p.peak_mib = max(p.peak_mib or 0, mem)
                point[pid] = mem
        self.timeline.append((round(now - self._t0, 2), point))

    def loop(self) -> None:
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)


def predict(model: str, context: int, resolution: Optional[tuple[int, int]] = None) -> dict:
    """Przewidywanie AI Bóbra (bez narzutu runtime'u) – do porównania z pomiarem."""
    import main
    from aibobr.calculator.memory import memory_need
    from aibobr.domain.models import RunSettings
    c = main.build_app()
    spec = c.load_model(model).spec()
    if spec.image is not None:                                # model obrazów (etap 10)
        from aibobr.calculator.image import image_need
        w, h = resolution or (None, None)
        n = image_need(spec, RunSettings(image_width=w, image_height=h))
        return {"model": spec.name, "quant": spec.quant, "resolution": f"{n.width}×{n.height}",
                "peak_gib": round(n.peak / GIB, 3), "all_gib": round(n.all_at_once / GIB, 3),
                "stage": n.peak_stage, "stages_gib": {k: round(v / GIB, 3) for k, v in n.stages.items()}}
    need = memory_need(spec, RunSettings(context_tokens=context, overhead=0), c.quants)
    llm = need.vram_part - need.decoder
    return {"model": spec.name, "quant": spec.quant, "context": context,
            "llm_gib": round(llm / GIB, 3), "decoder_gib": round(need.decoder / GIB, 3),
            "audio": spec.audio.name if spec.audio else None}


def series(timeline: list[tuple[float, dict[int, int]]], pid: int) -> list[tuple[float, int]]:
    return [(t, pts[pid]) for t, pts in timeline if pid in pts]


def plateaus(points: list[tuple[float, int]], tol_mib: int = 150, min_s: float = 1.0) -> list[tuple[float, float, int]]:
    """Etapy przebiegu: odcinki o podobnym zużyciu (±tol) trwające ≥ min_s → (od, do, szczyt MiB).

    Programy do obrazów pracują etapami (koder tekstu → UNet → VAE); każdy etap to
    „płaskowyż” zużycia karty. Krótkie skoki między nimi pomijamy."""
    out: list[list] = []
    for t, mib in points:
        if out and abs(mib - out[-1][3]) <= tol_mib:
            out[-1][1], out[-1][2] = t, max(out[-1][2], mib)
        else:
            out.append([t, t, mib, mib])                  # od, do, szczyt, poziom odniesienia
    return [(a, b, peak) for a, b, peak, _ in out if b - a >= min_s]


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(points: list[tuple[float, int]], width: int = 60) -> str:
    if not points:
        return ""
    values = [m for _, m in points]
    step = max(1, -(-len(values) // width))
    values = [max(values[i:i + step]) for i in range(0, len(values), step)]
    lo, hi = min(values), max(values)
    return "".join(SPARK[0 if hi == lo else int((v - lo) * (len(SPARK) - 1) / (hi - lo))] for v in values)


def gb(mib: Optional[int]) -> str:
    return "–" if mib is None else f"{mib * MIB / GIB:.2f} GB"


def report(r: Result, log=print, timeline: Optional[list] = None) -> None:
    log(f"\nPomiar: {r.label} ({r.seconds:.0f} s)")
    log(f"  karta razem: przed {gb(r.total_before_mib)}, szczyt {gb(r.total_peak_mib)}, "
        f"przyrost {gb(r.total_delta_mib)}")
    new = r.new_processes()
    if not new:
        log("  Żaden nowy proces nie użył karty (albo sterownik nie podaje pamięci procesów).")
    for p in new:
        log(f"  {gb(p.peak_mib):>9}  pid {p.pid}  {(p.command or p.name)[:90]}")
    for p in r.grown_processes():
        log(f"  {gb(p.peak_mib):>9}  pid {p.pid}  (przed {gb(p.before_mib)})  {(p.command or p.name)[:80]}")
    main = (new or r.grown_processes())[:1]
    if timeline and main:
        pts = series(timeline, main[0].pid)
        log(f"\n  przebieg pid {main[0].pid} ({gb(min(m for _, m in pts))} … {gb(max(m for _, m in pts))}):")
        log(f"  {sparkline(pts)}")
        for a, b, peak in plateaus(pts):
            log(f"    {a:6.1f}–{b:6.1f} s  {gb(peak):>9}")
    pr = r.prediction
    if pr and "peak_gib" in pr:
        log(f"\nAI Bóbr dla {pr['model']} · {pr['resolution']}: szczyt {pr['peak_gib']:.2f} GB "
            f"({pr['stage']}), wszystko naraz {pr['all_gib']:.2f} GB (szacunek)")
    elif pr:
        log(f"\nAI Bóbr dla {pr['model']} · {pr['quant']} · kontekst {pr['context']}: "
            f"model językowy {pr['llm_gib']:.2f} GB + narzut"
            + (f", dekoder {pr['decoder_gib']:.2f} GB (szacunek)" if pr["decoder_gib"] else ""))


def main_cli(argv: Optional[list[str]] = None, stdin=sys.stdin, sampler: Optional[Sampler] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="", help="opis pomiaru, np. „orpheus Q4_K_M, dekoder na GPU”")
    ap.add_argument("--model", help="ten sam model dla AI Bóbra (plik .gguf, link HF, ollama:NAZWA)")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--resolution", help="modele obrazów: np. 1024 albo 1024x768")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="co ile sekund pytać kartę; dla obrazów 0.2 (krótkie dekodowanie VAE)")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)

    sampler = sampler or Sampler(interval=args.interval)
    total_before, apps = sampler.baseline()
    if total_before is None:
        print("Brak nvidia-smi – to narzędzie mierzy tylko karty NVIDIA.")
        return 1
    print(f"Karta zajęta teraz: {gb(total_before)} (procesów: {len(apps)}).")
    prediction = None
    if args.model:
        try:
            from aibobr.cli.cli import parse_resolution
            res = parse_resolution(args.resolution) if args.resolution else None
            prediction = predict(args.model, args.context, res)
        except Exception as e:                                # pomiar ma działać i bez tego
            print(f"(Nie udało się policzyć przewidywania: {e})")
    print("Uruchom teraz model i wygeneruj nagranie (albo obraz). Gdy skończysz – naciśnij Enter.")
    thread = threading.Thread(target=sampler.loop, daemon=True)
    start = time.time()
    thread.start()
    try:
        stdin.readline()
    except KeyboardInterrupt:
        pass
    sampler.stop.set()
    thread.join(timeout=5)

    r = Result(label=args.label or "pomiar", seconds=time.time() - start,
               total_before_mib=total_before, total_peak_mib=sampler.total_peak,
               processes=list(sampler.processes.values()), prediction=prediction)
    report(r, timeline=sampler.timeline)
    out = args.output or Path(f"measure-{datetime.datetime.now():%Y%m%d-%H%M}.json")
    out.write_text(json.dumps({
        "date": datetime.datetime.now().isoformat(timespec="minutes"), "system": platform.platform(),
        **asdict(r), "total_delta_mib": r.total_delta_mib, "samples": sampler.samples,
        "grown": [p.pid for p in r.grown_processes()],
        "timeline": [[t, {str(pid): m for pid, m in pts.items()}] for t, pts in sampler.timeline],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Zapisano {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
