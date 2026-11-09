"""Benchmark prędkości 0.9 (SPEC §5c, decyzje autora 9.11.2026).

Odpowiada na pytanie: czy ten komputer uruchomi model i jak szybko będzie on działał. **Nie oceniamy
treści odpowiedzi** – zapisujemy ją do podglądu i eksportu. Pilnujemy wiarygodności technicznej
pomiaru: liczby wygenerowanych tokenów, czasu ich generowania i błędów wykonania.

Pakiet: 15 zadań (8 × polszczyzna, 7 × JSON i kod) przy kontekście 2k. Każde zadanie:
zwolnienie modelu (potwierdzone w /api/ps) → wczytanie pustym zapytaniem (ten sam num_ctx) →
zadanie (strumieniowo: czas do pierwszego kawałka mierzy aplikacja) → sprawdzenie pamięci promptu.
Czasy silnika (Ollama: load/prompt_eval/eval/total) osobno od czasów rzeczywistych (zegar aplikacji).

Werdykt: generowanie (średnia ważona: wszystkie tokeny / cały czas generowania) i TTFT (mediana
czasów rzeczywistych) oceniane osobno, werdykt = gorsza z dwóch ocen + która miara przesądziła.
Przy werdykcie zawsze podstawa, np. „54,2 tok/s z 1 214 tokenów w 22,4 s”.

Pomiar długiego kontekstu (1k/8k…) to osobna sekcja: liczy się RZECZYWISTA długość wejścia
(prompt_eval_count), nie ustawiony num_ctx. Długość dopasowujemy do tokenizatora modelu
(kalibracja), zostawiamy rezerwę na odpowiedź, a generowanie mierzymy PO długim wejściu.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from aibobr.bench import prompts
from aibobr.bench.runner import Stopped

SUITE_ID = "speed_v1"
CONTEXT = 2048
UNLOAD_TIMEOUT = 30.0

LEVELS = ("fast", "useful", "slow", "too_slow")          # 🟢 Szybko, 🔵 Użytecznie, 🟡 Wolno, 🔴 Zbyt wolno
THRESHOLDS_VERSION = "1"
# Progi autora (9.11.2026): generowanie ≥30 / ≥15 / ≥6 tok/s; TTFT ≤1 / ≤3 / ≤8 s.
DEFAULT_THRESHOLDS = {"gen_tps": [30.0, 15.0, 6.0], "ttft_s": [1.0, 3.0, 8.0]}

# Sprawdzenia wiarygodności (tabela zatwierdzona 9.11.2026)
FEW_TOKENS = 20               # odpowiedź krótsza – dopisek „mało tokenów”, do średniej wchodzi
MIN_TOTAL_TOKENS = 300        # łącznie mniej – ostrzeżenie „ograniczona podstawa pomiaru” (tylko pakiet 15 zadań)
MIN_TOTAL_GEN_S = 5.0         # …albo łącznie krócej generowania (decyzja autora 9.11.2026); werdykt bez zmian
MAX_OVERHEAD = 0.20           # narzut poza silnikiem (czas rzeczywisty − total_duration) / czas rzeczywisty
MAX_ERRORS = 2                # więcej błędów wykonania na 15 zadań – „brak wiarygodnego wyniku”
MAX_TPS = 5000.0              # prędkość generowania ponad – błąd pomiaru czasu
RELOAD_S = 0.5                # load_duration zadania ponad – model wczytywał się w trakcie (TTFT zafałszowany)
VRAM_RECOVER_MB = 300         # po zwolnieniu VRAM wraca do poziomu sprzed sesji ± tyle

# Długi kontekst
LONG_RESERVE = 128            # tokeny odpowiedzi po długim wejściu (generowanie przy zapełnionym kontekście)
LONG_MARGIN = 32              # zapas na szablon czatu i polecenie
LONG_RANGE = (0.80, 0.95)     # ważne wejście: 80–95% (num_ctx − rezerwa)
LONG_TARGET = 0.90
LONG_TOLERANCE = 0.05         # wejście dalej od celu – jedna poprawka (porównywalne przebiegi)


# ------------------------------------------------------------------ progi i werdykt
def thresholds_of(settings: Optional[dict]) -> dict:
    """Progi z ustawień (`bench_thresholds`); niepoprawne (nie malejące / nie rosnące) – domyślne."""
    th = (settings or {}).get("bench_thresholds") or {}
    out = {k: list(v) for k, v in DEFAULT_THRESHOLDS.items()}
    try:
        gen = [float(x) for x in th.get("gen_tps", out["gen_tps"])]
        ttft = [float(x) for x in th.get("ttft_s", out["ttft_s"])]
    except (TypeError, ValueError):
        return out
    if len(gen) == 3 and gen[0] > gen[1] > gen[2] > 0:
        out["gen_tps"] = gen
    if len(ttft) == 3 and 0 < ttft[0] < ttft[1] < ttft[2]:
        out["ttft_s"] = ttft
    return out


def gen_level(tps: Optional[float], th: dict = DEFAULT_THRESHOLDS) -> Optional[str]:
    if tps is None:
        return None
    fast, useful, slow = th["gen_tps"]
    return "fast" if tps >= fast else "useful" if tps >= useful else "slow" if tps >= slow else "too_slow"


def ttft_level(seconds: Optional[float], th: dict = DEFAULT_THRESHOLDS) -> Optional[str]:
    if seconds is None:
        return None
    fast, useful, slow = th["ttft_s"]
    return ("fast" if seconds <= fast else "useful" if seconds <= useful else "slow" if seconds <= slow
            else "too_slow")


def verdict(gen_tps: Optional[float], ttft_s: Optional[float],
            th: dict = DEFAULT_THRESHOLDS) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """(werdykt, która miara przesądziła: gen / ttft / both, ocena generowania, ocena TTFT)."""
    g, f = gen_level(gen_tps, th), ttft_level(ttft_s, th)
    if g is None and f is None:
        return None, None, None, None
    if f is None or (g is not None and LEVELS.index(g) > LEVELS.index(f)):
        return g, "gen", g, f
    if g is None or LEVELS.index(f) > LEVELS.index(g):
        return f, "ttft", g, f
    return g, "both", g, f


def recompute(run: dict, th: dict) -> Optional[str]:
    """Werdykt według innych (np. obecnych) progów z zapisanych surowych wartości przebiegu."""
    if run.get("status") == "unreliable":
        return None
    gen = run.get("gen_tps")
    ttft = run.get("ttft_wall_median_ms")
    return verdict(gen, ttft / 1000 if ttft is not None else None, th)[0]


# ------------------------------------------------------------------ pakiet zadań
def build_pack(suites: list[dict]) -> dict:
    """15 zadań z pakietów polszczyzny i formatu – tylko prompty i limity tokenów (bez oceniania)."""
    tasks = []
    for suite in suites:
        cfg = suite.get("default_generation_config", {})
        for test in suite.get("tests", ()):
            tasks.append({"id": test["id"], "category": test.get("category"), "prompt": test["prompt"],
                          "num_predict": int(test.get("max_tokens") or cfg.get("max_tokens") or 256)})
    version = "+".join(f"{s['suite_info']['id']}@{s['suite_info'].get('version', '?')}" for s in suites)
    return {"id": SUITE_ID, "version": version, "tasks": tasks}


# ------------------------------------------------------------------ wynik zadania
@dataclass
class TaskResult:
    task_no: int
    task_id: str
    category: Optional[str] = None
    prompt_text: str = ""
    num_ctx: int = CONTEXT
    num_predict: int = 256
    prompt_tokens: Optional[int] = None
    prompt_cached: Optional[int] = None
    gen_tokens: Optional[int] = None
    load_ms: Optional[float] = None
    prompt_eval_ms: Optional[float] = None
    eval_ms: Optional[float] = None
    total_ms: Optional[float] = None
    ttft_wall_ms: Optional[float] = None
    wall_ms: Optional[float] = None
    done_reason: Optional[str] = None
    response_text: str = ""
    error: Optional[str] = None
    counted_gen: bool = False
    counted_pp: bool = False
    flags: list = field(default_factory=list)
    load_wall_ms: Optional[float] = None          # puste zapytanie wczytujące: czas rzeczywisty
    preload_ms: Optional[float] = None            # …i load_duration silnika z tego zapytania
    released: bool = False

    @property
    def counted_ttft(self) -> bool:
        return self.error is None and self.ttft_wall_ms is not None and "loaded_during_task" not in self.flags

    def row(self) -> dict:
        return {"task_no": self.task_no, "task_id": self.task_id, "category": self.category,
                "prompt_text": self.prompt_text, "num_ctx": self.num_ctx, "num_predict": self.num_predict,
                "prompt_tokens": self.prompt_tokens, "prompt_cached": self.prompt_cached,
                "gen_tokens": self.gen_tokens, "load_ms": self.load_ms, "prompt_eval_ms": self.prompt_eval_ms,
                "eval_ms": self.eval_ms, "total_ms": self.total_ms, "ttft_wall_ms": self.ttft_wall_ms,
                "wall_ms": self.wall_ms, "done_reason": self.done_reason, "response_text": self.response_text,
                "error": self.error, "counted_gen": int(self.counted_gen), "counted_pp": int(self.counted_pp),
                "flags_json": self.flags}


def check(r: TaskResult) -> TaskResult:
    """Sprawdzenia wiarygodności technicznej – ustawia error, counted_* i flagi."""
    if r.error is None:
        if not r.gen_tokens or not r.response_text.strip():
            r.error = "no_tokens"
        elif not r.eval_ms or r.eval_ms <= 0 or r.gen_tokens / (r.eval_ms / 1000) > MAX_TPS:
            r.error = "bad_timing"
    if r.error is not None:
        r.counted_gen = r.counted_pp = False
        return r
    r.counted_gen = True
    if r.gen_tokens < FEW_TOKENS:
        r.flags.append("few_tokens")
    if r.prompt_cached is None:
        r.flags.append("cache_unverified")      # Ollama nie podaje; zwolnienie i tak wymusza liczenie od zera
    elif r.prompt_cached > 0:
        r.flags.append("cached")
    if not r.released:
        r.flags.append("not_released")
    r.counted_pp = (r.released and not r.prompt_cached and bool(r.prompt_tokens)
                    and bool(r.prompt_eval_ms and r.prompt_eval_ms > 0))
    if r.load_ms is not None and r.load_ms > RELOAD_S * 1000:
        r.flags.append("loaded_during_task")
    if r.done_reason == "length":
        r.flags.append("length")
    return r


# ------------------------------------------------------------------ podsumowanie przebiegu
def median(values: list) -> Optional[float]:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(tasks: list[TaskResult], th: dict = DEFAULT_THRESHOLDS) -> dict:
    """Surowe sumy, werdykt z dnia testu i ostrzeżenia (kolumny tabeli `runs`)."""
    gen = [t for t in tasks if t.counted_gen]
    pp = [t for t in tasks if t.counted_pp]
    errors = [t for t in tasks if t.error is not None]
    gen_tokens = sum(t.gen_tokens for t in gen)
    gen_ms = sum(t.eval_ms for t in gen)
    pp_tokens = sum(t.prompt_tokens for t in pp)
    pp_ms = sum(t.prompt_eval_ms for t in pp)
    ttft = [t.ttft_wall_ms for t in tasks if t.counted_ttft]
    done = [t for t in tasks if t.error is None]
    engine = sum(t.total_ms for t in done if t.total_ms is not None) if done else None
    wall = sum(t.wall_ms for t in done if t.wall_ms is not None) if done else None
    overhead = (wall - engine) / wall if engine is not None and wall else None
    out = {"gen_tokens_total": gen_tokens, "gen_ms_total": round(gen_ms, 1),
           "gen_tps": round(gen_tokens / (gen_ms / 1000), 2) if gen_ms > 0 else None,
           "pp_tokens_total": pp_tokens, "pp_ms_total": round(pp_ms, 1),
           "pp_tps": round(pp_tokens / (pp_ms / 1000), 1) if pp_ms > 0 else None,
           "ttft_wall_median_ms": round(median(ttft), 1) if ttft else None,
           "ttft_wall_max_ms": round(max(ttft), 1) if ttft else None,
           "engine_ms_total": round(engine, 1) if engine is not None else None,
           "wall_ms_total": round(wall, 1) if wall is not None else None,
           "overhead_ratio": round(overhead, 4) if overhead is not None else None,
           "load_ms": median([t.preload_ms for t in tasks]),
           "load_wall_ms": median([t.load_wall_ms for t in tasks]),
           "tasks_counted": len(gen), "tasks_error": len(errors)}
    warnings = []
    if gen_tokens < MIN_TOTAL_TOKENS or gen_ms < MIN_TOTAL_GEN_S * 1000:
        warnings.append("limited_basis")
    if overhead is not None and overhead > MAX_OVERHEAD:
        warnings.append("overhead")
    if any(t.error is None and not t.released for t in tasks):
        warnings.append("not_released")
    if len(errors) > MAX_ERRORS:
        out.update(status="unreliable", verdict=None, verdict_measure=None, gen_level=None, ttft_level=None)
        warnings.append("too_many_errors")
    else:
        level, measure, g, f = verdict(out["gen_tps"], out["ttft_wall_median_ms"] / 1000
                                       if out["ttft_wall_median_ms"] is not None else None, th)
        out.update(status="finished", verdict=level, verdict_measure=measure, gen_level=g, ttft_level=f)
    out["warnings"] = warnings
    return out


def spread(runs: list[dict]) -> Optional[tuple[float, float]]:
    """Zakres generowania z kilku przebiegów (min, max) – pomiar rozrzutu, nie dowód powtarzalności."""
    values = [r["gen_tps"] for r in runs if r.get("gen_tps") is not None and r.get("status") == "finished"]
    return (min(values), max(values)) if values else None


# ------------------------------------------------------------------ wykonanie
class SpeedRunner:
    """Jeden model, jeden przebieg pakietu (i opcjonalnie długi kontekst). Bez telemetrii i bazy –
    to robi `SpeedBenchmark`."""

    def __init__(self, bench, pack: dict, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep, unload_timeout: float = UNLOAD_TIMEOUT):
        self.bench, self.pack = bench, pack
        self.clock, self.sleep, self.unload_timeout = clock, sleep, unload_timeout

    def release(self, model: str) -> bool:
        """keep_alive 0 i czekanie, aż modelu nie ma w /api/ps (samo żądanie niczego nie potwierdza)."""
        start = self.clock()
        try:
            self.bench.unload(model)
            while self.bench.is_loaded(model):
                if self.clock() - start >= self.unload_timeout:
                    return False
                self.sleep(0.25)
        except Exception:
            return False
        return True

    def _generate(self, model: str, prompt: str, num_ctx: int, num_predict: int,
                  on_chunk: Callable[[str], None]) -> tuple:
        first = {}

        def chunk(text):
            if "t" not in first:
                first["t"] = self.clock()
            on_chunk(text)
        start = self.clock()
        g = self.bench.generate(model, prompt, num_ctx, num_predict, on_chunk=chunk)
        end = self.clock()
        ttft = (first["t"] - start) * 1000 if "t" in first else None
        return g, ttft, (end - start) * 1000

    def task(self, model: str, n: int, task: dict, on_chunk: Callable[[str], None] = lambda s: None) -> TaskResult:
        r = TaskResult(n, task["id"], task.get("category"), task["prompt"], CONTEXT, task["num_predict"])
        r.released = self.release(model)
        try:
            start = self.clock()
            engine_load = self.bench.load(model, CONTEXT)
            r.load_wall_ms = round((self.clock() - start) * 1000, 1)
            r.preload_ms = round(engine_load * 1000, 1) if engine_load is not None else None
            g, ttft, wall = self._generate(model, task["prompt"], CONTEXT, task["num_predict"], on_chunk)
        except Stopped:
            raise
        except Exception as e:                                   # HTTP, przekroczenie czasu, brak pamięci
            r.error = f"execution: {e}"[:500]
            return check(r)
        r.prompt_tokens, r.prompt_cached, r.gen_tokens = g.prompt_tokens, g.prompt_cached, g.gen_tokens
        r.load_ms = round(g.load_seconds * 1000, 1)
        r.prompt_eval_ms = round(g.prompt_seconds * 1000, 1)
        r.eval_ms = round(g.gen_seconds * 1000, 1)
        r.total_ms = round(g.total_seconds * 1000, 1) if g.total_seconds is not None else None
        r.ttft_wall_ms = round(ttft, 1) if ttft is not None else None
        r.wall_ms = round(wall, 1)
        r.done_reason, r.response_text = g.done_reason, g.text
        try:
            if not self.bench.is_loaded(model):
                r.error = "model_unloaded"                      # zniknął w trakcie (np. brak pamięci)
        except Exception:
            pass
        return check(r)

    def run_pack(self, model: str, report: Callable[[dict], None] = lambda e: None,
                 stop: Optional[threading.Event] = None,
                 on_task: Callable[[int], None] = lambda n: None) -> list[TaskResult]:
        out = []
        total = len(self.pack["tasks"])
        for n, task in enumerate(self.pack["tasks"], 1):
            if stop is not None and stop.is_set():
                raise Stopped()
            on_task(n)
            report({"event": "s_task", "model": model, "n": n, "total": total, "task": task["id"],
                    "prompt": task["prompt"]})
            r = self.task(model, n, task, lambda text: report({"event": "chunk", "text": text}))
            out.append(r)
            report({"event": "s_task_done", "model": model, "n": n, "total": total, "result": r})
        return out

    # -------------------------------------------------------------- długi kontekst
    def long_context(self, model: str, num_ctx: int, run: int = 0,
                     on_chunk: Callable[[str], None] = lambda s: None,
                     on_retry: Callable[[int, int, int], None] = lambda tokens, low, high: None) -> dict:
        """Wejście dopasowane do tokenizatora modelu. Dwie krótkie próbki różnej długości dają
        znaki na token ORAZ stały narzut (szablon czatu, polecenie) – jedna próbka zawyżała liczbę
        tokenów i pierwsze wejście wychodziło ok. 10–15% za krótkie (pomiar autora 9.11.2026: 653–671
        zamiast ok. 777 tokenów przy 1k). Wejście dalej niż 5% od celu – jedna poprawka, żeby kolejne
        przebiegi miały porównywalną długość; ważność: 80–95% (num_ctx − rezerwa)."""
        usable = num_ctx - LONG_RESERVE - LONG_MARGIN
        target = int(usable * LONG_TARGET)
        low, high = int((num_ctx - LONG_RESERVE) * LONG_RANGE[0]), int((num_ctx - LONG_RESERVE) * LONG_RANGE[1])
        out = {"num_ctx": num_ctx, "target_tokens": target, "reserve_tokens": LONG_RESERVE}
        calibration = {"attempts": [], "probes": []}
        try:
            released = self.release(model)
            self.bench.load(model, num_ctx)
            cpt, fixed = self._calibrate(model, num_ctx, usable, run, calibration)
            for attempt in range(2):
                text = long_text(max(64, (target - fixed) * cpt - len(LONG_TAIL)), run * 10 + attempt)
                released = self.release(model) and released
                self.bench.load(model, num_ctx)
                g, ttft, wall = self._generate(model, text, num_ctx, LONG_RESERVE, on_chunk)
                calibration["attempts"].append({"chars": len(text), "prompt_tokens": g.prompt_tokens})
                if attempt == 1 or not g.prompt_tokens or abs(g.prompt_tokens - target) <= LONG_TOLERANCE * target:
                    break
                if g.prompt_tokens > fixed:                              # poprawka z rzeczywistego wyniku
                    cpt = len(text) / (g.prompt_tokens - fixed)
                calibration["chars_per_token"] = round(cpt, 3)
                on_retry(g.prompt_tokens, low, high)
        except Stopped:
            raise
        except Exception as e:
            out.update(valid=0, invalid_reason="error", error=f"execution: {e}"[:500], calibration_json=calibration)
            return out
        free = num_ctx - g.prompt_tokens - g.gen_tokens
        out.update(prompt_tokens=g.prompt_tokens, prompt_cached=g.prompt_cached, gen_tokens=g.gen_tokens,
                   free_tokens=free, prompt_eval_ms=round(g.prompt_seconds * 1000, 1),
                   eval_ms=round(g.gen_seconds * 1000, 1), pp_tps=_round(g.pp_tps, 1), tg_tps=_round(g.tg_tps, 2),
                   ttft_wall_ms=_round(ttft, 1), done_reason=g.done_reason, calibration_json=calibration)
        try:
            out["gpu_share"] = self.bench.gpu_share(model)
        except Exception:
            out["gpu_share"] = None
        reason = None
        if g.prompt_tokens < low:
            reason = "input_too_short"
        elif g.prompt_tokens > high or free <= 0:
            reason = "possible_truncation"
        elif g.prompt_cached:
            reason = "cached"
        elif not released:
            reason = "not_released"
        elif not g.gen_tokens or not g.gen_seconds:
            reason = "no_tokens"
        out.update(valid=int(reason is None), invalid_reason=reason)
        return out

    def _calibrate(self, model: str, num_ctx: int, usable: int, run: int, calibration: dict) -> tuple[float, float]:
        """(znaki na token, stały narzut w tokenach) z dwóch próbek: tokeny = narzut + znaki / cpt."""
        sizes = (300, max(1200, min(6000, int(usable * prompts.CHARS_PER_TOKEN * 0.25))))
        points = []
        for i, chars in enumerate(sizes):
            text = long_text(chars, 9000 + run * 10 + i)
            g = self.bench.generate(model, text, num_ctx, 1)
            points.append((len(text), g.prompt_tokens))
            calibration["probes"].append({"chars": len(text), "prompt_tokens": g.prompt_tokens})
        (c1, t1), (c2, t2) = points
        if t2 > t1 and c2 > c1:
            cpt = (c2 - c1) / (t2 - t1)
            fixed = max(0.0, t2 - c2 / cpt)
        else:                                                           # brak liczb – zgrubnie
            cpt, fixed = (c2 / t2 if t2 else prompts.CHARS_PER_TOKEN), 0.0
        calibration["chars_per_token"] = round(cpt, 3)
        calibration["fixed_tokens"] = round(fixed, 1)
        return cpt, fixed


LONG_TAIL = "\n\nNow continue: write a long, detailed story about the beavers from the text above. Do not stop early."


def long_text(chars: float, run: int) -> str:
    """Tekst ok. `chars` znaków (opowieść z prompts.filler) + polecenie, po którym model pisze długo
    (mierzymy generowanie po długim wejściu)."""
    return prompts.filler(max(8, int(chars / prompts.CHARS_PER_TOKEN)), run) + LONG_TAIL


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return round(value, digits) if value is not None else None


def basis(run: dict) -> Optional[dict]:
    """Podstawa werdyktu do pokazania: tok/s, tokeny, sekundy (np. „54,2 tok/s z 1 214 tokenów w 22,4 s”)."""
    if run.get("gen_tps") is None:
        return None
    return {"tps": run["gen_tps"], "tokens": run["gen_tokens_total"], "seconds": round(run["gen_ms_total"] / 1000, 1)}


def as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False)
