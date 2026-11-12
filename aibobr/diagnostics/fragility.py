"""Kruchość promptu: odstęp między pierwszym a drugim kandydatem przy każdym wygenerowanym tokenie.

Skąd: wskazówka z r/LocalLLaMA po eksperymencie BOS (11.11.2026) – „zapisuj odstęp między najlepszym
kandydatem a drugim; krok z odstępem bliskim zera oznacza kruchy prompt, zanim porówna się silniki”.
Różnica wyniku na zadaniu kruchym może być przeważonym remisem; na stabilnym – rzeczywistą różnicą wejścia.

Zasady (decyzje autora 12.11.2026):
- osobny przebieg z `logprobs` – nigdy w trakcie benchmarku ani testu jakości (sama prośba o logprobs
  zmieniała zachowanie pamięci promptu w eksperymencie z 7.11);
- każde zadanie od zera (model zwolniony), te same ustawienia co test jakości (num_ctx, limit tokenów,
  temperatura 0, seed 42) – inaczej odstępy dotyczyłyby innej odpowiedzi;
- progi ustawiane przez użytkownika; domyślnie: kruche < 0,10, remis < 0,02 (różnica prawdopodobieństw);
- odstęp liczy się tylko na ścieżce, którą model wybrał – najwięcej mówi pierwszy prawie remis;
- etykieta liczy tylko kroki **w odpowiedzi i przed nią** (decyzja autora 12.11.2026, wariant A): pomiar na
  Gemmie 3 i Minitronie pokazał, że prawie remisy w objaśnieniu po odpowiedzi (docstring „czy”/„jest”,
  „przysłówkiem”/„przysłówków”) dawały fałszywe „kruche”. Miejsce odpowiedzi wskazuje ten sam kod oceny co
  w teście jakości (liczba, słowo, opcja, blok JSON, kod); kroki po niej – osobna liczba, bez etykiety. Przy
  kodzie odpowiedzią jest kod bez docstringów i komentarzy (S14 Gemmy: remis „czy”/„jest” w docstringu), przy
  JSON – cały blok. Gdy odpowiedzi nie da się wskazać – cała wypowiedź.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional

FRAGILE = 0.10
TIE = 0.02
TOP = 2
STABLE, FRAGILE_LABEL, TIE_LABEL, UNSUPPORTED, NOT_RUN = "stable", "fragile", "tie", "unsupported", "not_run"
CONTEXT_CHARS = 40


@dataclass(frozen=True)
class Thresholds:
    fragile: float = FRAGILE
    tie: float = TIE

    def __post_init__(self):
        if not (0 <= self.tie <= self.fragile <= 1):
            raise ValueError("progi: 0 ≤ remis ≤ kruche ≤ 1")


@dataclass(frozen=True)
class Step:
    pos: int
    token: str
    first: Optional[tuple[str, float]]       # (token, prawdopodobieństwo) – najlepszy kandydat
    second: Optional[tuple[str, float]]      # drugi kandydat; None – Ollama podała tylko jednego
    margin: float                            # p1 − p2 (bez drugiego: p1)


@dataclass(frozen=True)
class TaskFragility:
    task_id: str
    label: str                               # stable / fragile / tie / unsupported / not_run
    steps: int = 0
    min_margin: Optional[float] = None
    min_pos: Optional[int] = None
    min_pair: Optional[tuple] = None         # ((token1, p1), (token2, p2)) w najwęższym miejscu
    fragile_steps: int = 0
    tie_steps: int = 0
    first_pos: Optional[int] = None          # pierwszy krok poniżej progu „kruche”
    first_pair: Optional[tuple] = None
    first_context: Optional[str] = None      # tekst odpowiedzi tuż przed pierwszym prawie remisem
    answer: str = ""
    reason: str = ""
    scope: str = "whole"                     # answer – etykieta z kroków do końca odpowiedzi; whole – cała wypowiedź
    answer_steps: Optional[int] = None       # ile kroków do końca odpowiedzi (włącznie)
    after_fragile: int = 0                   # kroki poniżej progu „kruche” po odpowiedzi (bez wpływu na etykietę)
    after_tie: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _p(entry: dict) -> Optional[float]:
    lp = entry.get("logprob")
    return math.exp(lp) if isinstance(lp, (int, float)) else None


def steps(logprobs: Iterable[dict]) -> list[Step]:
    """Z odpowiedzi Ollamy (logprobs + top_logprobs) – odstęp przy każdym tokenie."""
    out = []
    for pos, e in enumerate(logprobs or []):
        cands = [(c.get("token"), _p(c)) for c in (e.get("top_logprobs") or []) if _p(c) is not None]
        cands.sort(key=lambda c: c[1], reverse=True)
        if not cands:                             # bez kandydatów – tylko wybrany token
            p = _p(e)
            if p is None:
                continue
            cands = [(e.get("token"), p)]
        first = cands[0]
        second = cands[1] if len(cands) > 1 else None
        out.append(Step(pos, str(e.get("token") or ""), first, second, first[1] - (second[1] if second else 0.0)))
    return out


def _number(text: str) -> Optional[float]:
    try:
        return float(re.sub(r"[\s ]", "", text).replace("−", "-").replace(",", "."))
    except ValueError:
        return None


def answer_end(text: str, spec: Optional[dict], lang: str) -> Optional[int]:
    """Gdzie w tekście kończy się to, co ocenia test jakości (indeks znaku za odpowiedzią); None – nie wiadomo.
    Ten sam kod oceny co w teście; szukamy ostatniego wystąpienia wydzielonej wartości (tak jak ocena bierze
    odpowiedź końcową)."""
    if not spec or not text:
        return None
    from aibobr.quality import checks
    kind = spec.get("type")
    if kind in (None, "manual"):
        return None
    try:
        c = checks.check(text, spec, lang, run_python=None)
    except Exception:
        return None
    value = c.value
    if isinstance(value, str) and checks.unwrap(value) == checks.unwrap(checks.body(text)) and \
            (c.reason in (checks.AMBIGUOUS, checks.WRONG, checks.EMPTY) or kind not in ("text", "choice")):
        return None                 # ocena nie wydzieliła odpowiedzi – oddała całą wypowiedź (H15 Minitrona)
    if kind in ("text", "choice") and isinstance(value, str) and value.strip():
        i = text.casefold().rfind(value.strip().casefold())
        return i + len(value.strip()) if i >= 0 else None
    if kind == "number" and isinstance(value, (int, float)):
        hits = [m for m in checks.NUMBER.finditer(text)
                if _number(m.group()) is not None and abs(_number(m.group()) - value) <= 1e-9 * max(1, abs(value))]
        return hits[-1].end() if hits else None
    if kind == "fraction" and isinstance(value, str) and "/" in value:
        num, den = value.split("/", 1)
        pattern = re.compile(rf"(?<!\d){re.escape(num)}\s*/\s*{re.escape(den)}(?!\d)|"
                             rf"\\[dt]?frac\{{\s*{re.escape(num)}\s*\}}\{{\s*{re.escape(den)}\s*\}}")
        hits = list(pattern.finditer(text))
        return hits[-1].end() if hits else None
    if kind == "json":
        end = max(text.rfind("}"), text.rfind("]"))
        return end + 1 if end >= 0 else None
    if kind == "python" and isinstance(value, str) and value.strip():
        i = text.find(value.strip())
        return i + len(value.strip()) if i >= 0 else None
    return None


def code_notes(text: str) -> list[tuple[int, int]]:
    """Zakresy znaków docstringów i komentarzy w kodzie Pythona z odpowiedzi (nie wpływają na wynik testów).
    Kod niepoprawny – brak zakresów (liczy się wszystko)."""
    import io
    import tokenize
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, re.S)
    start = m.start(1) if m else 0
    code = m.group(1) if m else text
    offsets, total = [], 0
    for line in code.splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and tok.string.lstrip("rRbBuUfF")[:3]
                                                 in ('"""', "'''")):
                (r1, c1), (r2, c2) = tok.start, tok.end
                if r1 - 1 < len(offsets) and r2 - 1 < len(offsets):
                    out.append((start + offsets[r1 - 1] + c1, start + offsets[r2 - 1] + c2))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    return out


def classify(task_id: str, logprobs: Optional[list], answer: str = "",
             thresholds: Thresholds = Thresholds(), spec: Optional[dict] = None, lang: str = "pl") -> TaskFragility:
    if logprobs is None:
        return TaskFragility(task_id, UNSUPPORTED, answer=answer, reason="no_logprobs")
    st = steps(logprobs)
    if not st:
        return TaskFragility(task_id, NOT_RUN, answer=answer, reason="empty")
    joined = "".join(s.token for s in st)            # pozycje znaków zgodne z tokenami
    end = answer_end(joined, spec, lang)
    scope, cut = "whole", len(st)
    if end is not None:
        pos, cut = 0, 0
        for s in st:
            if pos >= end:
                break
            pos += len(s.token)
            cut += 1
        scope = "answer"
    head, tail = st[:max(cut, 1)], st[max(cut, 1):]
    if spec and spec.get("type") == "python":            # docstringi i komentarze – poza odpowiedzią
        notes = code_notes(joined)
        if notes:
            pos, inside, outside = 0, [], []
            for s_ in head:
                a, b = pos, pos + len(s_.token)
                pos = b
                (outside if s_.token.strip() and any(x <= a and b <= y for x, y in notes) else inside).append(s_)
            if inside:
                head, tail = inside, outside + tail
    low = min(head, key=lambda s: s.margin)
    fragile = [s for s in head if s.margin < thresholds.fragile]
    ties = [s for s in head if s.margin < thresholds.tie]
    label = TIE_LABEL if ties else FRAGILE_LABEL if fragile else STABLE
    first = fragile[0] if fragile else None
    context = None
    if first is not None:
        context = "".join(s.token for s in st[:first.pos])[-CONTEXT_CHARS:]
    return TaskFragility(task_id, label, len(st), round(low.margin, 6), low.pos, (low.first, low.second),
                         len(fragile), len(ties), first.pos if first else None,
                         (first.first, first.second) if first else None, context, answer,
                         scope=scope, answer_steps=len(head) if scope == "answer" else None,
                         after_fragile=sum(1 for s in tail if s.margin < thresholds.fragile),
                         after_tie=sum(1 for s in tail if s.margin < thresholds.tie))


@dataclass
class FragilityRun:
    model: str
    thresholds: Thresholds
    tasks: list[TaskFragility] = field(default_factory=list)

    def counts(self) -> dict:
        out = {STABLE: 0, FRAGILE_LABEL: 0, TIE_LABEL: 0, UNSUPPORTED: 0, NOT_RUN: 0}
        for t in self.tasks:
            out[t.label] = out.get(t.label, 0) + 1
        return out


class FragilityProbe:
    """Przebieg dla jednego modelu i listy zadań [(id, prompt, limit tokenów)]."""

    def __init__(self, bench, clock: Callable[[], float], sleep: Callable[[float], None],
                 unload_timeout: float = 30.0, num_ctx: int = 2048):
        from aibobr.diagnostics.bos import BosProbe           # to samo zwalnianie modelu co pomiar tokenu początku
        self.bench, self.num_ctx = bench, num_ctx
        self._release = BosProbe(bench, clock, sleep, unload_timeout).release

    def run(self, model: str, tasks: list[tuple], thresholds: Thresholds = Thresholds(),
            report: Callable[[dict], None] = lambda e: None,
            stop: Callable[[], bool] = lambda: False) -> FragilityRun:
        result = FragilityRun(model, thresholds)
        try:
            for n, task in enumerate(tasks, 1):
                task_id, prompt, max_tokens = task[:3]
                spec, lang = (task[3], task[4]) if len(task) >= 5 else (None, "pl")
                if stop():
                    break
                report({"event": "f_task", "model": model, "task_id": task_id, "n": n, "total": len(tasks)})
                if not self._release(model):
                    item = TaskFragility(task_id, NOT_RUN, reason="not_released")
                else:
                    try:
                        g = self.bench.generate(model, prompt, self.num_ctx, int(max_tokens), top_logprobs=TOP)
                        item = classify(task_id, g.logprobs, g.text, thresholds, spec, lang)
                        if g.prompt_cached:                  # model miał być zwolniony – wynik nieważny
                            item = TaskFragility(task_id, NOT_RUN, answer=g.text, reason="cached")
                    except Exception as e:
                        item = TaskFragility(task_id, NOT_RUN, reason=f"error: {e}")
                result.tasks.append(item)
                report({"event": "f_task_done", "model": model, "task": item.as_dict()})
                if item.label == UNSUPPORTED:               # ta Ollama nie podaje prawdopodobieństw – dalej bez sensu
                    break
        finally:
            try:
                self._release(model)
            except Exception:
                pass
        return result


def suite_tasks(suite: dict, lang: str) -> list[tuple]:
    """Zadania testu jakości w tej samej postaci, w jakiej idą do modelu: (id, treść, limit tokenów,
    sprawdzenie, język) – sprawdzenie wskazuje miejsce odpowiedzi."""
    return [(t["id"], t["prompt"][lang], int(t.get("max_tokens") or 256), t.get("check"), lang)
            for t in suite["tasks"] if lang in t.get("prompt", {})]
