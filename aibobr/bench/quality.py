"""Testy Zacharskiego – jakość (etap 12c, SPEC §5c): polszczyzna, wiedza, JSON i kod.

Czy kwantyzacja nie psuje odpowiedzi. Temperatura 0, seed 42, myślenie wyłączone, kontekst 2k.

Pamięć promptu (eksperyment kontrolny, tools/eksperyment_cache.py): odpowiedź zależy od tego,
czy Ollama ma ten sam prompt w pamięci – liczony od zera daje inną odpowiedź niż wzięty z pamięci
(8 z 10 par model–zadanie). Dlatego przed KAŻDĄ próbą idzie krótki przerywnik z innym promptem:
w pamięci zostaje tylko wspólny początek szablonu czatu, jak przy nowym pytaniu w zwykłej pracy.
Każda próba zapisuje prompt_eval_cached_count; gdy z pamięci przyszła ponad połowa promptu,
wynik dostaje ostrzeżenie (cache_warn) zamiast cichego wliczenia.
Ocena:
- regex / contains_all – pakiet polszczyzny (wzorce autora),
- json – ŚCISŁY (cała odpowiedź to JSON) i ŁAGODNY (JSON wyjęty z ```json albo z tekstu),
- python_syntax – tylko ast.parse i nazwy funkcji / klas. Kodu NIGDY nie uruchamiamy.
Przed oceną odpowiedź jest oczyszczana: bloki <think>, spacje, cudzysłowy i kropka na końcu.
"""
from __future__ import annotations

import ast
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from aibobr.bench.runner import Stopped

CONTEXT = 2048
SUITES = ("pl_jezyk_v1", "en_format_v1")
# Kolumny tabeli „Jakość”. Polszczyzna w podkategoriach (uwaga z Reddita: składnia liczebnika
# to inne zjawisko niż odmiana – średnia z 7 zadań mieszała je); wiedza osobno.
SUBCATEGORIES = ("spelling", "inflection", "numerals", "aspect", "knowledge")
REPEATS = (1, 3, 5)
FILLER = "Reply with one word: OK"       # przerywnik – inny prompt niż każde zadanie
FILLER_TOKENS = 1

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.S)


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str = ""                     # klucz i18n albo krótki opis (do podglądu)
    strict: Optional[bool] = None        # tylko JSON: czy przeszedł też ściśle


@dataclass(frozen=True)
class QualityResult:
    model: str
    suite: str
    test_id: str
    title: str
    category: str
    passed: bool
    reason: str
    answer: str
    prompt: str
    strict: Optional[bool] = None
    skipped: Optional[str] = None
    subcategory: str = ""
    runs: int = 1                        # ile razy zadano to samo pytanie (powtórzenia)
    identical: int = 1                   # ile odpowiedzi identycznych z pierwszą
    passes: int = 0                      # ile przebiegów zaliczonych
    prompt_tokens: int = 0               # długość promptu w tokenach (prompt_eval_count)
    cached: Optional[int] = None         # najwięcej tokenów promptu z pamięci w którejś próbie; None = brak pola
    cache_warn: bool = False             # któraś próba wzięła z pamięci ponad połowę promptu


def clean(text: str) -> str:
    text = _THINK.sub("", text or "").strip()
    text = text.strip().strip('"„”“').strip()
    return text


def _short(text: str) -> str:
    """Do porównań wzorców: bez kropki na końcu i cudzysłowów wokół."""
    return clean(text).rstrip(".").strip('"„”“').strip()


# ---------------------------------------------------------------------- JSON
def _embedded(text: str) -> Any:
    """Pierwszy poprawny obiekt {...} albo tablica [...] w tekście; ValueError, gdy brak."""
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch in "[{":
            try:
                return decoder.raw_decode(text, start)[0]
            except ValueError:
                continue
    raise ValueError("no JSON")


class BadJson(ValueError):
    """JSON jest w odpowiedzi, ale ma błąd składni (np. "\\ " poza dozwolonymi sekwencjami)."""


def parse_json(answer: str) -> tuple[Any, bool]:
    """(dane, ściśle) – ściśle: cała odpowiedź to JSON; inaczej z bloku ``` albo z tekstu.
    ValueError, gdy JSON-u nie ma."""
    text = clean(answer)
    try:
        return json.loads(text), True
    except ValueError:
        pass
    for block in _FENCE.findall(text):
        try:
            return json.loads(block.strip()), False
        except ValueError:
            continue
    try:
        return _embedded(text), False
    except ValueError:
        if any(c in text for c in "[{"):
            raise BadJson("bad JSON") from None
        raise


def _at(data: Any, path: str) -> Any:
    for part in [p for p in path.split(".") if p]:
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def _lower(x: Any) -> Any:
    return x.lower() if isinstance(x, str) else [_lower(i) for i in x] if isinstance(x, list) else x


def check_json(answer: str, ev: dict) -> Verdict:
    try:
        data, strict = parse_json(answer)
    except BadJson:
        return Verdict(False, "quality.reason.bad_json", strict=False)
    except ValueError:
        return Verdict(False, "quality.reason.no_json", strict=False)
    expect = ev.get("expect")
    if expect == "object" and not isinstance(data, dict) or expect == "array" and not isinstance(data, list):
        return Verdict(False, "quality.reason.json_type", strict=strict)
    for key in ev.get("required_keys", ()):
        if not isinstance(data, dict) or key not in data:
            return Verdict(False, f"quality.reason.json_key:{key}", strict=strict)
    for chk in ev.get("checks", ()):
        try:
            value = _at(data, chk.get("path", ""))
        except (KeyError, IndexError, TypeError, ValueError):
            return Verdict(False, f"quality.reason.json_value:{chk.get('path')}", strict=strict)
        sensitive = chk.get("case_sensitive", True)
        v = value if sensitive else _lower(value)
        if "equals" in chk:
            want = chk["equals"] if sensitive else _lower(chk["equals"])
            ok = v == want or (isinstance(v, (int, float)) and isinstance(want, (int, float))
                               and not isinstance(v, bool) and float(v) == float(want))
        elif "contains" in chk:
            want = chk["contains"] if sensitive else _lower(chk["contains"])
            ok = isinstance(v, str) and want in v
        elif "contains_all" in chk:
            wants = chk["contains_all"] if sensitive else _lower(chk["contains_all"])
            ok = isinstance(v, list) and all(w in v for w in wants)
        else:
            ok = True
        if not ok:
            return Verdict(False, f"quality.reason.json_value:{chk.get('path') or '[]'}", strict=strict)
    return Verdict(True, "", strict=strict)


# ---------------------------------------------------------------------- Python
def _parse(code: str) -> Optional[ast.Module]:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def parse_code(answer: str) -> tuple[Optional[ast.Module], bool]:
    """(drzewo, ściśle) – ściśle: cała odpowiedź to kod; łagodnie: blok ``` albo kod po odcięciu
    wstępu („Code:”) i komentarza pod spodem. ast.parse tylko czyta – nic nie wykonuje."""
    text = clean(answer)
    tree = _parse(text)
    if tree is not None:
        return tree, True
    blocks = _FENCE.findall(text)
    for block in sorted(blocks, key=len, reverse=True):
        tree = _parse(block.strip())
        if tree is not None:
            return tree, False
    lines = text.splitlines()
    for start in range(len(lines)):                       # najdłuższy fragment, który jest kodem
        for end in range(len(lines), start, -1):
            chunk = "\n".join(lines[start:end]).strip()
            if chunk and any(k in chunk for k in ("def ", "class ")):
                tree = _parse(chunk)
                if tree is not None:
                    return tree, False
    return None, False


def check_python(answer: str, ev: dict) -> Verdict:
    """Tylko składnia i nazwy – ast.parse nie wykonuje kodu."""
    tree, strict = parse_code(answer)
    if tree is None:
        return Verdict(False, "quality.reason.syntax", strict=False)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if ev.get("function") and ev["function"] not in funcs:
        return Verdict(False, f"quality.reason.no_function:{ev['function']}", strict=strict)
    if ev.get("class"):
        cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == ev["class"]), None)
        if cls is None:
            return Verdict(False, f"quality.reason.no_class:{ev['class']}", strict=strict)
        methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        missing = [m for m in ev.get("methods", ()) if m not in methods]
        if missing:
            return Verdict(False, f"quality.reason.no_methods:{', '.join(missing)}", strict=strict)
    return Verdict(True, strict=strict)


# ---------------------------------------------------------------------- ocena
def evaluate(answer: str, ev: dict) -> Verdict:
    kind = ev.get("type")
    if kind == "regex":
        return Verdict(bool(re.search(ev["pattern"], _short(answer))), "quality.reason.pattern")
    if kind == "contains_all":
        flags = 0 if ev.get("case_sensitive") else re.I
        text = clean(answer)
        missing = [p for p in ev["patterns"] if not re.search(p, text, flags)]
        return Verdict(not missing, f"quality.reason.missing:{missing[0]}" if missing else "")
    if kind == "json":
        return check_json(answer, ev)
    if kind == "python_syntax":
        return check_python(answer, ev)
    return Verdict(False, f"quality.reason.unknown:{kind}")


class QualityRunner:
    def __init__(self, bench, suite: dict, repeats: int = 1):
        self.bench, self.suite = bench, suite
        self.repeats = max(1, int(repeats))
        cfg = suite.get("default_generation_config", {})
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self.temperature = float(cfg.get("temperature", 0))
        self.seed = int(cfg.get("seed", 42))

    @property
    def tests(self) -> list[dict]:
        return list(self.suite.get("tests", ()))

    def reset_cache(self, model: str) -> None:
        """Przerywnik: krótkie zapytanie z innym promptem (ten sam kontekst – model się nie przeładowuje),
        żeby próba nie trafiła na swój prompt w pamięci. Odpowiedź przerywnika nie jest oceniana."""
        self.bench.generate(model, FILLER, CONTEXT, FILLER_TOKENS, temperature=self.temperature, seed=self.seed)

    def run(self, model: str, report: Callable[[dict], None] = lambda e: None,
            stop: Optional[threading.Event] = None) -> list[QualityResult]:
        sid = self.suite["suite_info"]["id"]
        out = []
        for n, test in enumerate(self.tests, 1):
            if stop is not None and stop.is_set():
                raise Stopped()
            report({"event": "q_task", "model": model, "suite": sid, "n": n, "total": len(self.tests),
                    "test": test["id"], "title": test.get("title", test["id"]), "prompt": test["prompt"]})
            chunk = lambda text: report({"event": "chunk", "text": text})     # noqa: E731
            # Powtórzenia: to samo pytanie N razy (temperatura 0) – zgodność odpowiedzi pokazuje,
            # że wynik to nie szum losowania. Podgląd na żywo – tylko pierwszy przebieg.
            answers, verdicts, cached, tokens = [], [], [], 0
            for run in range(self.repeats):
                if run and stop is not None and stop.is_set():
                    raise Stopped()
                self.reset_cache(model)
                g = self.bench.generate(model, test["prompt"], CONTEXT, self.max_tokens,
                                        temperature=self.temperature, seed=self.seed,
                                        on_chunk=chunk if run == 0 else None)
                answers.append(g.text)
                verdicts.append(evaluate(g.text, test["evaluator"]))
                tokens = max(tokens, g.prompt_tokens)
                if g.prompt_cached is not None:
                    cached.append(g.prompt_cached)
                if run:
                    report({"event": "q_repeat", "model": model, "run": run + 1, "runs": self.repeats,
                            "same": g.text == answers[0]})
            passes = sum(v.passed for v in verdicts)
            verdict = verdicts[0]
            passed = passes * 2 > len(verdicts)            # większość przebiegów
            r = QualityResult(model, sid, test["id"], test.get("title", test["id"]), test.get("category", ""),
                              passed, "" if passed else (verdict.reason or next(
                                  (v.reason for v in verdicts if not v.passed), "")),
                              answers[0], test["prompt"], strict=verdict.strict,
                              subcategory=test.get("subcategory", ""), runs=len(answers),
                              identical=sum(a == answers[0] for a in answers), passes=passes,
                              prompt_tokens=tokens, cached=max(cached) if cached else None,
                              cache_warn=any(c * 2 > tokens for c in cached) if tokens else False)
            out.append(r)
            report({"event": "q_result", "model": model, "result": r})
        return out


def score(results: list[QualityResult]) -> dict:
    """{kolumna: (zaliczone, wszystkie)}: podkategorie polszczyzny, "json", "json_strict",
    "code", "code_strict" oraz "agree": (zadania z identycznymi odpowiedziami, wszystkie)."""
    out: dict = {}

    def add(key, ok):
        a, b = out.get(key, (0, 0))
        out[key] = (a + bool(ok), b + 1)
    for r in results:
        if r.skipped:
            continue
        if r.category == "format_adherence":
            add("json", r.passed)
            add("json_strict", r.passed and r.strict)
        elif r.category == "code_syntax":
            add("code", r.passed)
            add("code_strict", r.passed and r.strict)
        else:
            add(r.subcategory or r.category, r.passed)
        if r.runs > 1:
            add("agree", r.identical == r.runs)
    return out
