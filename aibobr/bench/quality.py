"""Testy Zacharskiego – jakość (etap 12c, SPEC §5c): polszczyzna, wiedza, JSON i kod.

Czy kwantyzacja nie psuje odpowiedzi. Każde zadanie raz (temperatura 0, seed 42, myślenie
wyłączone – odpowiedź za każdym razem ta sama), kontekst 2k. Ocena:
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
# Kategorie w tabeli „Jakość” (kolumny) – knowledge osobno: wiedza to nie kwantyzacja
CATEGORIES = ("language_integrity", "knowledge", "format_adherence", "code_syntax")

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
    return _embedded(text), False


def _at(data: Any, path: str) -> Any:
    for part in [p for p in path.split(".") if p]:
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def _lower(x: Any) -> Any:
    return x.lower() if isinstance(x, str) else [_lower(i) for i in x] if isinstance(x, list) else x


def check_json(answer: str, ev: dict) -> Verdict:
    try:
        data, strict = parse_json(answer)
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
def extract_code(answer: str) -> str:
    text = clean(answer)
    blocks = _FENCE.findall(text)
    return (max(blocks, key=len) if blocks else text).strip()


def check_python(answer: str, ev: dict) -> Verdict:
    """Tylko składnia i nazwy – ast.parse nie wykonuje kodu."""
    try:
        tree = ast.parse(extract_code(answer))
    except SyntaxError:
        return Verdict(False, "quality.reason.syntax")
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if ev.get("function") and ev["function"] not in funcs:
        return Verdict(False, f"quality.reason.no_function:{ev['function']}")
    if ev.get("class"):
        cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == ev["class"]), None)
        if cls is None:
            return Verdict(False, f"quality.reason.no_class:{ev['class']}")
        methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        missing = [m for m in ev.get("methods", ()) if m not in methods]
        if missing:
            return Verdict(False, f"quality.reason.no_methods:{', '.join(missing)}")
    return Verdict(True)


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
    def __init__(self, bench, suite: dict):
        self.bench, self.suite = bench, suite
        cfg = suite.get("default_generation_config", {})
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self.temperature = float(cfg.get("temperature", 0))
        self.seed = int(cfg.get("seed", 42))

    @property
    def tests(self) -> list[dict]:
        return list(self.suite.get("tests", ()))

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
            g = self.bench.generate(model, test["prompt"], CONTEXT, self.max_tokens,
                                    temperature=self.temperature, seed=self.seed, on_chunk=chunk)
            verdict = evaluate(g.text, test["evaluator"])
            r = QualityResult(model, sid, test["id"], test.get("title", test["id"]), test.get("category", ""),
                              verdict.passed, verdict.reason if not verdict.passed else "", g.text,
                              test["prompt"], strict=verdict.strict)
            out.append(r)
            report({"event": "q_result", "model": model, "result": r})
        return out


def score(results: list[QualityResult]) -> dict:
    """{kategoria: (zaliczone, wszystkie)} + "json_strict": (ściśle zaliczone, wszystkie JSON)."""
    out: dict = {}
    for r in results:
        if r.skipped:
            continue
        ok, total = out.get(r.category, (0, 0))
        out[r.category] = (ok + r.passed, total + 1)
        if r.strict is not None:
            s_ok, s_total = out.get("json_strict", (0, 0))
            out["json_strict"] = (s_ok + bool(r.passed and r.strict), s_total + 1)
    return out
