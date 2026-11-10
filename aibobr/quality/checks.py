"""Sprawdzanie odpowiedzi w testach jakości 0.10 – zwykły kod, ta sama odpowiedź = ten sam wynik.

Dwa niezależne pola (decyzja autora 10.11.2026):
  correct          – czy treść jest poprawna (None = nieocenione, np. kod bez piaskownicy),
  format_compliant – czy model wykonał polecenie co do formy („samo zdanie”, „tylko JSON”).
Różnice usuwane przez normalizację (spacje na brzegach, cudzysłów wokół, kropka na końcu, wielka
litera na początku, pogrubienie Markdown) nie naruszają formatu; każdy dodatkowy tekst go narusza.
Gdy z dodatkowego tekstu da się JEDNOZNACZNIE wydzielić jedną odpowiedź (jeden obiekt JSON, jedna
liczba, jeden blok kodu, jedna linia z odpowiedzią) – oceniamy ją; inaczej „nie można ocenić” i
niezaliczone. Niczego nie naprawiamy i nie zgadujemy intencji modelu.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Optional

CHECK_TYPES = ("text", "number", "fraction", "choice", "json", "python", "manual")

# Powody (klucze tłumaczeń quality.reason.*)
OK, FORMAT, WRONG, AMBIGUOUS, SYNTAX, EMPTY, NOT_RUN, MANUAL, TRUNCATED = (
    "ok", "format", "wrong", "ambiguous", "syntax", "empty", "not_run", "manual", "truncated")

QUOTES = "\"'„”“«»‚‘’`"
THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n?(.*?)```", re.S)
NUMBER = re.compile(r"[-−]?\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?(?!\d)|[-−]?\d+(?:[.,]\d+)?")
FRACTION = re.compile(r"(?<![\d/])(-?\d+)\s*/\s*(\d+)(?![\d/])")
CODE_START = re.compile(r"^(def |class |import |from |@)", re.M)


@dataclass
class Check:
    correct: Optional[bool]
    format_compliant: Optional[bool]
    reason: str
    value: Any = None          # odpowiedź wydzielona z tekstu (do pokazania i eksportu)

    @property
    def evaluated(self) -> bool:
        return self.correct is not None


def body(answer: str) -> str:
    """Bez bloku rozumowania <think> (część modeli go dokleja) i spacji na brzegach."""
    return THINK.sub("", answer or "").strip()


def unwrap(text: str) -> str:
    """Normalizacja otoczki: pogrubienie/kursywa Markdown i cudzysłów wokół całej odpowiedzi."""
    s = text.strip()
    changed = True
    while changed and s:
        changed = False
        for mark in ("**", "__", "*", "_"):
            if len(s) > 2 * len(mark) and s.startswith(mark) and s.endswith(mark):
                s, changed = s[len(mark):-len(mark)].strip(), True
        if len(s) > 1 and s[0] in QUOTES and s[-1] in QUOTES:
            s, changed = s[1:-1].strip(), True
    return s


def norm_text(text: str) -> str:
    """Porównanie tekstu: bez kropki na końcu, wielka litera na początku obojętna, spacje
    zwinięte. Znaki diakrytyczne, litery w środku i przecinki zostają – to one są oceniane."""
    s = re.sub(r"\s+", " ", unwrap(text))
    if s.endswith("."):
        s = s[:-1].rstrip()
    return s[:1].lower() + s[1:]


def _lines(text: str) -> list[str]:
    return [line for line in (part.strip() for part in text.splitlines()) if line]


# ------------------------------------------------------------------ rodzaje sprawdzeń
def check_text(answer: str, expected: list[str], sentence: Optional[list[str]] = None) -> Check:
    """sentence – w zadaniach „uzupełnij lukę”: całe zdanie z polecenia z poprawnie wypełnioną luką
    (decyzja autora 10.11.2026) – zaliczone, ale z niezgodnym formatem (prosiliśmy o samo słowo)."""
    s = body(answer)
    if not s:
        return Check(False, None, EMPTY)
    want = {norm_text(e) for e in expected}
    if norm_text(s) in want:
        return Check(True, True, OK, unwrap(s))
    if sentence and norm_text(s) in {norm_text(e) for e in sentence}:
        return Check(True, False, FORMAT, unwrap(s))
    fenced = FENCE.fullmatch(s)
    if fenced and norm_text(fenced.group(2)) in want:            # sama odpowiedź w bloku ```
        return Check(True, False, FORMAT, unwrap(fenced.group(2)))
    lines = _lines(s)
    hits = [line for line in lines if norm_text(line) in want]
    if len(lines) > 1 and len(hits) == 1:                        # odpowiedź w osobnej linii + komentarz
        return Check(True, False, FORMAT, unwrap(hits[0]))
    if len(lines) > 1 and not hits:
        return Check(False, None, AMBIGUOUS if len(lines) > 2 else WRONG, s)
    return Check(False, None, WRONG, unwrap(s))


def final_line(text: str) -> str:
    """Ostatnia linia z treścią – bez linii z samymi znacznikami (``` , \\], **, ---)."""
    lines = [line for line in _lines(text) if re.search(r"\w", line)]
    return lines[-1] if lines else ""


def final_answer(text: str, pattern: re.Pattern) -> Optional[str]:
    """Wynik końcowy z rozwiązania z obliczeniami (decyzja autora 10.11.2026, propozycja B) – tylko
    dwa ściśle określone wzorce: ostatnia linia ma dokładnie jedną wartość albo po ostatnim „=”
    w ostatniej linii stoi dokładnie jedna wartość. Inaczej None (nie można ocenić)."""
    last = final_line(text)
    found = pattern.findall(last)
    if len(found) == 1:
        return found[0]
    if "=" in last:
        tail = pattern.findall(last.rsplit("=", 1)[1])
        if len(tail) == 1:
            return tail[0]
    return None


def _to_number(token: str) -> float:
    return float(token.replace(" ", "").replace(" ", "").replace("−", "-").replace(",", "."))


def check_number(answer: str, expected: float, tolerance: float = 1e-9) -> Check:
    s = unwrap(body(answer))
    if not s:
        return Check(False, None, EMPTY)
    tokens = NUMBER.findall(s)
    if len(tokens) > 1:
        last = final_answer(s, NUMBER)
        if last is None:
            return Check(False, None, AMBIGUOUS, s)
        tokens = [last]
    if not tokens:
        return Check(False, None, WRONG, s)
    value = _to_number(tokens[0])
    ok = abs(value - float(expected)) <= max(tolerance, 1e-9 * max(1.0, abs(float(expected))))
    bare = norm_text(s) == norm_text(tokens[0])
    if not ok:
        return Check(False, None, WRONG, value)
    return Check(True, bare, OK if bare else FORMAT, value)


def check_fraction(answer: str, expected: str) -> Check:
    s = unwrap(body(answer))
    if not s:
        return Check(False, None, EMPTY)
    found = FRACTION.findall(s)
    if len(found) > 1:
        last = final_answer(s, FRACTION)
        if last is None:
            return Check(False, None, AMBIGUOUS, s)
        found = [last]
    if not found:
        return Check(False, None, WRONG, s)
    num, den = found[0]
    want_num, want_den = expected.split("/")
    ok = (int(num), int(den)) == (int(want_num), int(want_den))   # „nieskracalny” – 38/24 to błąd
    bare = re.sub(r"\s+", "", norm_text(s)) == f"{num}/{den}"
    if not ok:
        return Check(False, None, WRONG, f"{num}/{den}")
    return Check(True, bare, OK if bare else FORMAT, f"{num}/{den}")


def check_choice(answer: str, expected: str, options: list[str]) -> Check:
    """Odpowiedź z zamkniętej listy (tak/nie, dni tygodnia): oceniamy, gdy w tekście pada
    dokładnie jedna z możliwych odpowiedzi."""
    s = unwrap(body(answer))
    if not s:
        return Check(False, None, EMPTY)
    words = {w.casefold() for w in re.findall(r"\w+", s)}
    found = [o for o in options if o.casefold() in words]
    if len(found) != 1:
        return Check(False, None, AMBIGUOUS if found else WRONG, s)
    ok = found[0].casefold() == expected.casefold()
    bare = norm_text(s).casefold() == found[0].casefold()
    if not ok:
        return Check(False, None, WRONG, found[0])
    return Check(True, bare, OK if bare else FORMAT, found[0])


def same(a: Any, b: Any) -> bool:
    """Równość jak w JSON: true ≠ 1, 28 = 28.0, kolejność w liście ma znaczenie."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def extract_json(text: str) -> tuple[str, Any, bool]:
    """(stan, wartość, czy bez dodatków). stan: ok / ambiguous / syntax / empty."""
    s = body(text)
    if not s:
        return EMPTY, None, False
    try:
        return OK, json.loads(s), True
    except ValueError:
        pass
    fences = FENCE.findall(s)
    if len(fences) > 1:
        return AMBIGUOUS, None, False
    candidate = fences[0][1] if fences else s
    try:
        return OK, json.loads(candidate), False                     # jednoznaczna otoczka ```json
    except ValueError:
        pass
    decoder, found, i = json.JSONDecoder(), [], 0
    while i < len(candidate):
        if candidate[i] in "{[":
            try:
                value, end = decoder.raw_decode(candidate, i)
                found.append(value)
                i = end
                continue
            except ValueError:
                pass
        i += 1
    if len(found) == 1:
        return OK, found[0], False
    return (AMBIGUOUS, None, False) if found else (SYNTAX, None, False)


def check_json(answer: str, expected: Any) -> Check:
    state, value, bare = extract_json(answer)
    if state != OK:
        return Check(False, None, state)
    if not same(value, expected):
        return Check(False, None, WRONG, value)
    return Check(True, bare, OK if bare else FORMAT, value)


def extract_code(answer: str) -> tuple[str, Optional[str], bool]:
    """(stan, kod, czy bez dodatków): jeden blok ``` albo cały tekst; przed kodem może stać
    jednoznaczny wstęp („Code:”) – wtedy kod od pierwszego def/class/import."""
    s = body(answer)
    if not s:
        return EMPTY, None, False
    fences = FENCE.findall(s)
    if len(fences) > 1:
        return AMBIGUOUS, None, False
    if fences:
        code = fences[0][1]
        bare = FENCE.fullmatch(s) is not None
    else:
        code, bare = s, True
        try:
            compile(code, "<model>", "exec")
        except SyntaxError:
            start = CODE_START.search(s)
            if not start or start.start() == 0:
                return SYNTAX, None, False
            code, bare = s[start.start():], False
    try:
        compile(code, "<model>", "exec")
    except SyntaxError:
        return SYNTAX, None, False
    return OK, code, bare and not fences


RunPython = Callable[[str, str, list], tuple[Optional[bool], str]]


def check_python(answer: str, spec: dict, run: Optional[RunPython] = None) -> Check:
    """Kod modelu jest niezaufany: uruchamia go tylko `run` (piaskownica). Bez niej – nieocenione."""
    state, code, bare = extract_code(answer)
    if state != OK:
        return Check(False, None, state)
    if run is None:
        return Check(None, bare, NOT_RUN, code)
    passed, detail = run(code, spec["function"], spec["tests"])
    if passed is None:
        return Check(None, bare, NOT_RUN, code)
    if not passed:
        return Check(False, None, WRONG, detail or code)
    return Check(True, bare, OK if bare else FORMAT, code)


def check(answer: str, spec: dict, lang: str = "pl", run_python: Optional[RunPython] = None) -> Check:
    kind = spec["type"]

    def pick(value):
        return value.get(lang) if isinstance(value, dict) and lang in value else value
    if kind == "text":
        return check_text(answer, pick(spec["expected"]), pick(spec.get("sentence")))
    if kind == "number":
        return check_number(answer, spec["expected"], spec.get("tolerance", 1e-9))
    if kind == "fraction":
        return check_fraction(answer, spec["expected"])
    if kind == "choice":
        return check_choice(answer, pick(spec["expected"]), pick(spec["options"]))
    if kind == "json":
        return check_json(answer, spec["expected"])
    if kind == "python":
        return check_python(answer, spec, run_python)
    if kind == "manual":
        return Check(None, None, MANUAL, body(answer))
    raise ValueError(f"nieznany rodzaj sprawdzenia: {kind}")


def expected_text(spec: dict, lang: str = "pl") -> str:
    """Oczekiwana odpowiedź do pokazania obok odpowiedzi modelu."""
    kind, value = spec["type"], spec.get("expected")
    if isinstance(value, dict) and lang in value:
        value = value[lang]
    if kind == "text":
        return " | ".join(value)
    if kind == "json":
        return json.dumps(value, ensure_ascii=False)
    if kind == "python":
        return "; ".join(f"{spec['function']}({', '.join(map(repr, t['args']))}) == {t['expected']!r}"
                         for t in spec["tests"])
    if kind == "number" and isinstance(value, float) and Fraction(str(value)).denominator == 1:
        return str(int(value))
    return str(value)
