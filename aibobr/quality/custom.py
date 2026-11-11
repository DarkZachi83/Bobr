"""Własne testy jakości – zapis, ścisła walidacja, import/eksport i podgląd oceny (logika edytora testów).

Zasady (docs/SPEC.md, „Edytor testów”):
- własne testy leżą w <katalog ustawień>/quality/<id>.json, w tym samym formacie co standardowe;
  testy standardowe (data/quality) są tylko do odczytu – można zrobić z nich kopię;
- plik testu nigdy nie zawiera kodu: zadanie „python” to nazwa funkcji oraz argumenty i wyniki jako
  wartości JSON; uruchamiany jest wyłącznie kod modelu, w piaskownicy – dlatego cudzy test jest bezpieczny;
- zapis testu, który był już użyty w pomiarze, podnosi jego wersję (historia ma własną kopię zadań);
- błąd w teście blokuje zapis i wskazuje zadanie i pole.
"""
from __future__ import annotations

import copy
import json
import keyword
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from aibobr.quality import checks, suites
from aibobr.quality.checks import CHECK_TYPES

LANGUAGES = ("pl", "en")
MAX_TASKS = 100
MIN_TOKENS, MAX_TOKENS = 16, 4096
MAX_FILE_BYTES = 1_000_000
ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
FRACTION_RE = re.compile(r"^-?\d+/\d+$")
FIRST_VERSION = "1"

# Dozwolone pola sprawdzenia – nic poza nimi (żadnych miejsc na kod ani „ustawienia” harnessu).
CHECK_FIELDS = {
    "text": ({"expected"}, {"sentence"}),
    "number": ({"expected"}, {"tolerance"}),
    "fraction": ({"expected"}, set()),
    "choice": ({"expected", "options"}, set()),
    "json": ({"expected"}, set()),
    "python": ({"function", "tests"}, set()),
    "manual": (set(), set()),
}


@dataclass(frozen=True)
class Problem:
    """Błąd w teście: zadanie (None = cały test), pole i kod komunikatu (klucz q.err.<code> w i18n)."""
    task: Optional[str]
    field: str
    code: str

    def __str__(self) -> str:
        return f"{self.task or 'test'}: {self.field}: {self.code}"


def folder() -> Path:
    from aibobr.app.config import user_config_dir
    return user_config_dir() / "quality"


# --- walidacja -------------------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _per_lang(value: Any, langs: list[str], ok: Callable[[Any], bool]) -> bool:
    """Wartość wspólna dla wszystkich języków albo słownik {język: wartość} z każdym językiem testu."""
    if isinstance(value, dict):
        return all(lang in value and ok(value[lang]) for lang in langs)
    return ok(value)


def _texts(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) and v.strip() for v in value)


def _check_problems(tid: str, spec: dict, langs: list[str]) -> list[Problem]:
    kind = spec.get("type")
    if kind not in CHECK_TYPES:
        return [Problem(tid, "check.type", "check_type")]
    required, optional = CHECK_FIELDS[kind]
    out = [Problem(tid, f"check.{k}", "missing") for k in sorted(required) if k not in spec]
    out += [Problem(tid, f"check.{k}", "unknown_field") for k in sorted(set(spec) - required - optional - {"type"})]
    if out:
        return out
    if kind == "text":
        if not _per_lang(spec["expected"], langs, _texts):
            out.append(Problem(tid, "check.expected", "text_expected"))
        if "sentence" in spec and not _per_lang(spec["sentence"], langs, _texts):
            out.append(Problem(tid, "check.sentence", "text_expected"))
    elif kind == "number":
        if not _is_number(spec["expected"]):
            out.append(Problem(tid, "check.expected", "number"))
        if "tolerance" in spec and not (_is_number(spec["tolerance"]) and spec["tolerance"] >= 0):
            out.append(Problem(tid, "check.tolerance", "tolerance"))
    elif kind == "fraction":
        value = spec["expected"]
        if not (isinstance(value, str) and FRACTION_RE.match(value.replace(" ", ""))
                and int(value.replace(" ", "").split("/")[1]) > 0):
            out.append(Problem(tid, "check.expected", "fraction"))
    elif kind == "choice":
        def options_ok(v):
            return _texts(v) and len(v) >= 2 and len({checks.norm_text(o) for o in v}) == len(v)
        if not _per_lang(spec["options"], langs, options_ok):
            out.append(Problem(tid, "check.options", "choice_options"))
        else:
            for lang in langs:
                exp = spec["expected"].get(lang) if isinstance(spec["expected"], dict) else spec["expected"]
                opts = spec["options"].get(lang) if isinstance(spec["options"], dict) else spec["options"]
                if not isinstance(exp, str) or checks.norm_text(exp) not in {checks.norm_text(o) for o in opts}:
                    out.append(Problem(tid, "check.expected", "choice_expected"))
                    break
    elif kind == "python":
        name = spec["function"]
        if not (isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)):
            out.append(Problem(tid, "check.function", "python_function"))
        tests = spec["tests"]
        if not (isinstance(tests, list) and tests and all(
                isinstance(t, dict) and set(t) == {"args", "expected"} and isinstance(t["args"], list)
                for t in tests)):
            out.append(Problem(tid, "check.tests", "python_tests"))
    # json: dowolna wartość JSON – plik już się sparsował, więc jest poprawna.
    return out


def problems(data: Any) -> list[Problem]:
    """Wszystkie błędy własnego testu naraz – edytor pokazuje je przy zadaniach i polach."""
    if not isinstance(data, dict):
        return [Problem(None, "file", "not_object")]
    info = data.get("suite_info")
    if not isinstance(info, dict):
        return [Problem(None, "suite_info", "missing")]
    out: list[Problem] = []
    sid = info.get("id")
    if not isinstance(sid, str) or not ID_RE.match(sid):
        out.append(Problem(None, "suite_info.id", "bad_id"))
    elif sid in suites.STANDARD:
        out.append(Problem(None, "suite_info.id", "standard_id"))
    if info.get("kind") != "custom":
        out.append(Problem(None, "suite_info.kind", "kind"))
    if not isinstance(info.get("version"), str) or not info["version"].strip():
        out.append(Problem(None, "suite_info.version", "missing"))
    langs = info.get("languages")
    if not (isinstance(langs, list) and langs and len(set(langs)) == len(langs)
            and all(lang in LANGUAGES for lang in langs)):
        out.append(Problem(None, "suite_info.languages", "languages"))
        langs = []
    name = info.get("name")
    if not (isinstance(name, dict) and langs and all(isinstance(name.get(l), str) and name[l].strip()
                                                       for l in langs)):
        out.append(Problem(None, "suite_info.name", "name"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        out.append(Problem(None, "tasks", "no_tasks"))
        return out
    if len(tasks) > MAX_TASKS:
        out.append(Problem(None, "tasks", "too_many_tasks"))
    seen: set[str] = set()
    for n, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            out.append(Problem(f"#{n}", "task", "not_object"))
            continue
        tid = task.get("id")
        if not isinstance(tid, str) or not tid.strip():
            tid = f"#{n}"
            out.append(Problem(tid, "id", "missing"))
        elif tid in seen:
            out.append(Problem(tid, "id", "duplicate_id"))
        seen.add(tid)
        if not isinstance(task.get("category"), str) or not task["category"].strip():
            out.append(Problem(tid, "category", "missing"))
        if task.get("level") not in suites.LEVELS:
            out.append(Problem(tid, "level", "level"))
        mt = task.get("max_tokens")
        if not (isinstance(mt, int) and not isinstance(mt, bool) and MIN_TOKENS <= mt <= MAX_TOKENS):
            out.append(Problem(tid, "max_tokens", "max_tokens"))
        prompt = task.get("prompt")
        for lang in langs:
            if not (isinstance(prompt, dict) and isinstance(prompt.get(lang), str) and prompt[lang].strip()):
                out.append(Problem(tid, f"prompt.{lang}", "prompt"))
        spec = task.get("check")
        if not isinstance(spec, dict):
            out.append(Problem(tid, "check", "missing"))
        else:
            out += _check_problems(tid, spec, langs)
    return out


def level_counts(data: dict) -> dict[str, int]:
    """Rozkład poziomów – pokazywany w edytorze, nie wymagany (5/5/5 dotyczy testów standardowych)."""
    counts = {level: 0 for level in suites.LEVELS}
    for task in data.get("tasks") or []:
        if task.get("level") in counts:
            counts[task["level"]] += 1
    return counts


# --- tworzenie -------------------------------------------------------------------------------------

def slug(name: str) -> str:
    table = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    s = re.sub(r"[^a-z0-9]+", "_", name.translate(table).lower()).strip("_")
    return (s or "test")[:40]


def unique_id(base: str, taken: Iterable[str]) -> str:
    taken = set(taken) | set(suites.STANDARD)
    base = slug(base)
    if base not in taken:
        return base
    n = 2
    while f"{base[:36]}_{n}" in taken:
        n += 1
    return f"{base[:36]}_{n}"


def new_suite(suite_id: str, name: str, languages: list[str]) -> dict:
    """Pusty test (bez zadań – zapisać da się dopiero z co najmniej jednym)."""
    return {"suite_info": {"id": suite_id, "version": FIRST_VERSION, "kind": "custom",
                           "name": {lang: name for lang in languages}, "languages": list(languages)},
            "tasks": []}


def next_task_id(data: dict) -> str:
    used = {t.get("id") for t in data.get("tasks") or []}
    n = len(used) + 1
    while f"T{n}" in used:
        n += 1
    return f"T{n}"


def new_task(data: dict, kind: str = "text") -> dict:
    langs = data["suite_info"]["languages"]
    blank = {"text": {"expected": []}, "number": {"expected": 0}, "fraction": {"expected": "1/2"},
             "choice": {"expected": "", "options": []}, "json": {"expected": {}},
             "python": {"function": "", "tests": []}, "manual": {}}[kind]
    return {"id": next_task_id(data), "category": "", "level": "easy", "max_tokens": 256,
            "prompt": {lang: "" for lang in langs}, "check": {"type": kind, **copy.deepcopy(blank)}}


def copy_of_standard(suite_id: str, new_id: str, name: str) -> dict:
    """„Utwórz kopię jako mój test” – testy standardowe zostają nietknięte."""
    data = copy.deepcopy(suites.load(suite_id))
    info = data["suite_info"]
    data["suite_info"] = {"id": new_id, "version": FIRST_VERSION, "kind": "custom",
                          "name": {lang: name for lang in info["languages"]},
                          "languages": list(info["languages"]), "based_on": f"{suite_id} v{info['version']}"}
    return data


def bump(version: str) -> str:
    """'3' → '4', '1.1' → '1.2'; cokolwiek innego → '<wersja>.1'."""
    parts = version.strip().split(".")
    if parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    return f"{version.strip()}.1"


# --- zapis, import, eksport -----------------------------------------------------------------------

class CustomSuiteError(ValueError):
    def __init__(self, found: list[Problem]):
        super().__init__("; ".join(str(p) for p in found[:5]))
        self.problems = found


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_file(path: Path) -> dict:
    """Wczytanie pliku testu (własnego albo importowanego) ze sprawdzeniem – bez naprawiania."""
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise CustomSuiteError([Problem(None, "file", "too_large")])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CustomSuiteError([Problem(None, "file", "not_json")]) from None
    found = problems(data)
    if found:
        raise CustomSuiteError(found)
    return data


@dataclass
class Entry:
    """Pozycja na liście testów w edytorze."""
    suite_id: str
    path: Path
    data: Optional[dict]
    error: Optional[CustomSuiteError] = None


class CustomSuites:
    def __init__(self, base: Optional[Path] = None, used: Optional[Callable[[str, str], bool]] = None):
        """`used(id, wersja)` – czy ta wersja testu ma już wyniki w bazie (BenchDb.quality_suite_used)."""
        self.base = Path(base) if base is not None else folder()
        self.used = used or (lambda suite_id, version: False)

    def path(self, suite_id: str) -> Path:
        if not ID_RE.match(suite_id or ""):
            raise CustomSuiteError([Problem(None, "suite_info.id", "bad_id")])
        return self.base / f"{suite_id}.json"

    def entries(self) -> list[Entry]:
        """Wszystkie pliki z folderu – błędne też, z opisem błędu (nie znikają po cichu)."""
        out = []
        for path in sorted(self.base.glob("*.json")) if self.base.is_dir() else []:
            try:
                out.append(Entry(path.stem, path, read_file(path)))
            except CustomSuiteError as e:
                out.append(Entry(path.stem, path, None, e))
            except OSError:
                out.append(Entry(path.stem, path, None, CustomSuiteError([Problem(None, "file", "unreadable")])))
        return out

    def ids(self) -> list[str]:
        return [e.suite_id for e in self.entries()]

    def valid(self) -> list[dict]:
        """Testy gotowe do uruchomienia (nazwa pliku musi zgadzać się z id)."""
        return [e.data for e in self.entries() if e.data and e.data["suite_info"]["id"] == e.suite_id]

    def exists(self, suite_id: str) -> bool:
        return self.path(suite_id).exists()

    def load(self, suite_id: str) -> dict:
        return read_file(self.path(suite_id))

    def save(self, data: dict) -> dict:
        """Zapis po pełnym sprawdzeniu; użyta już wersja dostaje nowy numer. Zwraca zapisane dane."""
        found = problems(data)
        if found:
            raise CustomSuiteError(found)
        data = copy.deepcopy(data)
        info = data["suite_info"]
        target = self.path(info["id"])
        if target.exists():
            try:
                old = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = None
            if old == data:
                return data
            old_version = (old or {}).get("suite_info", {}).get("version")
            if old_version and info["version"] == old_version and self.used(info["id"], old_version):
                info["version"] = bump(old_version)
        _write_json(target, data)
        return data

    def delete(self, suite_id: str) -> None:
        """Usuwa tylko plik testu; wyniki w historii zostają (mają własną kopię zadań)."""
        self.path(suite_id).unlink(missing_ok=True)

    def export(self, suite_id: str, target: Path) -> Path:
        data = self.load(suite_id)
        target = Path(target)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        _write_json(target, data)
        return target

    def read_import(self, source: Path) -> tuple[dict, bool]:
        """Sprawdzony test z pliku i informacja, czy taki id już jest (edytor pyta: zastąpić czy kopia)."""
        data = read_file(source)
        return data, self.exists(data["suite_info"]["id"])

    def import_as_copy(self, data: dict) -> dict:
        data = copy.deepcopy(data)
        info = data["suite_info"]
        info["id"] = unique_id(info["id"], self.ids())
        info["version"] = FIRST_VERSION
        return self.save(data)

    def import_replace(self, data: dict) -> dict:
        """Zastąpienie: treść z pliku, ale numer wersji nie może się cofnąć ani powtórzyć użytej."""
        data = copy.deepcopy(data)
        if self.exists(data["suite_info"]["id"]):
            try:
                old = self.load(data["suite_info"]["id"])["suite_info"]["version"]
            except CustomSuiteError:
                old = None
            if old and old != data["suite_info"]["version"]:
                data["suite_info"]["version"] = old  # save() podniesie, jeśli była użyta
        return self.save(data)


# --- podgląd oceny ---------------------------------------------------------------------------------

def preview(spec: dict, answer: str, lang: str = "pl", sandbox=None) -> checks.Check:
    """Werdykt dla wklejonej odpowiedzi – ten sam kod oceny co w teście, bez modelu.

    Kod Pythona z odpowiedzi jest traktowany jak kod modelu: tylko w gotowej piaskownicy, inaczej nieoceniony.
    """
    run = sandbox.run_tests if sandbox is not None and sandbox.status().ready else None
    return checks.check(answer, spec, lang, run_python=run)
