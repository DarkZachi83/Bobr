"""Okno „Edytor testów” (0.11, decyzje autora 11.11.2026) – osobne okno, bo tworzenie testów jest opcjonalne.

Lewa strona: testy standardowe (tylko do odczytu, z kopią „jako mój test”) i własne (także pliki z błędem –
widoczne z opisem). Prawa: nazwa i języki testu, lista zadań, formularz zadania z polami zależnymi od rodzaju
sprawdzenia, lista błędów (każdy wskazuje zadanie i pole) i podgląd oceny wklejonej odpowiedzi bez modelu.
Logika – aibobr/quality/custom.py; tu tylko formularz ↔ słownik zadania. Zapis zablokowany w trakcie pomiaru.
"""
from __future__ import annotations

import copy
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable, Optional

from aibobr.gui.theme import COLORS
from aibobr.i18n import current_language, has_key, t
from aibobr.quality import custom, suites
from aibobr.quality.checks import CHECK_TYPES
from aibobr.quality.custom import CustomSuiteError, CustomSuites, Problem

MARK = {True: "✓", False: "✗", None: "–"}
PER_LANG = {"text": ("expected", "sentence"), "choice": ("expected", "options")}


def suite_name(data: dict) -> str:
    name = data["suite_info"].get("name") or {}
    return name.get(current_language()) or next(iter(name.values()), data["suite_info"]["id"])


def cat_name(category: str) -> str:
    return t(f"q.cat.{category}") if has_key(f"q.cat.{category}") else category


def field_label(field: str) -> str:
    if field.startswith("prompt."):
        return f"{t('q.ed.field.prompt')} ({field.split('.', 1)[1].upper()})"
    key = f"q.ed.field.{field}"
    return t(key) if has_key(key) else field


def problem_text(p: Problem) -> str:
    where = t("q.ed.whole_test") if p.task is None else p.task
    message = t(f"q.err.{p.code}") if has_key(f"q.err.{p.code}") else p.code
    return f"{where} · {field_label(p.field)} – {message}"


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_number(text: str):
    """'0,5' → 0.5, '204' → 204; nieliczba zostaje tekstem – walidacja pokaże błąd przy polu."""
    s = text.strip().replace(" ", "").replace(",", ".").replace("−", "-")
    try:
        return int(s) if s.lstrip("-").isdigit() else float(s)
    except ValueError:
        return text.strip()


def parse_python_tests(text: str) -> Optional[list]:
    """Wiersz „[1, 2] -> 3”: argumenty i wynik jako JSON. None = któregoś wiersza nie da się odczytać."""
    out = []
    for line in lines(text):
        pos = line.find("->")
        while pos != -1:
            try:
                args, expected = json.loads(line[:pos]), json.loads(line[pos + 2:])
                break
            except ValueError:
                pos = line.find("->", pos + 2)
        else:
            return None
        out.append({"args": args, "expected": expected})
    return out


def format_python_tests(tests) -> str:
    if not isinstance(tests, list):
        return ""
    return "\n".join(f"{json.dumps(c.get('args'), ensure_ascii=False)} -> "
                     f"{json.dumps(c.get('expected'), ensure_ascii=False)}" for c in tests if isinstance(c, dict))


def per_lang(value, lang: str):
    return value.get(lang) if isinstance(value, dict) else value


def lang_label(code: str) -> str:
    """„koreański (KO)”; kod spoza listy – sam kod."""
    key = f"q.lang.{code}"
    return f"{t(key)} ({code.upper()})" if has_key(key) else code.upper()


def empty_value(kind: str, key: str):
    return "" if (kind, key) == ("choice", "expected") else []


def put_lang(spec: dict, key: str, value, langs: list[str], lang: str) -> None:
    """Wartość jednego języka; pozostałe zostają (formularz pokazuje jeden język naraz – wariant B)."""
    current = spec.get(key)
    values = {l: per_lang(current, l) if current is not None else None for l in langs}
    values = {l: empty_value(spec["type"], key) if v is None else v for l, v in values.items()}
    values[lang] = value
    spec[key] = values[langs[0]] if len(langs) == 1 else values


class TestEditor(tk.Toplevel):
    def __init__(self, master, store: CustomSuites, sandbox=None, busy: Callable[[], bool] = lambda: False,
                 on_change: Callable[[], None] = lambda: None):
        super().__init__(master)
        self.store, self.sandbox, self.busy, self.on_change = store, sandbox, busy, on_change
        self.title(t("q.ed.title"))
        self.geometry("1180x840")
        self.minsize(900, 640)
        self.protocol("WM_DELETE_WINDOW", self.close)
        # zaczepy na okna dialogowe – testy podmieniają
        self.ask_name = lambda title, initial: simpledialog.askstring(title, t("q.ed.name"), initialvalue=initial,
                                                                      parent=self)
        self.ask_save = lambda: messagebox.askyesnocancel(t("q.ed.title"), t("q.ed.unsaved"), parent=self)
        self.ask_replace = lambda name: messagebox.askyesnocancel(t("q.ed.import"),
                                                                  t("q.ed.import_conflict", name=name), parent=self)
        self.confirm = lambda text: messagebox.askokcancel(t("q.ed.title"), text, parent=self)
        self.inform = lambda text: messagebox.showinfo(t("q.ed.title"), text, parent=self)

        self.data: Optional[dict] = None
        self.clean: Optional[str] = None          # zapisany stan (do pytania o niezapisane zmiany)
        self.readonly = True
        self.saved_id: Optional[str] = None       # id pliku, z którego wczytano test (None = jeszcze niezapisany)
        self.index: Optional[int] = None          # wybrane zadanie
        self.raw: dict[tuple[str, str], tuple[str, str]] = {}   # (zadanie, pole) → (tekst, kod błędu)
        self.inputs: dict[str, object] = {}
        self._shown_lang = ""                     # język widoczny w formularzu (przełącznik – wariant B)
        self.list_rows: dict[str, tuple[str, str]] = {}
        self._loading = False

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        pane = ttk.Panedwindow(body, orient="horizontal")
        pane.pack(fill="both", expand=True)
        pane.add(self._build_list(pane), weight=0)
        pane.add(self._build_main(pane), weight=4)
        self.refresh_list()
        self._show_suite(None)

    # ------------------------------------------------------------------ budowa
    def _build_list(self, parent) -> ttk.Frame:
        box = ttk.Frame(parent, padding=(0, 0, 8, 0))
        ttk.Label(box, text=t("q.ed.tests"), font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        frame = ttk.Frame(box)
        frame.pack(fill="both", expand=True, pady=(4, 6))
        self.tests = ttk.Treeview(frame, show="tree", selectmode="browse", height=14)
        self.tests.column("#0", width=250)
        self.tests.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tests.yview)
        self.tests.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tests.bind("<<TreeviewSelect>>", lambda e: self._pick_suite())
        self.list_buttons = {}
        for key, command in (("new", self.new_suite), ("copy", self.copy_suite), ("delete", self.delete_suite),
                             ("import", self.import_suite), ("export", self.export_suite)):
            b = ttk.Button(box, text=t(f"q.ed.{key}"), command=command)
            b.pack(fill="x", pady=1)
            self.list_buttons[key] = b
        return box

    def _build_main(self, parent) -> ttk.Frame:
        main = ttk.Frame(parent)
        head = ttk.Frame(main)
        head.pack(fill="x")
        ttk.Label(head, text=t("q.ed.name")).pack(side="left")
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(head, textvariable=self.name_var, width=30)
        self.name_entry.pack(side="left", padx=(6, 14))
        ttk.Label(head, text=t("q.ed.language")).pack(side="left")
        self.lang_codes = sorted(custom.COMMON_LANGUAGES, key=lambda c: lang_label(c).casefold())
        self.main_lang = ttk.Combobox(head, values=[lang_label(c) for c in self.lang_codes], state="readonly",
                                      width=20)
        self.main_lang.pack(side="left", padx=(6, 10))
        self.main_lang.bind("<<ComboboxSelected>>", lambda e: self._change_main_lang())
        self.english_var = tk.BooleanVar()
        self.english_check = ttk.Checkbutton(head, text=t("q.ed.english"), variable=self.english_var,
                                             command=self._toggle_english)
        self.english_check.pack(side="left")
        self.form_lang = tk.StringVar()
        self.info = ttk.Label(main, text="", foreground=COLORS["muted"])
        self.info.pack(anchor="w", pady=(4, 6))

        middle = ttk.Panedwindow(main, orient="horizontal")
        middle.pack(fill="both", expand=True)
        tasks = ttk.Frame(middle, padding=(0, 0, 8, 0))
        cols = ("id", "level", "category", "type")
        self.task_tree = ttk.Treeview(tasks, columns=cols, show="headings", selectmode="browse", height=14)
        for col, width in zip(cols, (50, 70, 120, 115)):
            self.task_tree.heading(col, text=t(f"q.ed.col.{col}"), anchor="w")
            self.task_tree.column(col, width=width, anchor="w", stretch=col == "category")
        self.task_tree.pack(fill="both", expand=True)
        self.task_tree.bind("<<TreeviewSelect>>", lambda e: self._pick_task())
        row = ttk.Frame(tasks)
        row.pack(fill="x", pady=(4, 0))
        self.task_buttons = {}
        for key, command in (("add", self.add_task), ("remove", self.remove_task),
                             ("up", lambda: self.move_task(-1)), ("down", lambda: self.move_task(1))):
            b = ttk.Button(row, text=t(f"q.ed.task.{key}"), command=command,
                           width=None if key in ("add", "remove") else 3)
            b.pack(side="left", padx=(0, 3))
            self.task_buttons[key] = b
        middle.add(tasks, weight=2)
        form_box = ttk.Frame(middle)
        canvas = tk.Canvas(form_box, highlightthickness=0, borderwidth=0, background=COLORS["bg"])
        bar = ttk.Scrollbar(form_box, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.form = ttk.Frame(canvas, padding=(4, 0, 8, 0))
        window = canvas.create_window(0, 0, window=self.form, anchor="nw")
        self.form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        middle.add(form_box, weight=3)

        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(8, 0))
        bottom.columnconfigure(0, weight=1, uniform="b")
        bottom.columnconfigure(1, weight=1, uniform="b")
        probs = ttk.Frame(bottom)
        probs.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(probs, text=t("q.ed.problems"), font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.problem_list = tk.Listbox(probs, height=6, activestyle="none")
        self.problem_list.pack(fill="both", expand=True)
        self.problem_list.bind("<<ListboxSelect>>", lambda e: self._goto_problem())
        self.problems: list[Problem] = []
        prev = ttk.Frame(bottom)
        prev.grid(row=0, column=1, sticky="nsew")
        line = ttk.Frame(prev)
        line.pack(fill="x")
        ttk.Label(line, text=t("q.ed.preview"), font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.preview_lang = ttk.Combobox(line, values=[], state="readonly", width=4)
        self.preview_lang.pack(side="right")
        self.preview_answer = self._text_widget(prev, 4)
        self.preview_answer.pack(fill="both", expand=True, pady=(2, 2))
        line = ttk.Frame(prev)
        line.pack(fill="x")
        self.preview_button = ttk.Button(line, text=t("q.ed.preview_run"), command=self.run_preview)
        self.preview_button.pack(side="left")
        self.preview_result = ttk.Label(line, text="", wraplength=380, justify="left")
        self.preview_result.pack(side="left", padx=8)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text=t("q.ed.close"), command=self.close).pack(side="right")
        self.save_button = ttk.Button(buttons, text=t("q.ed.save"), style="Accent.TButton", command=self.save)
        self.save_button.pack(side="right", padx=6)
        self.status = ttk.Label(buttons, text="")
        self.status.pack(side="left")
        return main

    # ------------------------------------------------------------------ lista testów
    def refresh_list(self, select: Optional[tuple[str, str]] = None) -> None:
        self.tests.delete(*self.tests.get_children())
        self.list_rows = {}
        for sid in suites.STANDARD:
            iid = self.tests.insert("", "end", text="🔒 " + suite_name(suites.load(sid)))
            self.list_rows[iid] = ("std", sid)
        for entry in self.store.entries():
            text = suite_name(entry.data) if entry.data else f"⚠ {entry.path.name}"
            iid = self.tests.insert("", "end", text=text)
            self.list_rows[iid] = ("own", entry.suite_id)
        if select:
            for iid, row in self.list_rows.items():
                if row == select:
                    self._loading = True
                    self.tests.selection_set(iid)
                    self.tests.see(iid)
                    self.update_idletasks()
                    self._loading = False

    def _selected(self) -> Optional[tuple[str, str]]:
        sel = self.tests.selection()
        return self.list_rows.get(sel[0]) if sel else None

    def _current_key(self) -> Optional[tuple[str, str]]:
        if self.data is None:
            return None
        return ("std", self.data["suite_info"]["id"]) if self.readonly else ("own", self.saved_id or "")

    def _pick_suite(self) -> None:
        if self._loading:
            return
        row = self._selected()
        if row is None or row == self._current_key():
            return
        if not self._leave():
            self.refresh_list(self._current_key())
            return
        kind, sid = row
        if kind == "std":
            self._show_suite(suites.load(sid), readonly=True)
            return
        try:
            self._show_suite(self.store.load(sid), saved_id=sid)
        except (CustomSuiteError, OSError) as e:
            self._show_suite(None, saved_id=sid)       # plik z błędem: widoczny opis, można go usunąć
            self._set_problems(getattr(e, "problems", [Problem(None, "file", "unreadable")]))

    def _leave(self) -> bool:
        """Niezapisane zmiany przed przejściem do innego testu albo zamknięciem: zapisz / porzuć / zostań."""
        if not self.dirty():
            return True
        answer = self.ask_save()
        if answer is None:
            return False
        return self.save() if answer else True

    def dirty(self) -> bool:
        if self.data is None or self.readonly:
            return False
        self._commit()
        return self._snapshot() != self.clean

    def _snapshot(self) -> str:
        return json.dumps([self.data, sorted(self.raw.items())], ensure_ascii=False, sort_keys=True)

    def _show_suite(self, data: Optional[dict], readonly: bool = False, saved_id: Optional[str] = None,
                    clean: bool = True) -> None:
        self.data = copy.deepcopy(data) if data is not None else None
        self.readonly = readonly or data is None
        self.saved_id = saved_id
        self.raw = {}
        self.index = None
        self._loading = True
        if data is None:
            self.name_var.set("")
            self.main_lang.set("")
            self.english_var.set(False)
        else:
            self.name_var.set(suite_name(self.data))
            self.form_lang.set(self.data["suite_info"]["languages"][0])
            self._show_langs()
        self._loading = False
        state = ["disabled"] if self.readonly else ["!disabled"]
        self.name_entry.state(state)
        self.main_lang.state(state)
        self.english_check.state(state)
        if not self.readonly and self.data["suite_info"]["languages"][0] == custom.ENGLISH:
            self.english_check.state(["disabled"])
        for key in ("add", "remove", "up", "down"):
            self.task_buttons[key].state(state)
        self.save_button.state(state)
        own = not readonly and saved_id is not None
        self.list_buttons["delete"].state(["!disabled"] if saved_id is not None else ["disabled"])
        self.list_buttons["export"].state(["!disabled"] if own else ["disabled"])
        self.list_buttons["copy"].state(["!disabled"] if data is not None else ["disabled"])
        self.preview_button.state(["!disabled"] if data is not None else ["disabled"])
        self.preview_result.configure(text="")
        self._set_problems([])
        self._fill_tasks()
        tasks = (self.data or {}).get("tasks") or []
        self._show_task(0 if tasks else None)
        if clean:
            self.clean = self._snapshot() if self.data is not None else None
        self._update_info()

    def _update_info(self) -> None:
        if self.data is None:
            self.info.configure(text="")
            return
        info = self.data["suite_info"]
        counts = custom.level_counts(self.data)
        parts = [t("q.ed.readonly") if self.readonly else t("q.ed.own")]
        parts.append(f"id: {info['id']} · {t('q.ed.version')} {info.get('version', '?')}")
        parts.append(t("q.ed.levels", n=len(self.data.get("tasks") or []), **counts))
        if info.get("based_on"):
            parts.append(t("q.ed.based_on", source=info["based_on"]))
        self.info.configure(text=" · ".join(parts))

    # ------------------------------------------------------------------ test: nowy, kopia, usuwanie, import, eksport
    def _taken(self) -> list[str]:
        return self.store.ids()

    def new_suite(self) -> None:
        if not self._leave():
            return
        name = self.ask_name(t("q.ed.new"), t("q.ed.new_name"))
        if not name or not name.strip():
            return
        lang = current_language() if current_language() in custom.COMMON_LANGUAGES else "pl"
        data = custom.new_suite(custom.unique_id(name, self._taken()), name.strip(), [lang])
        data["tasks"].append(custom.new_task(data))
        self._show_suite(data, clean=False)
        self.tests.selection_remove(*self.tests.selection())
        self.status.configure(text=t("q.ed.not_saved"))

    def copy_suite(self) -> None:
        if self.data is None:
            return
        self._commit()
        source = copy.deepcopy(self.data)
        if not self._leave():
            return
        name = self.ask_name(t("q.ed.copy"), t("q.ed.copy_name", name=suite_name(source)))
        if not name or not name.strip():
            return
        new_id = custom.unique_id(name, self._taken())
        info = source["suite_info"]
        if info.get("kind") != "custom":
            data = custom.copy_of_standard(info["id"], new_id, name.strip())
        else:
            data = source
            data["suite_info"].update(id=new_id, version=custom.FIRST_VERSION,
                                      name={lang: name.strip() for lang in info["languages"]})
        self._show_suite(data, clean=False)
        self.tests.selection_remove(*self.tests.selection())
        self.status.configure(text=t("q.ed.not_saved"))

    def delete_suite(self) -> None:
        if self.saved_id is None:
            return
        if not self.confirm(t("q.ed.delete_confirm", name=self.name_var.get() or self.saved_id)):
            return
        self.store.delete(self.saved_id)
        self._show_suite(None)
        self.refresh_list()
        self.on_change()

    def import_suite(self, path: Optional[str] = None) -> None:
        if not self._leave():
            return
        path = path or filedialog.askopenfilename(parent=self, title=t("q.ed.import"),
                                                  filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data, conflict = self.store.read_import(Path(path))
            if conflict:
                answer = self.ask_replace(suite_name(data))
                if answer is None:
                    return
                data = self.store.import_replace(data) if answer else self.store.import_as_copy(data)
            else:
                data = self.store.save(data)
        except CustomSuiteError as e:
            self.inform(t("q.ed.import_failed") + "\n\n" + "\n".join(problem_text(p) for p in e.problems[:8]))
            return
        except OSError as e:
            self.inform(t("q.ed.import_failed") + f"\n\n{e}")
            return
        sid = data["suite_info"]["id"]
        self.refresh_list(("own", sid))
        self._show_suite(data, saved_id=sid)
        self.status.configure(text=t("q.ed.imported"))
        self.on_change()

    def export_suite(self, path: Optional[str] = None) -> None:
        if self.saved_id is None or self.readonly:
            return
        if self.dirty() and not self.save():
            return
        path = path or filedialog.asksaveasfilename(parent=self, title=t("q.ed.export"), defaultextension=".json",
                                                    initialfile=f"{self.saved_id}.json",
                                                    filetypes=[("JSON", "*.json")])
        if not path:
            return
        target = self.store.export(self.saved_id, Path(path))
        self.status.configure(text=t("q.ed.exported", path=str(target)))

    # ------------------------------------------------------------------ języki
    def _show_langs(self) -> None:
        langs = self.data["suite_info"]["languages"]
        main = langs[0]
        if main not in self.lang_codes:                        # kod spoza listy (np. z importu)
            self.lang_codes.append(main)
            self.main_lang.configure(values=[lang_label(c) for c in self.lang_codes])
        self.main_lang.set(lang_label(main))
        self.english_var.set(len(langs) > 1)
        if self.form_lang.get() not in langs:
            self.form_lang.set(main)
        self.preview_lang.configure(values=[l.upper() for l in langs])
        if self.preview_lang.get().lower() not in langs:
            self.preview_lang.set(self.form_lang.get().upper())

    def _set_languages(self, langs: list[str], rename: Optional[tuple[str, str]] = None) -> None:
        """Nowy zestaw języków testu: [główny] albo [główny, en]. rename – zmiana języka głównego."""
        self._commit()
        info = self.data["suite_info"]
        old = info["languages"]
        name = self.name_var.get().strip()
        info["languages"] = langs
        info["name"] = {l: name for l in langs}
        for task in self.data["tasks"]:
            prompt = task.setdefault("prompt", {})
            spec = task.get("check") or {}
            keys = PER_LANG.get(spec.get("type"), ())
            if rename:
                src, dst = rename
                if src in prompt:
                    prompt[dst] = prompt.pop(src)
                for key in keys:
                    if isinstance(spec.get(key), dict) and src in spec[key]:
                        spec[key][dst] = spec[key].pop(src)
            for l in langs:
                prompt.setdefault(l, "")
            for l in list(prompt):
                if l not in langs:
                    del prompt[l]
            for key in keys:
                value = spec.get(key)
                if value is None:
                    continue
                if not isinstance(value, dict):          # wspólna wartość – rozpisana na języki
                    value = {l: copy.deepcopy(value) for l in old}
                    if rename:
                        value = {rename[1] if l == rename[0] else l: v for l, v in value.items()}
                for l in list(value):
                    if l not in langs:
                        del value[l]
                for l in langs:
                    value.setdefault(l, empty_value(spec["type"], key))
                spec[key] = value[langs[0]] if len(langs) == 1 else value
        self._show_langs()
        self.english_check.state(["disabled"] if langs[0] == custom.ENGLISH else ["!disabled"])
        self._show_task(self.index)

    def _change_main_lang(self) -> None:
        if self._loading or self.data is None or self.readonly:
            return
        labels = {lang_label(c): c for c in self.lang_codes}
        new = labels.get(self.main_lang.get())
        langs = self.data["suite_info"]["languages"]
        if new is None or new == langs[0]:
            return
        if new == custom.ENGLISH:                    # angielski jako główny – bez osobnej wersji angielskiej
            if len(langs) > 1:
                self._set_languages([custom.ENGLISH])
            else:
                self._set_languages([custom.ENGLISH], rename=(langs[0], custom.ENGLISH))
            return
        self._set_languages([new] + langs[1:], rename=(langs[0], new))

    def _toggle_english(self) -> None:
        if self._loading or self.data is None or self.readonly:
            return
        main = self.data["suite_info"]["languages"][0]
        if main == custom.ENGLISH:
            self.english_var.set(False)
            return
        self._set_languages([main, custom.ENGLISH] if self.english_var.get() else [main])

    def _switch_form_lang(self) -> None:
        """Przełącznik języka formularza (wariant B): jeden język naraz, drugi zostaje w danych."""
        self._commit()                                  # zapisuje język, który był widoczny (_shown_lang)
        self.preview_lang.set(self.form_lang.get().upper())
        self._show_task(self.index)

    # ------------------------------------------------------------------ zadania
    def _fill_tasks(self) -> None:
        self.task_tree.delete(*self.task_tree.get_children())
        for n, task in enumerate((self.data or {}).get("tasks") or []):
            self.task_tree.insert("", "end", iid=str(n), values=self._task_values(task))

    def _task_values(self, task: dict) -> tuple:
        level = task.get("level")
        kind = (task.get("check") or {}).get("type")
        return (task.get("id", "?"), t(f"q.level.{level}") if level in suites.LEVELS else str(level),
                cat_name(str(task.get("category", ""))),
                t(f"q.ed.type.{kind}") if kind in CHECK_TYPES else str(kind))

    def _pick_task(self) -> None:
        if self._loading:
            return
        sel = self.task_tree.selection()
        if not sel or int(sel[0]) == self.index:
            return
        self._commit()
        self._show_task(int(sel[0]))

    def _category_names(self) -> dict[str, str]:
        """Wyświetlana nazwa → zapisywana kategoria: standardowe (klucz, np. „przeksztalcanie” → „przekształcanie
        zdań”) i wszystkie kategorie z tego testu."""
        names = {}
        for sid in suites.STANDARD:
            for x in suites.load(sid)["tasks"]:
                names[cat_name(x["category"])] = x["category"]
        for x in (self.data or {}).get("tasks") or []:
            c = str(x.get("category") or "").strip()
            if c:
                names.setdefault(cat_name(c), c)
        return names

    def add_task(self) -> None:
        if self.data is None or self.readonly:
            return
        self._commit()
        task = custom.new_task(self.data)
        if self.index is not None and self.index < len(self.data["tasks"]):   # kategoria jak w bieżącym zadaniu
            task["category"] = self.data["tasks"][self.index].get("category", "")
        self.data["tasks"].append(task)
        self._fill_tasks()
        self._show_task(len(self.data["tasks"]) - 1)
        self._update_info()

    def remove_task(self) -> None:
        if self.data is None or self.readonly or self.index is None:
            return
        task = self.data["tasks"].pop(self.index)
        self.raw = {k: v for k, v in self.raw.items() if k[0] != task.get("id")}
        old, self.index = self.index, None
        self._fill_tasks()
        self._show_task(min(old, len(self.data["tasks"]) - 1) if self.data["tasks"] else None)
        self._update_info()

    def move_task(self, step: int) -> None:
        if self.data is None or self.readonly or self.index is None:
            return
        self._commit()
        tasks, i = self.data["tasks"], self.index
        j = i + step
        if not 0 <= j < len(tasks):
            return
        tasks[i], tasks[j] = tasks[j], tasks[i]
        self.index = None
        self._fill_tasks()
        self._show_task(j)

    # ------------------------------------------------------------------ formularz zadania
    def _label(self, text: str) -> None:
        ttk.Label(self.form, text=text).pack(anchor="w", pady=(6, 1))

    def _entry(self, key: str, label: str, value, width: int = 30) -> None:
        self._label(label)
        var = tk.StringVar(value="" if value is None else str(value))
        e = ttk.Entry(self.form, textvariable=var, width=width)
        e.pack(anchor="w", fill="x" if width > 30 else None)
        self.inputs[key] = (var, e)

    def _textbox(self, key: str, label: str, value: str, height: int = 3) -> None:
        self._label(label)
        box = self._text_widget(self.form, height)
        box.insert("1.0", value)
        box.pack(fill="x")
        self.inputs[key] = (None, box)

    @staticmethod
    def _text_widget(parent, height: int) -> tk.Text:
        return tk.Text(parent, height=height, wrap="word", font="TkFixedFont", undo=True, relief="flat",
                       background=COLORS["panel"], foreground=COLORS["text"], insertbackground=COLORS["text"],
                       highlightthickness=1, highlightbackground=COLORS["border"], padx=6, pady=4)

    def _get(self, key: str) -> str:
        var, widget = self.inputs[key]
        return var.get() if var is not None else widget.get("1.0", "end-1c")

    def _show_task(self, index: Optional[int]) -> None:
        for child in self.form.winfo_children():
            child.destroy()
        self.inputs = {}
        self.index = index
        if self.data is None or index is None or index >= len(self.data["tasks"]):
            self.index = None
            ttk.Label(self.form, text=t("q.ed.no_task"), foreground=COLORS["muted"]).pack(anchor="w", pady=10)
            return
        self._loading = True
        self.task_tree.selection_set(str(index))
        self.task_tree.see(str(index))
        self.update_idletasks()
        self._loading = False
        task = self.data["tasks"][index]
        langs = self.data["suite_info"]["languages"]
        tid = task.get("id", "?")
        ttk.Label(self.form, text=t("q.ed.task_title", id=tid), font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self._label(t("q.ed.field.category"))
        # Podpowiedzi: kategorie z tego testu i z testów standardowych (nazwy po polsku/angielsku); można wpisać
        # własną. Zgłoszenie autora 13.11.2026: przy nowym teście lista była pusta i wyglądała na zepsutą.
        self.cat_names = self._category_names()
        current = str(task.get("category", ""))
        var = tk.StringVar(value=cat_name(current) if current in self.cat_names.values() else current)
        cb = ttk.Combobox(self.form, textvariable=var, values=sorted(self.cat_names, key=str.casefold), width=30)
        cb.pack(anchor="w")
        ttk.Label(self.form, text=t("q.ed.hint.category"), foreground=COLORS["muted"]).pack(anchor="w")
        self.inputs["category"] = (var, cb)
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text=t("q.ed.field.level")).pack(side="left")
        level_names = [t(f"q.level.{l}") for l in suites.LEVELS]
        lv = tk.StringVar(value=t(f"q.level.{task.get('level')}") if task.get("level") in suites.LEVELS else "")
        lcb = ttk.Combobox(row, textvariable=lv, values=level_names, state="readonly", width=10)
        lcb.pack(side="left", padx=(6, 14))
        self.inputs["level"] = (lv, lcb)
        ttk.Label(row, text=t("q.ed.field.max_tokens")).pack(side="left")
        mv = tk.StringVar(value=str(task.get("max_tokens", "")))
        sb = ttk.Entry(row, textvariable=mv, width=7)
        sb.pack(side="left", padx=6)
        self.inputs["max_tokens"] = (mv, sb)
        if self.form_lang.get() not in langs:
            self.form_lang.set(langs[0])
        lang = self._shown_lang = self.form_lang.get()
        if len(langs) > 1:
            switch = ttk.Frame(self.form)
            switch.pack(anchor="w", pady=(10, 0))
            ttk.Label(switch, text=t("q.ed.form_lang")).pack(side="left", padx=(0, 6))
            for l in langs:
                ttk.Radiobutton(switch, text=lang_label(l), value=l, variable=self.form_lang,
                                command=self._switch_form_lang).pack(side="left", padx=(0, 8))
        self._textbox(f"prompt.{lang}", field_label(f"prompt.{lang}"),
                      str((task.get("prompt") or {}).get(lang, "")), height=4)
        spec = task.get("check") or {}
        kind = spec.get("type")
        self._label(t("q.ed.field.check.type"))
        tv = tk.StringVar(value=t(f"q.ed.type.{kind}") if kind in CHECK_TYPES else str(kind))
        tcb = ttk.Combobox(self.form, textvariable=tv, values=[t(f"q.ed.type.{k}") for k in CHECK_TYPES],
                           state="readonly", width=24)
        tcb.pack(anchor="w")
        tcb.bind("<<ComboboxSelected>>", lambda e: self._change_type())
        self.inputs["type"] = (tv, tcb)
        hint = t(f"q.ed.hint.{kind}") if has_key(f"q.ed.hint.{kind}") else ""
        if hint:
            ttk.Label(self.form, text=hint, foreground=COLORS["muted"], wraplength=420, justify="left").pack(
                anchor="w", pady=(2, 0))
        self._check_fields(tid, spec, langs, lang)
        if self.readonly:
            for var, widget in self.inputs.values():
                if isinstance(widget, tk.Text):
                    widget.configure(state="disabled")
                else:
                    widget.state(["disabled"])

    def _check_fields(self, tid: str, spec: dict, langs: list[str], lang: str) -> None:
        kind = spec.get("type")
        suffix = f" ({lang.upper()})" if len(langs) > 1 else ""
        if kind == "text":
            self._textbox(f"expected.{lang}", t("q.ed.field.check.expected") + suffix,
                          "\n".join(per_lang(spec.get("expected"), lang) or []))
            self._textbox(f"sentence.{lang}", t("q.ed.field.check.sentence") + suffix,
                          "\n".join(per_lang(spec.get("sentence"), lang) or []), height=2)
        elif kind == "number":
            self._entry("expected", t("q.ed.field.check.expected"), spec.get("expected"))
            self._entry("tolerance", t("q.ed.field.check.tolerance"), spec.get("tolerance"))
        elif kind == "fraction":
            self._entry("expected", t("q.ed.field.check.expected"), spec.get("expected"))
        elif kind == "choice":
            self._textbox(f"options.{lang}", t("q.ed.field.check.options") + suffix,
                          "\n".join(per_lang(spec.get("options"), lang) or []))
            self._entry(f"expected.{lang}", t("q.ed.field.check.expected") + suffix, per_lang(spec.get("expected"), lang))
        elif kind == "json":
            raw = self.raw.get((tid, "check.expected"))
            text = raw[0] if raw else json.dumps(spec.get("expected"), ensure_ascii=False, indent=1)
            self._textbox("json", t("q.ed.field.check.expected"), text, height=6)
        elif kind == "python":
            self._entry("function", t("q.ed.field.check.function"), spec.get("function"))
            raw = self.raw.get((tid, "check.tests"))
            self._textbox("tests", t("q.ed.field.check.tests"), raw[0] if raw else
                          format_python_tests(spec.get("tests")), height=6)

    def _change_type(self) -> None:
        if self.index is None or self.readonly:
            return
        names = {t(f"q.ed.type.{k}"): k for k in CHECK_TYPES}
        kind = names.get(self.inputs["type"][0].get())
        task = self.data["tasks"][self.index]
        if kind is None or kind == (task.get("check") or {}).get("type"):
            return
        self._commit(check=False)
        task["check"] = custom.new_task(self.data, kind)["check"]
        self.raw = {k: v for k, v in self.raw.items() if k[0] != task.get("id")}
        self._show_task(self.index)
        self.task_tree.item(str(self.index), values=self._task_values(task))

    def _commit(self, check: bool = True) -> None:
        """Formularz → słownik zadania. Wartości, których nie da się odczytać, zostają do pokazania błędu."""
        if self.data is None or self.readonly:
            return
        name = self.name_var.get().strip()
        info = self.data["suite_info"]
        info["name"] = {lang: name for lang in info["languages"]}
        if self.index is None or self.index >= len(self.data["tasks"]) or "category" not in self.inputs:
            return
        task = self.data["tasks"][self.index]
        langs = info["languages"]
        tid = task.get("id", "?")
        typed = self._get("category").strip()
        task["category"] = getattr(self, "cat_names", {}).get(typed, typed)    # nazwa z listy → klucz kategorii
        levels = {t(f"q.level.{l}"): l for l in suites.LEVELS}
        task["level"] = levels.get(self._get("level"), task.get("level"))
        mt = self._get("max_tokens").strip()
        task["max_tokens"] = int(mt) if mt.isdigit() else mt
        lang = self._shown_lang
        prompt = task.setdefault("prompt", {})
        prompt[lang] = self._get(f"prompt.{lang}").strip()
        for l in langs:
            prompt.setdefault(l, "")
        spec = task.get("check") or {}
        kind = spec.get("type")
        if check and kind in CHECK_TYPES:
            self._commit_check(tid, spec, kind, langs, lang)
        self.task_tree.item(str(self.index), values=self._task_values(task))
        self._update_info()

    def _commit_check(self, tid: str, spec: dict, kind: str, langs: list[str], lang: str) -> None:
        if kind == "text":
            put_lang(spec, "expected", lines(self._get(f"expected.{lang}")), langs, lang)
            put_lang(spec, "sentence", lines(self._get(f"sentence.{lang}")), langs, lang)
            if not any(per_lang(spec["sentence"], l) for l in langs):
                spec.pop("sentence", None)
        elif kind == "number":
            spec["expected"] = parse_number(self._get("expected"))
            tol = self._get("tolerance").strip()
            if tol:
                spec["tolerance"] = parse_number(tol)
            else:
                spec.pop("tolerance", None)
        elif kind == "fraction":
            spec["expected"] = self._get("expected").replace(" ", "").strip()
        elif kind == "choice":
            put_lang(spec, "options", lines(self._get(f"options.{lang}")), langs, lang)
            put_lang(spec, "expected", self._get(f"expected.{lang}").strip(), langs, lang)
        elif kind == "json":
            text = self._get("json")
            try:
                spec["expected"] = json.loads(text)
                self.raw.pop((tid, "check.expected"), None)
            except ValueError:
                self.raw[(tid, "check.expected")] = (text, "json_syntax")
        elif kind == "python":
            spec["function"] = self._get("function").strip()
            text = self._get("tests")
            tests = parse_python_tests(text)
            if tests is None:
                self.raw[(tid, "check.tests")] = (text, "python_tests_syntax")
            else:
                spec["tests"] = tests
                self.raw.pop((tid, "check.tests"), None)

    # ------------------------------------------------------------------ błędy, zapis
    def check_all(self) -> list[Problem]:
        self._commit()
        found = [Problem(tid, field, code) for (tid, field), (_, code) in sorted(self.raw.items())]
        return found + custom.problems(self.data)

    def _set_problems(self, found: list[Problem], checked: bool = False) -> None:
        self.problems = list(found)
        self.problem_list.delete(0, "end")
        for p in self.problems:
            self.problem_list.insert("end", problem_text(p))
        if checked and not self.problems and self.data is not None and not self.readonly:
            self.problem_list.insert("end", t("q.ed.no_problems"))

    def _goto_problem(self) -> None:
        sel = self.problem_list.curselection()
        if not sel or sel[0] >= len(self.problems) or self.data is None:
            return
        task = self.problems[sel[0]].task
        for n, x in enumerate(self.data["tasks"]):
            if x.get("id") == task or f"#{n + 1}" == task:
                if n != self.index:
                    self._commit()
                    self._show_task(n)
                return

    def save(self) -> bool:
        if self.data is None or self.readonly:
            return False
        if self.busy():
            self.inform(t("q.ed.busy"))
            return False
        found = self.check_all()
        self._set_problems(found)
        if found:
            self.status.configure(text=t("q.ed.save_blocked", n=len(found)))
            return False
        try:
            data = self.store.save(self.data)
        except CustomSuiteError as e:
            self._set_problems(e.problems)
            return False
        sid = data["suite_info"]["id"]
        index = self.index
        self.refresh_list(("own", sid))
        self._show_suite(data, saved_id=sid)
        if index is not None and index < len(data["tasks"]):
            self._show_task(index)
        self.status.configure(text=t("q.ed.saved", version=data["suite_info"]["version"]))
        self.on_change()
        return True

    # ------------------------------------------------------------------ podgląd oceny
    def run_preview(self) -> None:
        if self.data is None or self.index is None:
            return
        self._commit()
        task = self.data["tasks"][self.index]
        tid = task.get("id")
        bad = [p for p in self.check_all() if p.task == tid and p.field.startswith("check")]
        if bad:
            self.preview_result.configure(text=t("q.ed.preview_fix") + " " + problem_text(bad[0]))
            return
        answer = self.preview_answer.get("1.0", "end-1c")
        lang = self.preview_lang.get().lower() or self.data["suite_info"]["languages"][0]
        c = custom.preview(task["check"], answer, lang, self.sandbox)
        self.preview_result.configure(text=f"{MARK[c.correct]} {t(f'q.reason.{c.reason}')}")

    # ------------------------------------------------------------------ zamykanie
    def close(self) -> None:
        if self._leave():
            self.destroy()
