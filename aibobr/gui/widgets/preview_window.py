"""Okno „Podgląd” (uwaga autora po etapie 12c): prompt i odpowiedź modelu na żywo.

Otwiera się samo przy Starcie benchmarku. Zamknięcie tylko chowa okno – test trwa dalej,
a przycisk „Przywróć” w zakładce Benchmark pokazuje je znowu. Zapis całego przebiegu
(wszystkie zadania po kolei) zostaje do następnego Startu: Kopiuj, Zapisz…, Eksport CSV…
"""
from __future__ import annotations

import csv
import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

from aibobr.i18n import t

TAGS = {"head": "#66c9fa", "prompt": "#9bacc0", "answer": "#e9ecef", "ok": "#53d6a0", "bad": "#ff8282",
        "task": "#f5c65f"}


class PreviewWindow(tk.Toplevel):
    def __init__(self, master, on_status: Callable[[str], None] = lambda text: None):
        super().__init__(master)
        self.on_status = on_status
        self.title(t("bench.preview.window"))
        self.geometry("760x640")
        self.minsize(480, 360)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)        # chowa – test trwa dalej
        self.rows: list[dict] = []                               # zadania jakości – do CSV
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        self.task = ttk.Label(body, text=t("bench.preview.empty"), style="Title.TLabel", wraplength=700)
        self.task.pack(anchor="w", fill="x")
        self.task.bind("<Configure>", lambda e: self.task.configure(wraplength=max(300, e.width)))
        tools = ttk.Frame(body)
        tools.pack(fill="x", pady=(6, 6))
        ttk.Button(tools, text=t("bench.preview.copy"), command=self.copy).pack(side="left")
        ttk.Button(tools, text=t("bench.preview.save"), command=self.save).pack(side="left", padx=6)
        ttk.Button(tools, text=t("bench.log.csv"), command=self.export_csv).pack(side="left")
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(tools, text=t("bench.log.autoscroll"), variable=self.follow).pack(side="right")
        box = ttk.Frame(body)
        box.pack(fill="both", expand=True)
        self.text = tk.Text(box, wrap="word", font="TkFixedFont", background="#11171d", foreground="#cfd8e3",
                            relief="flat", borderwidth=0, padx=12, pady=10, state="disabled",
                            selectbackground="#354e66", selectforeground="#ffffff")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for tag, color in TAGS.items():
            self.text.tag_configure(tag, foreground=color)

    # ------------------------------------------------------------------ zawartość
    def _insert(self, text: str, tag: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", text, (tag,))
        self.text.configure(state="disabled")
        if self.follow.get():
            self.text.see("end")

    def clear(self) -> None:
        self.rows.clear()
        self.task.configure(text=t("bench.preview.empty"))
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def start(self, title: str, prompt: str) -> None:
        self.task.configure(text=title)
        if self.text.index("end-1c") != "1.0":
            self._insert("\n" + "─" * 60 + "\n", "head")
        self._insert(f"▶ {title}\n", "task")
        self._insert(t("bench.preview.prompt") + "\n", "head")
        self._insert(prompt + "\n\n", "prompt")
        self._insert(t("bench.preview.answer") + "\n", "head")

    def add(self, text: str) -> None:
        self._insert(text, "answer")

    def verdict(self, text: str, ok: bool, row: dict = None) -> None:
        self._insert("\n\n" + text + "\n", "ok" if ok else "bad")
        if row:
            self.rows.append(row)

    def content(self) -> str:
        return self.text.get("1.0", "end-1c")

    # ------------------------------------------------------------------ przyciski
    def show(self) -> None:
        self.deiconify()
        self.lift()

    def copy(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.content())
        self.on_status(t("bench.preview.copied"))

    def save(self) -> None:
        name = f"bobr_podglad_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        path = filedialog.asksaveasfilename(parent=self, title=t("bench.log.save"), initialfile=name,
                                            defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            Path(path).write_text(self.content() + "\n", encoding="utf-8")
            self.on_status(t("bench.log.saved", path=path))

    def export_csv(self) -> None:
        """Zadania jakości: model, pakiet, zadanie, ocena, powód, prompt, odpowiedź."""
        if not self.rows:
            self.on_status(t("bench.csv_none"))
            return
        name = f"bobr_odpowiedzi_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        path = filedialog.asksaveasfilename(parent=self, title=t("bench.log.csv"), initialfile=name,
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["model", "suite", "test", "passed", "reason", "prompt", "answer"])
            for r in self.rows:
                w.writerow([r["model"], r["suite"], r["test"], r["passed"], r["reason"], r["prompt"], r["answer"]])
        self.on_status(t("bench.log.saved", path=path))
