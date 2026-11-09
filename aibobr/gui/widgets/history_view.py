"""Zakładka „Historia” benchmarku 0.9 (decyzja autora, pytanie 9a).

Lista sesji (model w sesji = wiersz): data, przebiegi, werdykt z dnia testu, zakres generowania
(pomiar rozrzutu), TTFT, uwagi. Filtr po modelu, przełącznik „według obecnych progów” (kolumna
pokazuje werdykt tylko tam, gdzie wyszedłby inny), szczegóły (dwuklik), eksport sesji do folderu,
usunięcie sesji z potwierdzeniem. Pod spodem „Archiwum (przed 0.9)” – tabele perf i quality tylko
do odczytu, z eksportem; ich zawartości nie zmieniamy.
"""
from __future__ import annotations

import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from aibobr.bench import export, history, speed
from aibobr.gui import speed_format as sf
from aibobr.i18n import fmt_context, t

COLS = ("date", "model", "runs", "verdict", "now", "gen", "ttft", "notes")
WIDTHS = {"date": 130, "model": 190, "runs": 50, "verdict": 150, "now": 180, "gen": 140, "ttft": 130,
          "notes": 260}
PERF_COLS = ("measured_at", "model", "context", "pp_tps", "ttft_ms", "tg_tps", "runtime_version")
QUALITY_COLS = ("measured_at", "model", "suite", "test_id", "passed", "mode")


def date_text(iso: str) -> str:
    return (iso or "")[:16].replace("T", " ")


def verdicts_text(values: list) -> str:
    """Werdykty przebiegów: jeden – „🟢 Szybko”; różne – po kolei (rozrzut też widać)."""
    labels = []
    for v in values:
        label = sf.level(v) if v in sf.DOTS else t(f"hist.status.{v}") if v else t("speed.no_data")
        if label not in labels:
            labels.append(label)
    return " / ".join(labels)


class HistoryView(ttk.Frame):
    def __init__(self, master, controller, on_status: Callable[[str], None] = lambda text: None):
        super().__init__(master, style="Panel.TFrame", padding=8)
        self.c, self.on_status = controller, on_status
        self.confirm = lambda title, text: messagebox.askokcancel(title, text, parent=self)
        self.rows: dict[str, dict] = {}
        split = ttk.Panedwindow(self, orient="vertical")       # sesje u góry, archiwum pod spodem (przesuwane)
        split.pack(fill="both", expand=True)
        top = ttk.Frame(split, style="Inner.TFrame")
        bottom = ttk.Frame(split, style="Inner.TFrame")
        split.add(top, weight=3)
        split.add(bottom, weight=2)
        bar = ttk.Frame(top, style="Inner.TFrame")
        bar.pack(fill="x")
        ttk.Label(bar, text=t("hist.model"), style="Panel.TLabel").pack(side="left")
        self.model = tk.StringVar(value=t("hist.all"))
        self.model_box = ttk.Combobox(bar, textvariable=self.model, state="readonly", width=34)
        self.model_box.pack(side="left", padx=6)
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.current = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text=t("hist.current"), variable=self.current, style="Panel.TCheckbutton",
                        command=self.refresh).pack(side="left", padx=6)
        ttk.Button(bar, text=t("hist.delete"), command=self.delete).pack(side="right")
        ttk.Button(bar, text=t("speed.export"), command=self.export).pack(side="right", padx=6)
        ttk.Button(bar, text=t("hist.details"), command=self.details).pack(side="right")
        box = ttk.Frame(top, style="Inner.TFrame")
        box.pack(fill="both", expand=True, pady=(6, 0))
        self.table = ttk.Treeview(box, columns=COLS, show="headings", height=6, selectmode="browse")
        for col in COLS:
            self.table.heading(col, text=t(f"hist.col.{col}"))
            self.table.column(col, width=WIDTHS[col], minwidth=40, anchor="w", stretch=col in ("model", "notes"))
        for name, color in (("fast", "#53d6a0"), ("useful", "#66c9fa"), ("slow", "#f5c65f"),
                            ("too_slow", "#ff8282")):
            self.table.tag_configure(name, foreground=color)
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.table.bind("<Double-Button-1>", lambda e: self.details())
        self.empty = ttk.Label(top, text="", style="Muted.TLabel")
        self.empty.pack(anchor="w", pady=(4, 0))

        arch = ttk.Frame(bottom, style="Inner.TFrame")
        arch.pack(fill="x", pady=(8, 0))
        ttk.Label(arch, text=t("hist.archive"), style="Panel.TLabel", font=("TkDefaultFont", 10, "bold")).pack(
            side="left")
        self.archive_kind = tk.StringVar(value="perf")
        for kind in ("perf", "quality"):
            ttk.Radiobutton(arch, text=t(f"hist.archive.{kind}"), value=kind, variable=self.archive_kind,
                            style="Panel.TRadiobutton", command=self.fill_archive).pack(side="left", padx=6)
        ttk.Button(arch, text=t("hist.archive.export"), command=self.export_archive).pack(side="right")
        self.archive_note = ttk.Label(bottom, text=t("hist.archive.note"), style="Muted.TLabel")
        self.archive_note.pack(side="bottom", anchor="w")
        abox = ttk.Frame(bottom, style="Inner.TFrame")
        abox.pack(fill="both", expand=True, pady=(4, 0))
        self.archive = ttk.Treeview(abox, show="headings", height=4)
        ascroll = ttk.Scrollbar(abox, orient="vertical", command=self.archive.yview)
        self.archive.configure(yscrollcommand=ascroll.set)
        self.archive.pack(side="left", fill="both", expand=True)
        ascroll.pack(side="right", fill="y")

    @property
    def db(self):
        return getattr(self.c, "bench_db", None)

    def thresholds(self) -> dict:
        return speed.thresholds_of(self.c.settings)

    # ------------------------------------------------------------------ lista
    def refresh(self) -> None:
        db = self.db
        self.table.delete(*self.table.get_children())
        self.rows.clear()
        if db is None:
            return
        names = history.models(db)
        self.model_box.configure(values=[t("hist.all")] + names)
        if self.model.get() not in names:
            self.model.set(t("hist.all"))
        chosen = None if self.model.get() == t("hist.all") else self.model.get()
        current = self.thresholds() if self.current.get() else None
        for row in history.rows(db, chosen, current):
            rng, ttft = row["spread"], row["ttft"]
            gen = "–" if rng is None else (f"{sf.num(rng[0])} tok/s" if rng[0] == rng[1] else
                                          f"{sf.num(rng[0])}–{sf.num(rng[1])} tok/s")
            values = {"date": date_text(row["started_at"]), "model": row["model"],
                      "runs": f"{row['runs']}" + ("" if row["status"] == "finished" else " ⚠"),
                      "verdict": verdicts_text(row["verdicts"]),
                      "now": verdicts_text(row["verdicts_now"]) if current and row["changed"] else "",
                      "gen": gen, "ttft": "–" if ttft is None else (sf.ms(ttft[0]) if ttft[0] == ttft[1] else
                                                                     f"{sf.ms(ttft[0])}–{sf.ms(ttft[1])}"),
                      "notes": " · ".join(t(f"speed.warn.{w}") for w in row["warnings"])
                      + ("" if row["status"] == "finished" else t(f"hist.status.{row['status']}"))}
            level = next((v for v in row["verdicts"] if v in sf.DOTS), None)
            iid = self.table.insert("", "end", values=[values[c] for c in COLS], tags=(level,) if level else ())
            self.rows[iid] = row
        self.empty.configure(text="" if self.rows else t("hist.empty"))
        self.fill_archive()

    def selected(self):
        sel = self.table.selection()
        return self.rows.get(sel[0]) if sel else None

    def details(self) -> None:
        row = self.selected()
        if row is None or self.db is None:
            return
        win = tk.Toplevel(self)
        win.title(t("hist.details_title", model=row["model"], date=date_text(row["started_at"])))
        win.geometry("820x640")
        text = tk.Text(win, wrap="none", font="TkFixedFont", background="#11171d", foreground="#cfd8e3",
                       relief="flat", padx=12, pady=10)
        scroll = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        th = self.thresholds()
        parts = []
        for rid in row["run_ids"]:
            run = history.run_details(self.db, rid)
            if run is None:
                continue
            block = sf.details(run, th)
            now = speed.recompute(run, th)
            if run.get("status") == "finished" and now != run.get("verdict"):
                block = block.replace("\n", "\n" + t("hist.now_line", verdict=sf.level(now)) + "\n", 1)
            parts.append(block)
        text.insert("end", ("\n\n" + "═" * 70 + "\n\n").join(parts))
        text.configure(state="disabled")
        self.details_window = win

    def export(self) -> None:
        row = self.selected()
        if row is None or self.db is None:
            messagebox.showinfo(t("speed.export"), t("hist.pick"), parent=self)
            return
        parent = filedialog.askdirectory(parent=self, title=t("speed.export"), mustexist=True)
        if parent:
            folder = export.export_session(self.db, row["session_id"], parent)
            self.on_status(t("speed.exported", path=str(folder)))

    def delete(self) -> None:
        row = self.selected()
        if row is None or self.db is None:
            messagebox.showinfo(t("hist.delete"), t("hist.pick"), parent=self)
            return
        session = self.db.session(row["session_id"])
        models = ", ".join(sf.loads(session.get("models_json"), []) or [row["model"]])
        if not self.confirm(t("hist.delete"), t("hist.delete_confirm", date=date_text(row["started_at"]),
                                                  models=models)):
            return
        self.db.delete_session(row["session_id"])
        self.on_status(t("hist.deleted"))
        self.refresh()

    # ------------------------------------------------------------------ archiwum (przed 0.9)
    def fill_archive(self) -> None:
        db = self.db
        kind = self.archive_kind.get()
        cols = PERF_COLS if kind == "perf" else QUALITY_COLS
        self.archive.delete(*self.archive.get_children())
        self.archive.configure(columns=cols)
        for col in cols:
            self.archive.heading(col, text=t(f"hist.acol.{col}"))
            self.archive.column(col, width=200 if col == "model" else 140 if col == "measured_at" else 110,
                                minwidth=40, anchor="w",
                                stretch=col == "model")
        if db is None:
            return
        for r in reversed(db.archive(kind)):
            values = []
            for col in cols:
                v = r.get(col)
                if col == "context" and v:
                    v = fmt_context(v)
                elif col == "passed":
                    v = "✓" if v else "✗"
                elif isinstance(v, float):
                    v = sf.num(v, 0 if col in ("pp_tps", "ttft_ms") else 1)
                values.append("–" if v is None else v)
            self.archive.insert("", "end", values=values)

    def export_archive(self) -> None:
        if self.db is None:
            return
        parent = filedialog.askdirectory(parent=self, title=t("hist.archive.export"), mustexist=True)
        if parent:
            folder = export.export_archive(self.db, parent, f"{datetime.date.today():%Y-%m-%d}")
            self.on_status(t("speed.exported", path=str(folder)))
