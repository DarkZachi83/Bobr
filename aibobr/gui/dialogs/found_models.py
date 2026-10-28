"""„Modele na dysku…” (etap 9a): zgoda → szukanie w tle → lista → sprawdzenie modelu.

Działa w obu trybach – „co ja w ogóle mam?” to pytanie początkującego.
Szukanie idzie w wątku w tle (Worker); okno tylko co chwilę odczytuje licznik plików.
"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Optional

from aibobr.i18n import fmt_gb, t
from aibobr.scan.model import FoundModel, Kind, ScanReport

ORDER = {Kind.TEXT: 0, Kind.IMAGE: 1, Kind.ADDON: 2, Kind.OTHER: 3}


class FoundModelsDialog(tk.Toplevel):
    def __init__(self, master, worker, find: Callable[..., ScanReport],
                 consent: bool, save_consent: Callable[[bool], None]):
        super().__init__(master)
        self.title(t("gui.scan_title"))
        self.transient(master)
        self.worker, self.find, self.save_consent = worker, find, save_consent
        self.result: Optional[str] = None
        self.report = ScanReport()
        self.rows: dict[str, FoundModel] = {}
        self.stop = threading.Event()
        self.seen = 0
        self.running = False

        self.body = ttk.Frame(self, padding=12)
        self.body.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda e: self.close())
        if consent:
            self._build_list()
            self.search()
        else:
            self._build_consent()
        self.grab_set()

    # ------------------------------------------------------------------ zgoda
    def _build_consent(self) -> None:
        self.consent_frame = ttk.Frame(self.body)
        self.consent_frame.pack(fill="both", expand=True)
        ttk.Label(self.consent_frame, text=t("gui.scan_consent"), wraplength=460,
                  justify="left").pack(anchor="w", pady=(0, 12))
        buttons = ttk.Frame(self.consent_frame)
        buttons.pack(anchor="e")
        ttk.Button(buttons, text=t("gui.scan_consent_no"), command=self.close).pack(side="right")
        self.consent_yes = ttk.Button(buttons, text=t("gui.scan_consent_yes"), style="Accent.TButton",
                                      command=self._agree)
        self.consent_yes.pack(side="right", padx=6)

    def _agree(self) -> None:
        self.save_consent(True)
        self.consent_frame.destroy()
        self._build_list()
        self.search()

    # ------------------------------------------------------------------ lista
    def _build_list(self) -> None:
        self.status = ttk.Label(self.body, text="", wraplength=640, justify="left")
        self.status.pack(anchor="w", pady=(0, 6))
        frame = ttk.Frame(self.body)
        frame.pack(fill="both", expand=True)
        cols = ("kind", "size", "app")
        self.tree = ttk.Treeview(frame, columns=cols, height=14, selectmode="browse")
        self.tree.heading("#0", text=t("gui.scan_col.name"))
        self.tree.column("#0", width=330)
        for col, width, anchor in (("kind", 190, "w"), ("size", 80, "e"), ("app", 110, "w")):
            self.tree.heading(col, text=t(f"gui.scan_col.{col}"))
            self.tree.column(col, width=width, anchor=anchor, stretch=False)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-Button-1>", lambda e: self._ok())
        self.tree.bind("<Return>", lambda e: self._ok())
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._selected())
        self.path_label = ttk.Label(self.body, text="", foreground="#667085", wraplength=640)
        self.path_label.pack(anchor="w", pady=(4, 0))

        buttons = ttk.Frame(self.body)
        buttons.pack(fill="x", pady=(8, 0))
        self.folder_button = ttk.Button(buttons, text=t("gui.scan_folder"), command=self.pick_folder)
        self.folder_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text=t("gui.scan_stop"), command=self.stop.set)
        ttk.Button(buttons, text=t("gui.cancel"), command=self.close).pack(side="right")
        self.ok = ttk.Button(buttons, text=t("gui.choose"), style="Accent.TButton", command=self._ok)
        self.ok.pack(side="right", padx=6)
        self.ok.state(["disabled"])

    # ------------------------------------------------------------------ szukanie
    def search(self, folder: Optional[Path] = None) -> None:
        if self.running:
            return
        self.running, self.seen, self.folder = True, 0, folder
        self.stop.clear()
        self.folder_button.state(["disabled"])
        self.stop_button.pack(side="left", padx=6)

        def progress(n: int) -> None:          # wątek w tle – tylko przypisanie liczby
            self.seen = n
        self.worker.run("find_models", lambda: self.find(folder, self.stop, progress),
                        self._done, self._failed)
        self._tick()

    def _tick(self) -> None:
        if self.running and self.winfo_exists():
            self.status.configure(text=t("gui.scan_searching", count=self.seen))
            self.after(200, self._tick)

    def _finish(self) -> None:
        self.running = False
        if not self.winfo_exists():
            return
        self.folder_button.state(["!disabled"])
        self.stop_button.pack_forget()

    def _done(self, report: ScanReport) -> None:
        self._finish()
        if not self.winfo_exists():
            return
        self.report.merge(report)
        self._fill()
        if not report.models:
            text = t("gui.scan_none_folder") if self.folder else t("gui.scan_none")
        else:
            text = t("gui.scan_found", count=len(self.report.models), size=fmt_gb(self.report.total_size))
        if report.cancelled:
            text += " " + t("gui.scan_cancelled")
        if report.denied:
            text += "\n" + t("gui.scan_denied", folders=", ".join(str(d) for d in report.denied[:3]))
        self.status.configure(text=text)

    def _failed(self, error: BaseException) -> None:
        self._finish()
        if self.winfo_exists():
            self.status.configure(text=t("gui.unexpected", error=repr(error)))

    def _fill(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        models = sorted(self.report.models, key=lambda m: (ORDER[m.kind], m.name.lower()))
        for i, m in enumerate(models):
            iid = str(i)
            self.rows[iid] = m
            kind = t(f"scan.kind.{m.kind.value}") + (f" · {t(m.note)}" if m.note else "")
            name = m.name + (f" · {m.quant}" if m.quant else "")
            self.tree.insert("", "end", iid=iid, text=name, values=(kind, fmt_gb(m.size), m.app))
        first = next((iid for iid, m in self.rows.items() if m.checkable), None)
        if first:
            self.tree.selection_set(first)
            self.tree.see(first)
            self.tree.focus_set()

    def _selected(self) -> Optional[FoundModel]:
        sel = self.tree.selection()
        m = self.rows.get(sel[0]) if sel else None
        self.ok.state(["!disabled"] if m and m.checkable else ["disabled"])
        if m:
            self.path_label.configure(text=str(m.path) if m.checkable
                                      else f"{m.path}\n{t('gui.scan_not_checkable')}")
        return m

    def pick_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title=t("gui.scan_folder_title"), mustexist=True)
        if folder:
            self.search(Path(folder))

    def _ok(self) -> None:
        m = self._selected()
        if m and m.checkable:
            self.result = m.target
            self.close()

    def close(self) -> None:
        self.stop.set()                          # szukanie w tle kończy się przy następnym folderze
        self.destroy()
