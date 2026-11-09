"""Okno „Stabilizacja temperatury trwa dłużej niż oczekiwano” (tekst autora, 9.11.2026).

Pomiar idzie w wątku roboczym, okno musi powstać w wątku Tk: `ThreadsafeAsk(root)` daje obiekt wywoływalny
dla `telemetry.cool_down(on_timeout=...)`, która prosi wątek Tk o okno i czeka na odpowiedź.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from aibobr.i18n import t


class CoolingDialog(tk.Toplevel):
    """Dwa przyciski z opisem pod spodem. Zamknięcie okna = „Czekaj dalej” (nic nie przyspieszamy)."""

    def __init__(self, master):
        super().__init__(master)
        self.result = False
        self.title(t("bench.cooling.title"))
        self.transient(master)
        self.resizable(False, False)
        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=t("bench.cooling.title"), font=("TkDefaultFont", 11, "bold"),
                  wraplength=460).pack(anchor="w")
        ttk.Label(body, text=t("bench.cooling.body"), wraplength=460, justify="left").pack(anchor="w", pady=(6, 12))
        choices = ttk.Frame(body)
        choices.pack(fill="x")
        for col, (key, value, style) in enumerate((("continue", True, "Accent.TButton"),
                                                   ("wait", False, "TButton"))):
            box = ttk.Frame(choices)
            box.grid(row=0, column=col, sticky="nw", padx=(0 if col == 0 else 12, 0))
            button = ttk.Button(box, text=t(f"bench.cooling.{key}"), style=style,
                                command=lambda v=value: self.choose(v))
            button.pack(anchor="w")
            ttk.Label(box, text=t(f"bench.cooling.{key}_hint"), wraplength=220, justify="left").pack(
                anchor="w", pady=(4, 0))
            if value:
                self.continue_button = button
            else:
                self.wait_button = button
        self.protocol("WM_DELETE_WINDOW", lambda: self.choose(False))
        self.bind("<Escape>", lambda e: self.choose(False))

    def choose(self, value: bool) -> None:
        self.result = value
        self.destroy()


def ask_cooling_timeout(master) -> bool:
    dialog = CoolingDialog(master)
    dialog.grab_set()
    master.wait_window(dialog)
    return dialog.result


class ThreadsafeAsk:
    """Funkcja dla wątku roboczego: prośba trafia do kolejki, okno pokazuje wątek Tk (sprawdza
    kolejkę co 100 ms – tkintera nie wolno dotykać z innego wątku), odpowiedź wraca przez Event.
    Tworzymy w wątku Tk (przy Starcie), `close()` po teście. Zatrzymanie testu (`stop`) w trakcie
    czekania = „Kontynuuj”, żeby wątek nie wisiał – i tak zaraz kończy."""

    def __init__(self, root, ask: Callable = ask_cooling_timeout, stop: Optional[threading.Event] = None,
                 poll_ms: int = 100):
        self.root, self.ask, self.stop, self.poll_ms = root, ask, stop, poll_ms
        self.requests: "queue.Queue[tuple[threading.Event, dict]]" = queue.Queue()
        self.closed = False
        self.root.after(self.poll_ms, self._poll)

    def _poll(self) -> None:
        if self.closed:
            return
        while True:
            try:
                done, answer = self.requests.get_nowait()
            except queue.Empty:
                break
            try:
                answer["value"] = bool(self.ask(self.root))
            finally:
                done.set()
        self.root.after(self.poll_ms, self._poll)

    def close(self) -> None:
        self.closed = True

    def __call__(self) -> bool:
        done, answer = threading.Event(), {"value": False}
        self.requests.put((done, answer))
        while not done.wait(0.2):
            if self.closed or (self.stop is not None and self.stop.is_set()):
                return True
        return answer["value"]
