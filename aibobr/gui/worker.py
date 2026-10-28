"""Zadania w tle: pobieranie z Hugging Face i skan nie mogą zamrozić okna.

Wątek wykonuje funkcję, wynik trafia do kolejki, a okno odbiera go co 50 ms
(tkinter wolno dotykać tylko z głównego wątku). Nowsze zlecenie tego samego rodzaju
unieważnia starsze – wynik spóźnionego pobrania nie nadpisze nowszego.

Odśmiecanie pamięci: gdy automatyczne odśmiecanie Pythona ruszy w wątku w tle, potrafi
usunąć stare obiekty tkintera (np. zmienne okna po zmianie języka) – a Tk wolno usuwać
tylko z wątku okna. Na Windowsie kończy się to „Tcl_AsyncDelete: async handler deleted
by the wrong thread” i zamknięciem programu (złapane w testach na GitHubie). Dlatego przed
startem wątku sprzątamy w wątku okna, a na czas pracy wątków automatyczne odśmiecanie
jest wstrzymane (trwają sekundy) i wraca, gdy skończy się ostatni.
"""
from __future__ import annotations

import gc
import logging
import queue
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, root, poll_ms: int = 50):
        self.root = root
        self.poll_ms = poll_ms
        self.results: "queue.Queue[tuple]" = queue.Queue()
        self.latest: dict[str, int] = {}
        self._counter = 0
        self._polling = False
        self.running = 0                  # wątki, których wyniku jeszcze nie odebraliśmy

    def busy(self, kind: str) -> bool:
        return kind in self.latest

    def run(self, kind: str, fn: Callable[[], Any], on_done: Callable[[Any], None],
            on_error: Callable[[BaseException], None]) -> None:
        self._counter += 1
        ticket = self._counter
        self.latest[kind] = ticket

        def target():
            try:
                self.results.put((kind, ticket, True, fn(), on_done, on_error))
            except BaseException as e:      # przekazujemy do okna, nie gubimy w wątku
                log.info("background %s failed: %r", kind, e)
                self.results.put((kind, ticket, False, e, on_done, on_error))

        if self.running == 0:
            gc.collect()                  # w wątku okna – zanim ruszy wątek w tle
            gc.disable()
        self.running += 1
        threading.Thread(target=target, daemon=True).start()
        if not self._polling:
            self._polling = True
            self.root.after(self.poll_ms, self._poll)

    def _poll(self) -> None:
        while True:
            try:
                kind, ticket, ok, value, on_done, on_error = self.results.get_nowait()
            except queue.Empty:
                break
            self.running -= 1
            if self.running == 0:
                gc.enable()
            if self.latest.get(kind) != ticket:
                continue                                  # unieważnione nowszym zleceniem
            del self.latest[kind]
            (on_done if ok else on_error)(value)
        if self.running:                                  # także unieważnione, ale wciąż pracujące
            self.root.after(self.poll_ms, self._poll)
        else:
            self._polling = False
