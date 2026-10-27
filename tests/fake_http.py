"""Fałszywy klient HTTP: odpowiedzi z pamięci, bez sieci. Liczy zapytania."""
from __future__ import annotations

from aibobr.sources.base import SourceError
from aibobr.sources.http import RemoteFile


class FakeHttp:
    def __init__(self, json=None, files=None, denied=(), chunk=1024 * 1024):
        self.json = dict(json or {})       # url → obiekt
        self.files = dict(files or {})     # url → bajty
        self.denied = set(denied)          # url → 401
        self.chunk = chunk
        self.calls: list[str] = []

    def _check(self, url):
        self.calls.append(url)
        if url in self.denied:
            raise SourceError("source.access_denied", url)

    def get_json(self, url):
        self._check(url)
        if url not in self.json:
            raise SourceError("source.not_found", url)
        return self.json[url]

    def get_range(self, url, start, length):
        self._check(url)
        if url not in self.files:
            raise SourceError("source.not_found", url)
        return self.files[url][start:start + length]

    def open_remote(self, url):
        return RemoteFile(self, url, chunk=self.chunk)
