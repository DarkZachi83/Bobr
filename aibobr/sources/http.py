"""Klient HTTP na urllib (bez `requests`, SPEC §13).

Dwie rzeczy, które robimy inaczej niż domyślny urllib:
  * Token HF wysyłamy tylko do huggingface.co. Pliki są przekierowywane na serwery CDN
    innego hosta i tam nagłówek Authorization jest usuwany.
  * RemoteFile czyta plik zdalny kawałkami (HTTP Range), więc z wielogigabajtowego GGUF
    pobieramy tylko nagłówek – zwykle kilka MB. Po pierwszym przekierowaniu zapamiętuje
    adres serwera z plikiem, a kawałki rosną (1 → 2 → 4 … 16 MB), żeby nagłówek ze sporym
    słownikiem tokenizera przyszedł w kilku zapytaniach, a nie kilkunastu.
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Optional
from urllib.parse import urlparse

from aibobr import __version__
from aibobr.sources.base import SourceError

USER_AGENT = f"AI-Bobr/{__version__} (+https://github.com/DarkZachi83/Bobr)"
CHUNK = 1024 * 1024
MAX_CHUNK = 16 * 1024 * 1024


class _StripAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(newurl).hostname != urlparse(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


class HttpClient:
    def __init__(self, token: Optional[str] = None, timeout: float = 20.0):
        self.token = token or None
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_StripAuthOnHostChange())

    def _request(self, url: str, headers: Optional[dict] = None, data: Optional[bytes] = None,
                 timeout: Optional[float] = None):
        req = urllib.request.Request(url, data=data,
                                     headers={"User-Agent": USER_AGENT, **(headers or {})})
        if self.token and urlparse(url).hostname in ("huggingface.co", "hf.co"):
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            return self._opener.open(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            code = {401: "source.access_denied", 403: "source.access_denied",
                    404: "source.not_found", 416: "source.bad_file"}.get(e.code, "source.http_error")
            raise SourceError(code, f"HTTP {e.code} {url}") from e
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            raise SourceError("source.network", f"{url}: {e}") from e

    def get_json(self, url: str) -> Any:
        with self._request(url) as resp:
            try:
                return json.loads(resp.read().decode("utf-8"))
            except ValueError as e:
                raise SourceError("source.http_error", f"bad JSON from {url}") from e

    def post_json(self, url: str, payload: dict, timeout: Optional[float] = None) -> Any:
        body = json.dumps(payload).encode("utf-8")
        with self._request(url, {"Content-Type": "application/json"}, body, timeout) as resp:
            try:
                return json.loads(resp.read().decode("utf-8") or "{}")
            except ValueError as e:
                raise SourceError("source.http_error", f"bad JSON from {url}") from e

    def get_range(self, url: str, start: int, length: int) -> tuple[bytes, str]:
        """Bajty [start, start+length) – mniej, jeśli plik się wcześniej kończy –
        oraz adres, pod którym je ostatecznie podano (po przekierowaniach)."""
        with self._request(url, {"Range": f"bytes={start}-{start + length - 1}"}) as resp:
            final = resp.geturl() or url
            if resp.status == 206:
                return resp.read(), final
            if resp.status == 200 and start == 0:
                # Serwer zignorował Range: czytamy tylko tyle, ile trzeba, i zamykamy połączenie.
                return resp.read(length), final
            raise SourceError("source.range_unsupported", url)

    def open_remote(self, url: str) -> "RemoteFile":
        return RemoteFile(self, url)


class RemoteFile:
    """Plik zdalny czytany po kolei, rosnącymi kawałkami (implementuje ByteSource)."""

    def __init__(self, client: HttpClient, url: str, chunk: int = CHUNK, max_chunk: int = MAX_CHUNK):
        self._client, self._url = client, url
        self._chunk, self._max_chunk = chunk, max_chunk
        self._pos = 0
        self._buf_start = 0
        self._buf = b""
        self.requests = 0                      # do testów i logu

    def skip(self, n: int) -> None:
        self._pos += n

    def read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            offset = self._pos - self._buf_start
            if 0 <= offset < len(self._buf):
                piece = self._buf[offset: offset + n - len(out)]
            else:
                self._buf_start = self._pos
                self._buf, self._url = self._client.get_range(
                    self._url, self._pos, max(self._chunk, n - len(out)))
                self._chunk = min(self._chunk * 2, self._max_chunk)
                self.requests += 1
                if not self._buf:
                    raise SourceError("source.bad_file", "unexpected end of remote file")
                continue
            out += piece
            self._pos += len(piece)
        return bytes(out)
