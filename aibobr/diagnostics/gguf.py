"""Odczyt metadanych z nagłówka pliku GGUF – tylko klucze, bez wag; tylko do odczytu.

Po co: Ollama w /api/show nie zawsze podaje szablon czatu z pliku GGUF (`tokenizer.chat_template`), a to on
mówi, czy autorzy modelu podają token początku. Przykład z 12.11.2026: Bielik-Minitron z hf.co ma
`add_bos_token = false`, ale szablon z pliku GGUF zaczyna się od „{{bos_token}}” – token początku dodaje szablon,
nie tokenizer. Plik modelu wskazuje linia FROM w „modelfile” z /api/show (lokalna Ollama).

Format GGUF: „GGUF”, wersja (u32), liczba tensorów (u64), liczba kluczy (u64), potem klucze: nazwa (u64 długość +
UTF-8), typ (u32), wartość. Tablice (np. słownik tokenów) pomijamy bez wczytywania do pamięci.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Iterable, Optional

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
MAX_STRING = 4_000_000
MAX_KEYS = 100_000


class GgufError(ValueError):
    pass


def _read(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GgufError("truncated")
    return data


def _u64(f) -> int:
    return struct.unpack("<Q", _read(f, 8))[0]


def _string(f, keep: bool = True) -> Optional[str]:
    n = _u64(f)
    if n > MAX_STRING:
        raise GgufError("string too long")
    if not keep:
        f.seek(n, 1)
        return None
    return _read(f, n).decode("utf-8", errors="replace")


def _value(f, kind: int, keep: bool):
    if kind in SCALARS:
        fmt = SCALARS[kind]
        value = struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]
        return value if keep else None
    if kind == STRING:
        return _string(f, keep)
    if kind == ARRAY:
        item = struct.unpack("<I", _read(f, 4))[0]
        count = _u64(f)
        if item in SCALARS:
            f.seek(struct.calcsize(SCALARS[item]) * count, 1)
        elif item == STRING:
            for _ in range(count):
                _string(f, keep=False)
        else:
            for _ in range(count):
                _value(f, item, keep=False)
        return None
    raise GgufError(f"unknown type {kind}")


def read_keys(path: Path, wanted: Iterable[str]) -> dict:
    """Wybrane klucze (bez tablic) z nagłówka; czyta tylko do ostatniego potrzebnego klucza."""
    wanted = set(wanted)
    found: dict = {}
    with open(path, "rb") as f:
        if _read(f, 4) != b"GGUF":
            raise GgufError("not a GGUF file")
        version = struct.unpack("<I", _read(f, 4))[0]
        if version < 2:
            raise GgufError("GGUF v1 not supported")
        _u64(f)                                      # liczba tensorów
        keys = _u64(f)
        if keys > MAX_KEYS:
            raise GgufError("too many keys")
        for _ in range(keys):
            name = _string(f)
            kind = struct.unpack("<I", _read(f, 4))[0]
            keep = name in wanted
            value = _value(f, kind, keep)
            if keep:
                found[name] = value
                if len(found) == len(wanted):
                    break
    return found


def model_path(show: dict) -> Optional[Path]:
    """Plik modelu z linii FROM w modelfile z /api/show (tylko gdy istnieje na tym komputerze)."""
    for line in str((show or {}).get("modelfile") or "").splitlines():
        line = line.strip()
        if line.upper().startswith("FROM "):
            candidate = Path(line[5:].strip().strip('"'))
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                return None
    return None
