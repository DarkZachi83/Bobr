"""Nagłówek pliku .safetensors – nazwy, typy i rozmiary tensorów bez czytania wag.

Format (huggingface/safetensors): uint64 LE długość nagłówka, potem JSON
{"nazwa": {"dtype": "F16", "shape": [...], "data_offsets": [początek, koniec]}, "__metadata__": {...}}.
Ten sam kod czyta plik lokalny i zdalny (ByteSource z gguf_reader – HTTP Range).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Optional

from aibobr.sources.base import SourceError

MAX_HEADER = 100 * 1024**2           # nagłówki dużych modeli mają kilka MB


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class SafetensorsHeader:
    tensors: tuple[Tensor, ...]
    metadata: dict

    @property
    def total(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def main_dtype(self) -> Optional[str]:
        """Typ, w którym zapisana jest większość bajtów (np. "F16", "BF16", "F8_E4M3")."""
        by: dict[str, int] = {}
        for t in self.tensors:
            by[t.dtype] = by.get(t.dtype, 0) + t.nbytes
        return max(by, key=by.get) if by else None


def read_header(src) -> SafetensorsHeader:
    try:
        (length,) = struct.unpack("<Q", src.read(8))
    except struct.error as e:
        raise SourceError("source.bad_file", str(e)) from e
    if not 2 <= length <= MAX_HEADER:
        raise SourceError("source.bad_file", f"safetensors header length {length}")
    try:
        data = json.loads(src.read(length))
    except ValueError as e:
        raise SourceError("source.bad_file", f"safetensors header: {e}") from e
    if not isinstance(data, dict):
        raise SourceError("source.bad_file", "safetensors header is not an object")
    tensors = []
    for name, info in data.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        offsets = info.get("data_offsets") or [0, 0]
        try:
            nbytes = int(offsets[1]) - int(offsets[0])
        except (TypeError, ValueError, IndexError):
            nbytes = 0
        tensors.append(Tensor(name, str(info.get("dtype", "?")), max(0, nbytes)))
    meta = data.get("__metadata__") if isinstance(data.get("__metadata__"), dict) else {}
    return SafetensorsHeader(tuple(tensors), meta)


# safetensors dtype → etykieta „kwantyzacji” pokazywana użytkownikowi
DTYPE_LABEL = {"F32": "FP32", "F16": "FP16", "BF16": "BF16", "F8_E4M3": "FP8", "F8_E5M2": "FP8"}
