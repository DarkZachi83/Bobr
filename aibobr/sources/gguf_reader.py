"""Czytnik nagłówka GGUF – bez numpy i bez pakietu `gguf` (SPEC §13).

Czytamy tylko nagłówek: metadane i opisy tensorów. Same wagi (gigabajty) pomijamy,
więc ten sam kod działa na pliku lokalnym i na pliku zdalnym czytanym kawałkami
(zapytania HTTP Range, zob. http.RemoteFile).

Format według referencyjnej implementacji llama.cpp (gguf-py/gguf/constants.py,
gguf_reader.py), wersje 2 i 3, little-endian:

    uint32 magic "GGUF" · uint32 wersja · uint64 liczba tensorów · uint64 liczba metadanych
    metadane:  string klucz · uint32 typ · wartość
    tensory:   string nazwa · uint32 n_dims · uint64[n_dims] wymiary · uint32 typ · uint64 offset
    string = uint64 długość + bajty UTF-8
"""
from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from aibobr.domain.enums import ModelSource
from aibobr.sources.base import (
    SPLIT_RE, ArchInfo, LoadedModel, SourceError, Variant, count_swa_layers, quant_from_filename,
)

GGUF_MAGIC = 0x46554747  # b"GGUF" czytane jako little-endian uint32

# GGUFValueType → format struct (typy skalarne)
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9

# Tablice dłuższe niż to (słownik tokenizera: 100–250 tys. pozycji) pomijamy bez
# zapamiętywania – potrzebne nam tablice (np. głowice KV per warstwa) są krótkie.
_MAX_KEPT_ARRAY = 4096
# Z długiej tablicy napisów (słownik) zapamiętujemy koniec – tam modele mowy dopisują
# tokeny dźwięku (Orpheus: <custom_token_…>, OuteTTS: <|c1_…|>). Klucz: "<klucz>#tail".
TAIL_SUFFIX = "#tail"
_TAIL = 4096
_MAX_COUNT = 10_000_000      # zabezpieczenie przed uszkodzonym plikiem
_MAX_STRING = 64 * 1024 * 1024

# llama_ftype (include/llama.h) → nazwa kwantyzacji
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}


class ByteSource(Protocol):
    """Coś, z czego czytamy kolejno bajty: plik lokalny albo zdalny (RemoteFile)."""

    def read(self, n: int) -> bytes: ...
    def skip(self, n: int) -> None: ...


class LocalFile:
    def __init__(self, f):
        self._f = f

    def read(self, n: int) -> bytes:
        data = self._f.read(n)
        if len(data) != n:
            raise SourceError("source.bad_file", "unexpected end of file")
        return data

    def skip(self, n: int) -> None:
        self._f.seek(n, os.SEEK_CUR)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]

    @property
    def n_elements(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class GgufHeader:
    version: int
    metadata: dict[str, Any]
    tensors: tuple[TensorInfo, ...]


class _Parser:
    def __init__(self, src: ByteSource):
        self.src = src

    def _u(self, fmt: str):
        return struct.unpack(fmt, self.src.read(struct.calcsize(fmt)))[0]

    def _string(self) -> str:
        n = self._u("<Q")
        if n > _MAX_STRING:
            raise SourceError("source.bad_file", f"string too long ({n})")
        return self.src.read(n).decode("utf-8", errors="replace")

    def _value(self, vtype: int, tail: Optional[list] = None) -> Any:
        if vtype in _SCALAR:
            return self._u(_SCALAR[vtype])
        if vtype == _STRING:
            return self._string()
        if vtype == _ARRAY:
            etype, count = self._u("<I"), self._u("<Q")
            if count > _MAX_COUNT:
                raise SourceError("source.bad_file", f"array too long ({count})")
            keep = count <= _MAX_KEPT_ARRAY
            if etype in _SCALAR and not keep:
                self.src.skip(count * struct.calcsize(_SCALAR[etype]))
                return None
            if etype == _STRING and not keep:
                start = count - _TAIL           # słownik tokenizera: przewijamy, koniec czytamy
                for i in range(count):
                    n = self._u("<Q")
                    if tail is not None and i >= start and n <= 256:
                        tail.append(self.src.read(n).decode("utf-8", errors="replace"))
                    else:
                        self.src.skip(n)
                return None
            items = [self._value(etype) for _ in range(count)]
            return items if keep else None
        raise SourceError("source.bad_file", f"unknown value type {vtype}")

    def header(self) -> GgufHeader:
        if self._u("<I") != GGUF_MAGIC:
            raise SourceError("source.not_gguf")
        version = self._u("<I")
        if version not in (2, 3):
            # 1 = bardzo stary format; ogromna liczba = plik big-endian
            raise SourceError("source.gguf_version", str(version))
        n_tensors, n_kv = self._u("<Q"), self._u("<Q")
        if n_tensors > _MAX_COUNT or n_kv > _MAX_COUNT:
            raise SourceError("source.bad_file", "header counts out of range")
        metadata = {}
        for _ in range(n_kv):
            key = self._string()
            tail: list = []
            metadata[key] = self._value(self._u("<I"), tail)
            if tail:
                metadata[key + TAIL_SUFFIX] = tail
        tensors = []
        for _ in range(n_tensors):
            name = self._string()
            n_dims = self._u("<I")
            if n_dims > 8:
                raise SourceError("source.bad_file", f"tensor {name}: {n_dims} dims")
            shape = tuple(self._u("<Q") for _ in range(n_dims))
            self._u("<I")  # typ tensora
            self._u("<Q")  # offset danych
            tensors.append(TensorInfo(name, shape))
        return GgufHeader(version, metadata, tuple(tensors))


def read_header(src: ByteSource) -> GgufHeader:
    try:
        return _Parser(src).header()
    except struct.error as e:
        raise SourceError("source.bad_file", str(e)) from e


# --------------------------------------------------------------------------- interpretacja

# Domyślny wzór warstw z oknem przesuwnym, gdy plik go nie podaje – jak w llama.cpp
# (src/models/*.cpp: load_swa_pattern). Tylko architektury, które naprawdę używają okna,
# i tylko gdy plik ma klucz {arch}.attention.sliding_window > 0. Dla reszty liczymy pełny
# kontekst we wszystkich warstwach (wynik zawyżony, czyli bezpieczny).
SWA_DEFAULT_PATTERN = {"gemma2": 2, "gemma3": 6, "gpt-oss": 2, "cohere2": 4, "exaone4": 4}

def _int_or_max(value: Any) -> Optional[int]:
    """Wartość per warstwa bywa tablicą – bierzemy maksimum (ostrożnie)."""
    if isinstance(value, list):
        nums = [v for v in value if isinstance(v, (int, float))]
        return int(max(nums)) if nums else None
    return int(value) if isinstance(value, (int, float)) else None


_DIGITS = re.compile(r"\d+")


def token_pattern(tokens) -> Optional[str]:
    """Najczęstszy wzór tokenów na końcu słownika (cyfry → „#”), gdy jest wyraźny.

    Modele mowy dopisują do słownika tysiące tokenów dźwięku o jednym wzorze:
    Orpheus „<custom_token_#>”, OuteTTS 1.0 „<|c#_#|>” (c1_…, c2_…). Zwykły model ma
    tam mieszankę słów – wtedy None."""
    if not isinstance(tokens, list) or not tokens:
        return None
    tail = [t for t in tokens[-_TAIL:] if isinstance(t, str)]
    counts: dict[str, int] = {}
    for tok in tail:
        if any(c.isdigit() for c in tok):
            key = _DIGITS.sub("#", tok)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    best, n = max(counts.items(), key=lambda kv: kv[1])
    # Tokeny dźwięku to zawsze tokeny specjalne (<…>, <|…|>, […]); liczby w zwykłym
    # słowniku („Ġ2024”, „000”…) się nie liczą.
    special = any(c in best for c in "<[|")
    return best if special and n >= max(256, len(tail) // 8) else None


def arch_from_headers(headers: list[GgufHeader]) -> ArchInfo:
    """Architektura z nagłówka (pierwszej części) + tensory ze wszystkich części pliku."""
    md = headers[0].metadata
    arch = md.get("general.architecture")
    if not isinstance(arch, str):
        raise SourceError("source.bad_file", "no general.architecture")

    def g(key: str) -> Any:
        return md.get(f"{arch}.{key}")

    layers = _int_or_max(g("block_count"))
    heads = _int_or_max(g("attention.head_count"))
    kv_heads = _int_or_max(g("attention.head_count_kv")) or heads
    embed = _int_or_max(g("embedding_length"))
    k_len = _int_or_max(g("attention.key_length")) or (embed // heads if embed and heads else None)
    v_len = _int_or_max(g("attention.value_length")) or k_len
    head_dim = math.ceil((k_len + v_len) / 2) if k_len and v_len else None

    # MLA (DeepSeek i pochodne): cache trzyma jeden skompresowany wektor na warstwę,
    # a nie K i V dla każdej głowicy. 2 × 1 × ceil(x/2) ≈ x elementów na warstwę.
    kv_lora = _int_or_max(g("attention.kv_lora_rank"))
    if kv_lora:
        rope = _int_or_max(g("rope.dimension_count")) or 0
        kv_heads, head_dim = 1, math.ceil((kv_lora + rope) / 2)

    experts = _int_or_max(g("expert_count"))
    used = _int_or_max(g("expert_used_count"))

    tensors = [t for h in headers for t in h.tensors]
    total = sum(t.n_elements for t in tensors) or None
    embd = sum(t.n_elements for t in tensors if t.name.startswith("token_embd.")) or None
    tied = (not any(t.name == "output.weight" for t in tensors)) if embd else None
    active = None
    if total and experts and experts > 1 and used:
        # Tensory ekspertów są złożone w jeden ("…_exps"); aktywna jest ich część used/experts.
        expert_params = sum(t.n_elements for t in tensors if "_exps" in t.name)
        active = total - expert_params * (1 - used / experts)
        # Tablicy embedding nie liczymy (na token czytany jest jeden jej wiersz) – chyba że
        # jest też wyjściem modelu (brak output.weight = wagi wspólne). Tak liczy OpenAI
        # dla gpt-oss (3,6B aktywnych z 20,9B).
        if tied is False:
            active -= sum(t.n_elements for t in tensors if t.name.startswith("token_embd."))
        active = int(active)

    window = _int_or_max(g("attention.sliding_window"))
    swa_layers = None
    if window and layers:
        pattern = g("attention.sliding_window_pattern")
        if pattern is None:
            pattern = SWA_DEFAULT_PATTERN.get(arch)
        if pattern is not None:
            swa_layers = count_swa_layers(layers, pattern)
    if not swa_layers:
        window = None

    return ArchInfo(
        sliding_window=window, swa_layers=swa_layers,
        params_total=total, n_layers=layers, n_kv_heads=kv_heads, head_dim=head_dim,
        max_context=_int_or_max(g("context_length")),
        n_experts=experts if experts and experts > 1 else None,
        n_experts_used=used if experts and experts > 1 else None,
        params_active=active, architecture=arch, embd_params=embd, tied_embeddings=tied,
        token_pattern=token_pattern(md.get("tokenizer.ggml.tokens" + TAIL_SUFFIX)
                                    or md.get("tokenizer.ggml.tokens")),
    )


def quant_from_header(header: GgufHeader, filename: str = "") -> str:
    ftype = header.metadata.get("general.file_type")
    if isinstance(ftype, int) and ftype in FILE_TYPES:
        return FILE_TYPES[ftype]
    return quant_from_filename(filename) or "?"


def display_name(header: GgufHeader, filename: str) -> str:
    name = header.metadata.get("general.name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    match = SPLIT_RE.match(filename)
    return match.group("base") if match else filename.removesuffix(".gguf")


# --------------------------------------------------------------------------- plik lokalny

def file_size(path: Path) -> int:
    """Rozmiar pliku – osobna funkcja, żeby testy mogły udawać pliki wielogigabajtowe
    bez tworzenia ich na dysku (na Windowsie plik „rzadki” zajmuje prawdziwe miejsce)."""
    return path.stat().st_size


def split_parts(path: Path) -> list[Path]:
    """Wszystkie części pliku dzielonego (…-00001-of-00003.gguf) albo [path]."""
    match = SPLIT_RE.match(path.name)
    if not match:
        return [path]
    count = int(match.group("count"))
    parts = [path.with_name(f"{match.group('base')}-{i:05d}-of-{count:05d}.gguf")
             for i in range(1, count + 1)]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise SourceError("source.split_incomplete", ", ".join(missing))
    return parts


def load_gguf_file(path: str | os.PathLike) -> LoadedModel:
    """Model z lokalnego pliku: dokładny rozmiar i architektura, bez internetu."""
    path = Path(path)
    if not path.is_file():
        raise SourceError("source.file_missing", str(path))
    parts = split_parts(path)
    headers = []
    for part in parts:
        with open(part, "rb") as f:
            headers.append(read_header(LocalFile(f)))
    quant = quant_from_header(headers[0], parts[0].name)
    size = sum(file_size(p) for p in parts)
    return LoadedModel(
        name=display_name(headers[0], parts[0].name),
        source=ModelSource.GGUF_FILE,
        arch=arch_from_headers(headers),
        variants=(Variant(quant=quant, size=size, files=tuple(str(p) for p in parts)),),
        origin=str(parts[0]),
        preferred_quant=quant,
    )
