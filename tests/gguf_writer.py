"""Minimalny zapis nagłówka GGUF do testów (bez danych tensorów).

Wystarczy do sprawdzenia czytnika: metadane + opisy tensorów, reszta pliku może być
pusta albo wypełniona zerami, żeby udawać rozmiar."""
from __future__ import annotations

import struct
from functools import lru_cache
from typing import Any

_FMT = {"u8": (0, "<B"), "i8": (1, "<b"), "u16": (2, "<H"), "i16": (3, "<h"),
        "u32": (4, "<I"), "i32": (5, "<i"), "f32": (6, "<f"), "bool": (7, "<?"),
        "u64": (10, "<Q"), "i64": (11, "<q"), "f64": (12, "<d")}


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _typed(value: Any, kind: str) -> bytes:
    """kind: 'str', 'u32', …, albo 'arr:u32' / 'arr:str'."""
    if kind == "str":
        return struct.pack("<I", 8) + _string(value)
    if kind.startswith("arr:"):
        inner = kind[4:]
        if inner == "str":
            body = b"".join(_string(v) for v in value)
            return struct.pack("<IIQ", 9, 8, len(value)) + body
        code, fmt = _FMT[inner]
        return struct.pack("<IIQ", 9, code, len(value)) + b"".join(struct.pack(fmt, v) for v in value)
    code, fmt = _FMT[kind]
    return struct.pack("<I", code) + struct.pack(fmt, value)


def gguf_bytes(metadata: list[tuple[str, Any, str]],
               tensors: list[tuple[str, tuple[int, ...]]] = (),
               version: int = 3, magic: int = 0x46554747) -> bytes:
    out = struct.pack("<IIQQ", magic, version, len(tensors), len(metadata))
    for key, value, kind in metadata:
        out += _string(key) + _typed(value, kind)
    for name, shape in tensors:
        out += _string(name) + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", d) for d in shape)
        out += struct.pack("<IQ", 0, 0)
    return out


@lru_cache(maxsize=None)
def llama_like(arch="llama", layers=32, heads=32, kv_heads=8, embed=4096, ctx=131072,
               file_type=15, name="Test Model 8B", vocab=128256, extra=(), tokens=None) -> bytes:
    """Nagłówek podobny do Llama 3.1 8B: duży słownik, tensory jednej warstwy × layers."""
    md = [
        ("general.architecture", arch, "str"),
        ("general.name", name, "str"),
        ("general.file_type", file_type, "u32"),
        (f"{arch}.block_count", layers, "u32"),
        (f"{arch}.context_length", ctx, "u32"),
        (f"{arch}.embedding_length", embed, "u32"),
        (f"{arch}.attention.head_count", heads, "u32"),
        (f"{arch}.attention.head_count_kv", kv_heads, "u32"),
        ("tokenizer.ggml.tokens", list(tokens or [f"t{i}" for i in range(vocab)]), "arr:str"),
        ("tokenizer.ggml.scores", [0.0] * len(tokens or range(vocab)), "arr:f32"),
        *extra,
    ]
    head_dim = embed // heads
    tensors = [("token_embd.weight", (embed, vocab))]
    for i in range(layers):
        tensors += [
            (f"blk.{i}.attn_q.weight", (embed, heads * head_dim)),
            (f"blk.{i}.attn_k.weight", (embed, kv_heads * head_dim)),
            (f"blk.{i}.attn_v.weight", (embed, kv_heads * head_dim)),
            (f"blk.{i}.attn_output.weight", (heads * head_dim, embed)),
            (f"blk.{i}.ffn_up.weight", (embed, 14336)),
            (f"blk.{i}.ffn_gate.weight", (embed, 14336)),
            (f"blk.{i}.ffn_down.weight", (14336, embed)),
            (f"blk.{i}.attn_norm.weight", (embed,)),
            (f"blk.{i}.ffn_norm.weight", (embed,)),
        ]
    tensors += [("output_norm.weight", (embed,)), ("output.weight", (embed, vocab))]
    return gguf_bytes(md, tensors)
