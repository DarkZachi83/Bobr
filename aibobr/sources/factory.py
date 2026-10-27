"""Jedno pole „Wskaż model” → właściwe źródło (odpowiednik transport/factory.py z NIUCHACZA)."""
from __future__ import annotations

import re
from pathlib import Path

from aibobr.sources.base import LoadedModel, SourceError
from aibobr.sources.gguf_reader import load_gguf_file
from aibobr.sources.http import HttpClient
from aibobr.sources.huggingface import HuggingFaceSource, parse_reference

# Strony, które wyglądają na źródło modelu, ale nie mają API (SPEC §4).
_UNSUPPORTED_SITES = re.compile(r"(^|//|\.)(ollama\.com|lmstudio\.ai|jan\.ai)(/|$)", re.I)


def is_local(text: str) -> bool:
    """Czy to ścieżka do pliku (a nie link)? GUI/CLI pokazują wtedy inny komunikat."""
    text = text.strip().strip('"').strip("'")
    path = Path(text).expanduser()
    return path.suffix.lower() == ".gguf" or path.exists()


def load_model(text: str, client: HttpClient, cache=None) -> LoadedModel:
    """Ścieżka do .gguf, link/identyfikator HF albo błąd z kluczem i18n."""
    text = text.strip().strip('"').strip("'")
    # Wklejone całe polecenie: "ollama run hf.co/bartowski/…:Q4_K_M"
    text = re.sub(r"^ollama\s+(run|pull)\s+", "", text, flags=re.I)
    if not text:
        raise SourceError("source.empty_input")

    path = Path(text).expanduser()
    if path.suffix.lower() == ".gguf" or path.exists():
        if path.is_dir():
            raise SourceError("source.is_directory", str(path))
        return load_gguf_file(path)

    if _UNSUPPORTED_SITES.search(text):
        raise SourceError("source.unsupported_site", text)

    ref = parse_reference(text)
    if ref is None:
        raise SourceError("source.unrecognized_input", text)
    return HuggingFaceSource(client, cache).load(ref)
