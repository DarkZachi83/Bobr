"""Jedno pole „Wskaż model” → właściwe źródło (odpowiednik transport/factory.py z NIUCHACZA)."""
from __future__ import annotations

import re
from pathlib import Path

from aibobr.sources.base import LoadedModel, SourceError
from aibobr.sources.gguf_reader import load_gguf_file
from aibobr.sources.http import HttpClient
from aibobr.sources.huggingface import HuggingFaceSource, parse_reference
from aibobr.sources.ollama import DEFAULT_HOST, OllamaSource

OLLAMA_PREFIX = "ollama:"

# Strony, które wyglądają na źródło modelu, ale nie mają API (SPEC §4).
_UNSUPPORTED_SITES = re.compile(r"(^|//|\.)(ollama\.com|lmstudio\.ai|jan\.ai)(/|$)", re.I)


def source_kind(text: str) -> str:
    """"ollama", "file" albo "remote" – GUI/CLI pokazują inny komunikat podczas wczytywania."""
    text = text.strip().strip('"').strip("'")
    if text.lower().startswith(OLLAMA_PREFIX):
        return "ollama"
    path = Path(text).expanduser()
    return "file" if path.suffix.lower() == ".gguf" or path.exists() else "remote"


def is_local(text: str) -> bool:
    """Czy wczytanie obędzie się bez internetu (plik albo Ollama na tym komputerze)?"""
    return source_kind(text) != "remote"


def load_model(text: str, client: HttpClient, cache=None,
               ollama_host: str = DEFAULT_HOST) -> LoadedModel:
    """Ścieżka do .gguf, link/identyfikator HF, ollama:NAZWA albo błąd z kluczem i18n.

    „ollama:” to opcja trybu Zaawansowanego (model już zainstalowany w Ollamie)."""
    text = text.strip().strip('"').strip("'")
    if text.lower().startswith(OLLAMA_PREFIX):
        name = text[len(OLLAMA_PREFIX):].strip()
        if not name:
            raise SourceError("source.empty_input")
        return OllamaSource(client, ollama_host).load(name)
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
