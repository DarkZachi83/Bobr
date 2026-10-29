"""Jedno pole „Wskaż model” → właściwe źródło (odpowiednik transport/factory.py z NIUCHACZA)."""
from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

from aibobr.sources.base import LoadedModel, SourceError
from aibobr.sources.gguf_reader import load_gguf_file
from aibobr.sources.http import HttpClient
from aibobr.sources.huggingface import HuggingFaceSource, load_transformers_folder, parse_reference
from aibobr.sources.image import load_diffusers_folder, load_safetensors_file
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
    return "file" if path.suffix.lower() in (".gguf", ".safetensors") or path.exists() else "remote"


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
        try:
            return OllamaSource(client, ollama_host).load(name)
        except SourceError as e:
            if e.code != "ollama.not_running":
                raise
            # Ollama wyłączona – model i tak jest na dysku (znaleziony przy szukaniu modeli).
            from aibobr.scan.locations import known_locations
            from aibobr.scan.ollama_store import find_blob
            blob = find_blob(name, [p for app, p in known_locations(sys.platform) if app == "Ollama"])
            if blob is None:
                raise
            return replace(load_gguf_file(blob), name=name, origin=f"{OLLAMA_PREFIX}{name}")
    # Wklejone całe polecenie: "ollama run hf.co/bartowski/…:Q4_K_M"
    text = re.sub(r"^ollama\s+(run|pull)\s+", "", text, flags=re.I)
    if not text:
        raise SourceError("source.empty_input")

    path = Path(text).expanduser()
    if path.suffix.lower() in (".gguf", ".safetensors") or path.exists():
        if path.is_dir():
            if (path / "model_index.json").is_file():
                return load_diffusers_folder(path)           # model obrazów (etap 10)
            if (path / "config.json").is_file():
                return load_transformers_folder(path)
            raise SourceError("source.is_directory", str(path))
        if path.suffix.lower() == ".safetensors":
            return load_safetensors_file(path)
        return load_gguf_file(path)

    if _UNSUPPORTED_SITES.search(text):
        raise SourceError("source.unsupported_site", text)

    ref = parse_reference(text)
    if ref is None:
        raise SourceError("source.unrecognized_input", text)
    return HuggingFaceSource(client, cache).load(ref)
