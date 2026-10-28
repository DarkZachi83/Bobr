"""Znane miejsca, w których programy trzymają modele (Linux i Windows).

Zwracamy tylko foldery, które istnieją. Ścieżki według dokumentacji i domyślnych
ustawień programów; zmienne środowiskowe (OLLAMA_MODELS, HF_HOME, …) mają pierwszeństwo.
Stable Diffusion WebUI, ComfyUI i podobne leżą tam, gdzie je pobrano – te znajduje
dopiero „Przeszukaj folder…”.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional


def known_locations(platform: str, env: Optional[Mapping[str, str]] = None,
                    home: Optional[Path] = None) -> list[tuple[str, Path]]:
    env = os.environ if env is None else env
    home = home or Path.home()
    windows = platform.startswith("win")
    local = Path(env["LOCALAPPDATA"]) if windows and env.get("LOCALAPPDATA") else home / "AppData" / "Local"
    xdg_cache = Path(env.get("XDG_CACHE_HOME") or home / ".cache")
    xdg_data = Path(env.get("XDG_DATA_HOME") or home / ".local" / "share")

    places: list[tuple[str, Path]] = []

    def add(app: str, path) -> None:
        if path:
            places.append((app, Path(path).expanduser()))

    # Ollama: folder z podfolderami manifests/ i blobs/
    add("Ollama", env.get("OLLAMA_MODELS"))
    add("Ollama", home / ".ollama" / "models")
    if not windows:                                   # Ollama jako usługa systemowa
        add("Ollama", "/usr/share/ollama/.ollama/models")
        add("Ollama", "/var/lib/ollama/.ollama/models")
        add("Ollama", "/var/lib/ollama")

    add("LM Studio", home / ".lmstudio" / "models")
    add("LM Studio", home / ".cache" / "lm-studio" / "models")   # starsze wersje

    hub = env.get("HF_HUB_CACHE") or (Path(env["HF_HOME"]) / "hub" if env.get("HF_HOME") else None)
    add("Hugging Face", hub)
    add("Hugging Face", (home / ".cache" if windows else xdg_cache) / "huggingface" / "hub")

    add("llama.cpp", env.get("LLAMA_CACHE"))
    add("llama.cpp", local / "llama.cpp" if windows else xdg_cache / "llama.cpp")

    add("GPT4All", local / "nomic.ai" / "GPT4All" if windows else xdg_data / "nomic.ai" / "GPT4All")

    result, seen = [], set()
    for app, path in places:
        try:
            key = path.resolve()
            if key in seen or not path.is_dir():
                continue
        except OSError:
            continue
        seen.add(key)
        result.append((app, path))
    return result
