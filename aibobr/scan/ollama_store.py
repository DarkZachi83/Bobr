"""Modele Ollamy prosto z dysku – bez uruchamiania Ollamy.

Układ folderu (ollama/server/manifest.go):
    models/manifests/<rejestr>/<przestrzeń>/<model>/<tag>   – JSON z warstwami
    models/blobs/sha256-<skrót>                              – zawartość warstw
Warstwa „application/vnd.ollama.image.model” to zwykły plik GGUF.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_REGISTRY = "registry.ollama.ai"
MODEL_LAYER = "application/vnd.ollama.image.model"


def is_store(path: Path) -> bool:
    return (path / "manifests").is_dir() and (path / "blobs").is_dir()


def model_name(manifest: Path, manifests_root: Path) -> Optional[str]:
    parts = manifest.relative_to(manifests_root).parts
    if len(parts) < 3:
        return None
    *repo, tag = parts
    registry, rest = repo[0], repo[1:]
    if registry == DEFAULT_REGISTRY:
        if rest and rest[0] == "library":
            rest = rest[1:]
        return f"{'/'.join(rest)}:{tag}"
    return f"{registry}/{'/'.join(rest)}:{tag}"


def blob_path(store: Path, digest: str) -> Path:
    return store / "blobs" / digest.replace(":", "-")


def entries(store: Path) -> Iterator[tuple[str, Path, int]]:
    """(nazwa jak w `ollama list`, plik GGUF z wagami, rozmiar warstw razem)."""
    root = store / "manifests"
    for manifest in sorted(p for p in root.rglob("*") if p.is_file()):
        name = model_name(manifest, root)
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        layers = data.get("layers") if isinstance(data, dict) else None
        if not name or not isinstance(layers, list):
            continue
        model = next((l for l in layers if isinstance(l, dict) and l.get("mediaType") == MODEL_LAYER), None)
        if not model or not isinstance(model.get("digest"), str):
            continue
        size = sum(int(l.get("size") or 0) for l in layers if isinstance(l, dict))
        yield name, blob_path(store, model["digest"]), size


def find_blob(name: str, stores: list[Path]) -> Optional[Path]:
    """Plik GGUF modelu o nazwie z `ollama list` (bez „:latest” też trafia)."""
    wanted = {name, f"{name}:latest"} if ":" not in name.rsplit("/", 1)[-1] else {name}
    for store in stores:
        if not is_store(store):
            continue
        for entry, blob, _ in entries(store):
            if entry in wanted and blob.is_file():
                return blob
    return None
