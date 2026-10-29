"""Rozpoznawanie pliku wag po zawartości – bez wczytywania samych wag.

* GGUF: nagłówek (ten sam czytnik co przy sprawdzaniu modelu).
* safetensors: nagłówek JSON z nazwami tensorów – po nich poznajemy model obrazów,
  LoRA, VAE albo model językowy.
* .ckpt / .pt / .pth / .bin: to pliki pickle – NIGDY ich nie otwieramy (mogą zawierać
  kod). Typ zgadujemy wyłącznie z nazwy folderu (np. Stable-diffusion, Lora).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

from aibobr.scan.model import FoundModel, Kind
from aibobr.sources.base import SPLIT_RE, SourceError
from aibobr.sources import gguf_reader
from aibobr.sources.gguf_reader import (
    TAIL_SUFFIX, LocalFile, display_name, quant_from_header, read_header, token_pattern,
)

MIN_SIZE = 50 * 1024**2              # mniejsze pliki to nie modele (tokenizery, konfiguracja)
PICKLE_SUFFIXES = {".ckpt", ".pt", ".pth", ".bin"}
_MAX_SAFETENSORS_HEADER = 64 * 1024**2

# Nazwy folderów używane przez Stable Diffusion WebUI, ComfyUI, Forge, InvokeAI…
# Porównujemy bez „-”, „_” i spacji: Stable-diffusion (WebUI) = StableDiffusion (Stability Matrix).
_IMAGE_DIRS = {"stablediffusion", "checkpoints", "unet", "diffusionmodels", "sd", "sdxl"}
_LORA_DIRS = {"lora", "loras", "lycoris", "locon"}
_VAE_DIRS = {"vae", "vaeapprox"}
_UPSCALER_DIRS = {"esrgan", "realesrgan", "upscalemodels", "upscalers", "swinir", "dat"}


def is_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def classify_gguf(path: Path, size: int, app: str, name: Optional[str] = None) -> Optional[FoundModel]:
    try:
        with open(path, "rb") as f:
            header = read_header(LocalFile(f))
    except (OSError, SourceError):
        return None
    md = header.metadata
    arch = md.get("general.architecture")
    projector = arch == "clip" or md.get("general.type") == "mmproj" or "mmproj" in path.name.lower()
    fname = path.name
    title = name or display_name(header, fname)
    if projector:
        return FoundModel(title, Kind.ADDON, "GGUF", size, path, app, note="scan.note.projector")
    tokens = md.get("tokenizer.ggml.tokens" + TAIL_SUFFIX) or md.get("tokenizer.ggml.tokens")
    return FoundModel(title, Kind.TEXT, "GGUF", size, path, app,
                      quant=quant_from_header(header, fname), target=str(path),
                      token_pattern=token_pattern(tokens))


def safetensors_keys(path: Path) -> Optional[list[str]]:
    try:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            if not 2 <= length <= _MAX_SAFETENSORS_HEADER:
                return None
            data = json.loads(f.read(length))
    except (OSError, ValueError, struct.error):
        return None
    return [k for k in data if k != "__metadata__"] if isinstance(data, dict) else None


def kind_from_keys(keys: list[str]) -> tuple[Kind, Optional[str]]:
    """Typ modelu po nazwach tensorów safetensors."""
    lower = [k.lower() for k in keys]
    if any("lora" in k for k in lower):
        return Kind.ADDON, "scan.note.lora"
    if any(k.startswith(("model.diffusion_model.", "double_blocks.", "joint_blocks.",
                         "down_blocks.", "transformer_blocks.", "single_transformer_blocks."))
           for k in lower):
        return Kind.IMAGE, None
    if lower and all(k.startswith(("encoder.", "decoder.", "quant_conv", "post_quant_conv",
                                   "first_stage_model.")) for k in lower):
        return Kind.ADDON, "scan.note.vae"
    if any(".layers." in k or k.startswith(("model.embed_tokens", "transformer.h.", "lm_head"))
           for k in lower):
        return Kind.TEXT, None
    return Kind.OTHER, None


def _norm(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def kind_from_folder(path: Path) -> tuple[Kind, Optional[str]]:
    names = {_norm(p.name) for p in path.parents}
    if names & _LORA_DIRS:
        return Kind.ADDON, "scan.note.lora"
    if names & _VAE_DIRS:
        return Kind.ADDON, "scan.note.vae"
    if names & _IMAGE_DIRS:
        return Kind.IMAGE, None
    if names & _UPSCALER_DIRS:
        return Kind.ADDON, "scan.note.upscaler"
    return Kind.OTHER, None


# Przerwane pobrania: Stability Matrix (.smdownload, np. „Unconfirmed 5348838.smdownload”),
# przeglądarki (.crdownload, .part, .download), pamięć HF (.incomplete), Ollama (…-partial).
PARTIAL_SUFFIXES = {".smdownload", ".crdownload", ".part", ".partial", ".download", ".incomplete"}


def is_partial(path: Path) -> bool:
    return path.suffix.lower() in PARTIAL_SUFFIXES or "-partial" in path.name


def partial(path: Path, size: int, app: str) -> FoundModel:
    return FoundModel(path.name, Kind.OTHER, "partial", size, path, app, note="scan.note.partial")


def classify_file(path: Path, size: int, app: str = "folder") -> Optional[FoundModel]:
    """Pojedynczy plik (nie część folderu transformers/diffusers) albo None."""
    if size < MIN_SIZE:
        return None
    if is_partial(path):
        return partial(path, size, app)
    suffix = path.suffix.lower()
    match = SPLIT_RE.match(path.name)
    if match and int(match.group("no")) != 1:
        return None                                   # dalsze części liczy pierwsza
    if suffix == ".gguf" or (not suffix and is_gguf(path)):
        if match:
            size = _split_size(path, match)
        return classify_gguf(path, size, app)
    if suffix == ".safetensors":
        keys = safetensors_keys(path)
        kind, note = kind_from_keys(keys) if keys else (Kind.OTHER, None)
        return FoundModel(path.stem, kind, "safetensors", size, path, app, note=note)
    if suffix in PICKLE_SUFFIXES:
        kind, note = kind_from_folder(path)
        # Bez podpowiedzi z folderu pickle to zwykle nie model (np. pamięć shaderów gier
        # *.bin) – pokazujemy tylko .ckpt, które prawie zawsze jest modelem obrazów.
        if kind == Kind.OTHER:
            if suffix != ".ckpt":
                return None
            kind = Kind.IMAGE
        return FoundModel(path.stem, kind, "ckpt", size, path, app, note=note)
    return None


def _split_size(first: Path, match) -> int:
    total = 0
    count = int(match.group("count"))
    for i in range(1, count + 1):
        part = first.with_name(f"{match.group('base')}-{i:05d}-of-{count:05d}.gguf")
        try:
            total += gguf_reader.file_size(part)
        except OSError:
            pass
    return total
