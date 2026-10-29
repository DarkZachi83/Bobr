"""Modele obrazów (etap 10, SPEC §5a): plik .safetensors, folder diffusers, GGUF.

Źródło zbiera tylko surowe dane: rozmiary części (main / encoders / vae) i znaczniki
rozpoznawcze (image_hint). Rodzinę, rozdzielczość i współczynniki dobiera
knowledge/images.py z data/image_models.json.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

from aibobr.domain.enums import ModelSource
from aibobr.sources import gguf_reader
from aibobr.sources.base import ArchInfo, LoadedModel, SourceError, Variant
from aibobr.sources.gguf_reader import LocalFile
from aibobr.sources.safetensors_reader import DTYPE_LABEL, SafetensorsHeader, read_header

# Prefiksy tensorów w plikach „wszystko w jednym” (format ComfyUI / A1111 / Forge).
ENCODER_PREFIXES = ("cond_stage_model.", "conditioner.embedders.", "text_encoders.", "text_encoder.")
VAE_PREFIXES = ("first_stage_model.", "vae.")
# Znaczniki rodzin – data/image_models.json odwołuje się do nich w keys_all / keys_any.
MARKERS = ("model.diffusion_model.", "conditioner.embedders.", "cond_stage_model.transformer.",
           "cond_stage_model.model.", "double_blocks.", "single_blocks.", "joint_blocks.")
LORA_HINTS = ("lora_", ".lora_", "lora.", "lora_unet", "lora_te")
TEXT_MODEL_HINTS = ("model.layers.", "model.embed_tokens", "lm_head.")
from aibobr.sources.gguf_reader import GGUF_IMAGE_ARCHS  # noqa: E402


def role(name: str) -> str:
    if name.startswith(ENCODER_PREFIXES):
        return "encoders"
    if name.startswith(VAE_PREFIXES):
        return "vae"
    return "main"


def parts_from_header(h: SafetensorsHeader) -> tuple[tuple[str, int], ...]:
    sizes: dict[str, int] = {}
    for t in h.tensors:
        sizes[role(t.name)] = sizes.get(role(t.name), 0) + t.nbytes
    return tuple((r, sizes[r]) for r in ("main", "encoders", "vae") if sizes.get(r))


def hint_from_header(h: SafetensorsHeader) -> str:
    names = [t.name for t in h.tensors]
    found = [m for m in MARKERS if any(n.startswith(m) for n in names)]
    return "keys:" + ",".join(found)


def check_whole_model(h: SafetensorsHeader, path: str) -> None:
    """Dodatki (LoRA, sam VAE) i pojedyncze pliki modeli językowych nie są modelem obrazów."""
    names = [t.name.lower() for t in h.tensors]
    if any(any(x in n for x in LORA_HINTS) for n in names):
        raise SourceError("source.image_addon", path)
    if names and all(n.startswith(("encoder.", "decoder.", "quant_conv", "post_quant_conv")) for n in names):
        raise SourceError("source.image_addon", path)
    if any(n.startswith(TEXT_MODEL_HINTS) for n in names):
        raise SourceError("source.text_safetensors", path)


def image_model(name: str, header: SafetensorsHeader, size: int, origin: str,
                source: ModelSource = ModelSource.GGUF_FILE, files=()) -> LoadedModel:
    quant = DTYPE_LABEL.get(header.main_dtype() or "", header.main_dtype() or "?")
    arch = ArchInfo(params_total=None, architecture="image",
                    image_parts=parts_from_header(header), image_hint=hint_from_header(header))
    return LoadedModel(name=name, source=source, arch=arch,
                       variants=(Variant(quant, size, tuple(files)),), origin=origin,
                       preferred_quant=quant)


# --------------------------------------------------------------------------- plik lokalny

def load_safetensors_file(path) -> LoadedModel:
    path = Path(path)
    if not path.is_file():
        raise SourceError("source.file_missing", str(path))
    with open(path, "rb") as f:
        header = read_header(LocalFile(f))
    check_whole_model(header, str(path))
    return image_model(path.stem, header, gguf_reader.file_size(path), str(path), files=(str(path),))


# --------------------------------------------------------------------------- diffusers

MAIN_DIRS = ("unet", "transformer", "prior", "decoder")
VAE_DIRS = ("vae", "vqgan", "movq")
WEIGHT_SUFFIXES = (".safetensors", ".bin")


def parts_from_files(index: dict, files: Iterable[tuple[str, Optional[int]]]) -> tuple[tuple[str, int], ...]:
    """Rozmiary części z listy plików folderu diffusers (ścieżka względna, rozmiar).

    W jednym podfolderze bywa kilka wariantów (model.safetensors i model.fp16.safetensors) –
    bierzemy fp16, bo tak ładują programy; inaczej wszystkie pliki wag."""
    by_dir: dict[str, list[tuple[str, int]]] = {}
    for rel, size in files:
        p = PurePosixPath(rel)
        if len(p.parts) != 2 or not p.name.endswith(WEIGHT_SUFFIXES) or not size:
            continue
        by_dir.setdefault(p.parts[0], []).append((p.name, size))
    sizes: dict[str, int] = {}
    for folder, items in by_dir.items():
        if folder not in index:
            continue
        fp16 = [s for n, s in items if ".fp16." in n]
        chosen = fp16 or [s for n, s in items if not any(f".{v}." in n for v in ("fp32", "bf16"))] \
            or [s for _, s in items]
        r = "main" if folder.startswith(MAIN_DIRS) else "vae" if folder in VAE_DIRS \
            else "encoders" if folder.startswith("text_encoder") else None
        if r:
            sizes[r] = sizes.get(r, 0) + sum(chosen)
    return tuple((r, sizes[r]) for r in ("main", "encoders", "vae") if sizes.get(r))


def diffusers_model(name: str, index: dict, files, origin: str, source: ModelSource) -> LoadedModel:
    parts = parts_from_files(index, files)
    if not parts:
        raise SourceError("source.no_model_files", origin)          # sam szkielet, bez wag
    arch = ArchInfo(architecture="image", image_parts=parts,
                    image_hint=f"index:{index.get('_class_name', '')}")
    size = sum(s for _, s in parts)
    return LoadedModel(name=name, source=source, arch=arch, variants=(Variant("FP16", size),),
                       origin=origin, preferred_quant="FP16")


def load_diffusers_folder(path) -> LoadedModel:
    folder = Path(path)
    try:
        index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SourceError("source.bad_file", str(e)) from e
    files = [(p.relative_to(folder).as_posix(), gguf_reader.file_size(p))
             for p in folder.glob("*/*") if p.is_file()]
    name = folder.name
    if folder.parent.name == "snapshots" and folder.parent.parent.name.startswith("models--"):
        name = folder.parent.parent.name[len("models--"):].replace("--", "/")
    return diffusers_model(name, index if isinstance(index, dict) else {}, files, str(folder),
                           ModelSource.GGUF_FILE)


# --------------------------------------------------------------------------- GGUF

def gguf_image_arch(arch: Optional[str]) -> bool:
    return arch in GGUF_IMAGE_ARCHS
