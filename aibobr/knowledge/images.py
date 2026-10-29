"""Modele obrazów (etap 10, SPEC §5a): rodzina, rozdzielczość natywna, brakujące części.

Źródło daje surowe dane (rozmiary części, znaczniki). Tu dopasowujemy rodzinę z
data/image_models.json i uzupełniamy części, których nie ma w pliku (FLUX: enkodery
T5-XXL + CLIP-L i VAE leżą osobno) – te są oznaczone jako szacunek.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from aibobr.domain.models import ImageInfo
from aibobr.domain.units import gib
from aibobr.sources.base import LoadedModel

log = logging.getLogger(__name__)
SD_PIPELINE = "StableDiffusionPipeline"
SD2_ENCODER_MIN = gib(0.45)              # OpenCLIP-H (SD 2.x) ok. 0,66 GB, CLIP-L (SD 1.5) ok. 0,23 GB


@dataclass(frozen=True)
class ImageFamily:
    id: str
    name: str
    keys_all: tuple[str, ...]
    keys_any: tuple[str, ...]
    diffusers: tuple[str, ...]
    gguf: tuple[str, ...]
    native: int
    parts: dict
    main_gb_per_mpx: float
    vae_gb_per_mpx: float
    overhead_gb: float
    measured: bool

    def matches_keys(self, found: set[str]) -> bool:
        if self.keys_all and not set(self.keys_all) <= found:
            return False
        if self.keys_any and not set(self.keys_any) & found:
            return False
        return bool(self.keys_all or self.keys_any)


class ImageCatalog:
    def __init__(self, families=(), fallback: Optional[dict] = None):
        self.families: tuple[ImageFamily, ...] = tuple(families)
        self.fallback = dict(fallback or {"native": 1024, "main_gb_per_mpx": 1.5,
                                          "vae_gb_per_mpx": 3.0, "overhead_gb": 0.6})

    @classmethod
    def load(cls, path: Path) -> "ImageCatalog":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("image catalog %s ignored: %s", path, e)
            return cls()
        families = []
        for f in data.get("families") or []:
            try:
                families.append(ImageFamily(
                    id=f["id"], name=f["name"], keys_all=tuple(f.get("keys_all") or ()),
                    keys_any=tuple(f.get("keys_any") or ()), diffusers=tuple(f.get("diffusers") or ()),
                    gguf=tuple(f.get("gguf") or ()), native=int(f["native"]),
                    parts={k: gib(float(v)) for k, v in (f.get("parts_gb") or {}).items()},
                    main_gb_per_mpx=float(f["main_gb_per_mpx"]), vae_gb_per_mpx=float(f["vae_gb_per_mpx"]),
                    overhead_gb=float(f["overhead_gb"]), measured=bool(f.get("measured"))))
            except (KeyError, TypeError, ValueError) as e:
                log.warning("image family skipped: %s", e)
        return cls(families, data.get("fallback"))

    # ------------------------------------------------------------------ rozpoznanie
    def family(self, hint: str, parts: dict) -> tuple[Optional[ImageFamily], str]:
        kind, _, value = (hint or "").partition(":")
        if kind == "keys":
            found = set(filter(None, value.split(",")))
            return next((f for f in self.families if f.matches_keys(found)), None), "keys"
        if kind == "index":
            if value == SD_PIPELINE:                     # SD 1.5 i 2.x mają tę samą klasę
                want = "sd2" if parts.get("encoders", 0) >= SD2_ENCODER_MIN else "sd15"
                return next((f for f in self.families if f.id == want), None), "index"
            return next((f for f in self.families if value in f.diffusers), None), "index"
        if kind == "gguf":
            return next((f for f in self.families if value in f.gguf), None), "gguf"
        return None, kind or "?"

    def recognize(self, model: LoadedModel) -> Optional[ImageInfo]:
        arch = model.arch
        if not arch.image_hint:
            return None
        parts = dict(arch.image_parts or ())
        fam, by = self.family(arch.image_hint, parts)
        assumed = []
        for role in ("encoders", "vae"):              # w pliku nie ma – typowe z tabeli rodziny
            if not parts.get(role) and fam and fam.parts.get(role):
                parts[role] = fam.parts[role]
                assumed.append(role)
        fb = self.fallback
        return ImageInfo(
            family=fam.id if fam else None, name=fam.name if fam else None, by=by,
            parts=tuple((r, parts[r]) for r in ("main", "encoders", "vae") if parts.get(r)),
            assumed=tuple(assumed),
            native=fam.native if fam else int(fb["native"]),
            main_gb_per_mpx=fam.main_gb_per_mpx if fam else float(fb["main_gb_per_mpx"]),
            vae_gb_per_mpx=fam.vae_gb_per_mpx if fam else float(fb["vae_gb_per_mpx"]),
            overhead_gb=fam.overhead_gb if fam else float(fb["overhead_gb"]),
            measured=bool(fam and fam.measured))

    def annotate(self, model: LoadedModel) -> LoadedModel:
        info = self.recognize(model)
        return replace(model, arch=replace(model.arch, image=info)) if info else model
