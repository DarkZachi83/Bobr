"""Pamięć podręczna modeli z Hugging Face na dysku.

Pobranie nagłówka GGUF trwa kilka sekund; architektura i lista plików repozytorium
zmieniają się rzadko. Drugie sprawdzenie tego samego modelu (inna kwantyzacja, inny
kontekst) bierze dane stąd. Wpis jest ważny przez MAX_AGE; uszkodzony = brak wpisu.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from aibobr.domain.enums import ModelSource
from aibobr.sources.base import ArchInfo, LoadedModel, Variant

log = logging.getLogger(__name__)
MAX_AGE = 7 * 24 * 3600
FORMAT = 7              # 7: warstwy MTP (mtp_layers) – starsze pobieramy od nowa


def model_to_dict(m: LoadedModel) -> dict:
    arch = asdict(m.arch)
    arch.pop("audio", None)                 # wynik rozpoznania liczymy od nowa po wczytaniu
    arch.pop("image", None)
    return {"name": m.name, "source": m.source.value, "arch": arch,
            "variants": [{"quant": v.quant, "size": v.size, "files": list(v.files)}
                         for v in m.variants],
            "origin": m.origin}


def _arch(d: dict) -> ArchInfo:
    d = {k: v for k, v in d.items() if k not in ("audio", "image")}
    if isinstance(d.get("image_parts"), list):                      # JSON: listy zamiast krotek
        d["image_parts"] = tuple(tuple(p) for p in d["image_parts"])
    return ArchInfo(**d)


def model_from_dict(d: dict) -> LoadedModel:
    return LoadedModel(
        name=d["name"], source=ModelSource(d["source"]), arch=_arch(d["arch"]),
        variants=tuple(Variant(v["quant"], v["size"], tuple(v["files"])) for v in d["variants"]),
        origin=d.get("origin"),
    )


class ModelCache:
    def __init__(self, directory: Path, max_age: float = MAX_AGE, clock=time.time):
        self.dir = Path(directory)
        self.max_age = max_age
        self.clock = clock

    def _path(self, key: str) -> Path:
        return self.dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")

    def get(self, key: str) -> Optional[LoadedModel]:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            if data.get("format") != FORMAT or data.get("key") != key \
                    or self.clock() - data["saved"] > self.max_age:
                return None
            return model_from_dict(data["model"])
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.info("cache entry for %s ignored: %s", key, e)
            return None

    def put(self, key: str, model: LoadedModel) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(json.dumps({
                "format": FORMAT, "key": key, "saved": self.clock(),
                "model": model_to_dict(model)}, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.info("cannot write cache: %s", e)
