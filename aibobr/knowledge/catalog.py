"""Katalog modeli do podpowiedzi „inny model się zmieści” (SPEC §9).

* data/models/catalog.json – dołączony do programu (budowany przez tools/build_catalog.py),
* kopia pobrana przyciskiem „Sprawdź aktualizacje” – w katalogu użytkownika.
Używana jest nowsza z nich (pole "updated"). Analizowany model nigdy nie pochodzi
z katalogu – katalog służy tylko podpowiedziom.

Format (wersja 1):
    {"format": 1, "updated": "2026-10-27", "source": "…",
     "models": [{"name": "…", "repo": "bartowski/…-GGUF",
                 "arch": {pola ArchInfo}, "variants": {"Q4_K_M": 4920000000, …}}]}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from aibobr.domain.enums import ModelSource
from aibobr.domain.models import ModelSpec
from aibobr.sources.base import ArchInfo, LoadedModel, SourceError, Variant

log = logging.getLogger(__name__)
FORMAT = 1
CATALOG_URL = "https://raw.githubusercontent.com/DarkZachi83/Bobr/main/data/models/catalog.json"
# Podpowiadany wariant: najpopularniejszy kompromis jakości i rozmiaru, potem bliskie.
PREFERRED_QUANTS = ("Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_0", "Q5_K_M")
_ARCH_FIELDS = {f.name for f in fields(ArchInfo)}


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class Catalog:
    updated: str
    models: tuple[LoadedModel, ...]
    path: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.models)

    def suggestion_specs(self) -> list[ModelSpec]:
        """Po jednym wariancie każdego modelu (Q4_K_M albo najbliższy) – dla doradcy."""
        specs = []
        for m in self.models:
            quants = {v.quant for v in m.variants}
            quant = next((q for q in PREFERRED_QUANTS if q in quants), None)
            if quant:
                specs.append(m.spec(quant))
        return specs


def parse(data: object, path: Optional[Path] = None) -> Catalog:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise CatalogError("unknown catalog format")
    models = []
    for i, entry in enumerate(data.get("models") or []):
        try:
            arch = ArchInfo(**{k: v for k, v in (entry.get("arch") or {}).items() if k in _ARCH_FIELDS})
            variants = tuple(Variant(q, int(size)) for q, size in entry["variants"].items()
                             if isinstance(size, (int, float)) and size > 0)
            if not variants or not entry.get("name"):
                raise ValueError("no variants or name")
            models.append(LoadedModel(
                name=str(entry["name"]), source=ModelSource.HUGGINGFACE, arch=arch,
                variants=variants, origin=f"https://huggingface.co/{entry['repo']}"
                if entry.get("repo") else None))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            log.warning("catalog entry %d skipped: %s", i, e)
    return Catalog(updated=str(data.get("updated", "")), models=tuple(models), path=path)


def load_file(path: Path) -> Optional[Catalog]:
    try:
        return parse(json.loads(Path(path).read_text(encoding="utf-8")), Path(path))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        log.warning("catalog %s ignored: %s", path, e)
        return None


def load(bundled: Path, downloaded: Optional[Path]) -> Catalog:
    """Nowszy z dwóch katalogów; pusty, gdy żadnego nie ma."""
    candidates = [c for c in (load_file(bundled), load_file(downloaded) if downloaded else None) if c]
    if not candidates:
        return Catalog(updated="", models=())
    return max(candidates, key=lambda c: c.updated)


def update(client, target: Path, current: Catalog, url: str = CATALOG_URL) -> tuple[Catalog, bool]:
    """Pobiera katalog z repozytorium projektu. (katalog, czy_nowszy).

    Zapisuje tylko poprawny i nowszy plik – uszkodzona odpowiedź nie zepsuje działającego."""
    data = client.get_json(url)
    try:
        fresh = parse(data)
    except CatalogError as e:
        raise SourceError("catalog.bad", str(e)) from e
    if not fresh.models or fresh.updated <= current.updated:
        return current, False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return Catalog(fresh.updated, fresh.models, target), True
