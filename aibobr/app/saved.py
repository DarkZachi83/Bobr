"""Zapisane modele do porównania (etap 11a, SPEC §5b).

Zapisujemy MODEL Z USTAWIENIAMI (skąd go wczytać, kwantyzacja, kontekst albo rozdzielczość),
a nie tylko wynik – porównanie w zakładce Statystyka liczy się dla obecnego komputera.
Wynik z dnia zapisu zostaje dla historii. Do 10 modeli w każdej grupie (decyzja autora).
Plik: saved_models.json obok ustawień użytkownika; uszkodzony = pusta lista.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

KINDS = ("text", "image", "audio")
MAX_PER_KIND = 10
FORMAT = 1


class ListFull(Exception):
    """W tej grupie jest już MAX_PER_KIND modeli – trzeba któryś usunąć."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True)
class SavedModel:
    kind: str                         # "text" | "image" | "audio"
    name: str                         # nazwa modelu do wyświetlenia
    target: str                       # tekst do pola „Model”: link HF, ścieżka, ollama:NAZWA
    quant: str
    saved_at: str                     # RRRR-MM-DD GG:MM (czas lokalny)
    level: str                        # werdykt w dniu zapisu: green / yellow / red
    need_gib: float                   # ile model potrzebował (szczyt przy obrazach)
    context: Optional[int] = None     # modele językowe i mowy
    kv_cache: Optional[str] = None
    resolution: Optional[tuple[int, int]] = None   # modele obrazów: rozdzielczość wyniku
    native_resolution: bool = True                 # …natywna rodziny (nie wpisana ręcznie)
    batch: int = 1
    hardware: str = ""                # krótki opis sprzętu z dnia zapisu („RTX 5050 8 GB · RAM 16 GB”)

    @property
    def key(self) -> tuple:
        """Ten sam model z tymi samymi ustawieniami = jeden wpis (zapis ponownie go odświeża)."""
        return (self.kind, self.target, self.quant, self.context, self.kv_cache,
                None if self.native_resolution else self.resolution, self.batch)


class SavedStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.items: list[SavedModel] = self._load()

    # ------------------------------------------------------------------ plik
    def _load(self) -> list[SavedModel]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as e:
            log.warning("cannot read %s: %s", self.path, e)
            return []
        items = []
        for raw in data.get("models", []) if isinstance(data, dict) else []:
            try:
                if raw.get("resolution"):
                    raw["resolution"] = tuple(raw["resolution"])
                item = SavedModel(**raw)
            except (TypeError, AttributeError):
                continue                                  # wpis z innej wersji – pomijamy
            if item.kind in KINDS:
                items.append(item)
        return items

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"format": FORMAT, "models": [asdict(i) for i in self.items]},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)                            # cały plik albo nic

    # ------------------------------------------------------------------ operacje
    def of_kind(self, kind: str) -> list[SavedModel]:
        """Od najnowszego."""
        return sorted((i for i in self.items if i.kind == kind), key=lambda i: i.saved_at, reverse=True)

    def add(self, item: SavedModel) -> bool:
        """True = nowy wpis, False = odświeżony istniejący. ListFull, gdy grupa jest pełna."""
        for n, old in enumerate(self.items):
            if old.key == item.key:
                self.items[n] = item
                self._write()
                return False
        if len(self.of_kind(item.kind)) >= MAX_PER_KIND:
            raise ListFull(item.kind)
        self.items.append(item)
        self._write()
        return True

    def remove(self, item: SavedModel) -> None:
        self.items = [i for i in self.items if i.key != item.key]
        self._write()

    def contains(self, item: SavedModel) -> bool:
        return any(i.key == item.key for i in self.items)

