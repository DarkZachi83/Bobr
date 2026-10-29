"""Wynik szukania: znaleziony model i raport z przebiegu."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Optional


class Kind(str, Enum):
    TEXT = "text"          # model językowy – można sprawdzić
    IMAGE = "image"        # generowanie obrazów – ocena od etapu 10
    ADDON = "addon"        # dodatek: LoRA, VAE, projektor obrazu – sam nie działa
    OTHER = "other"        # duży plik wag, którego nie rozpoznaliśmy


@dataclass(frozen=True)
class FoundModel:
    name: str
    kind: Kind
    fmt: str                         # "GGUF", "safetensors", "ckpt", "transformers", "diffusers"
    size: int                        # bajty (wszystkie części / pliki razem)
    path: Path                       # plik albo folder modelu
    app: str                         # skąd: "Ollama", "LM Studio", "Hugging Face", … albo "folder"
    quant: Optional[str] = None
    note: Optional[str] = None       # klucz i18n z dopiskiem, np. "scan.note.lora"
    target: Optional[str] = None     # tekst do pola „Wskaż model”; None = nie do sprawdzenia

    @property
    def place(self) -> str:
        """Gdzie leży – dla Ollamy folder modeli i nazwa zamiast pliku sha256-…"""
        if self.target and self.target.startswith("ollama:"):
            return f"{self.path.parent.parent}  ({self.target})"
        return str(self.path)

    @property
    def checkable(self) -> bool:
        return self.target is not None


@dataclass
class ScanReport:
    models: list[FoundModel] = field(default_factory=list)
    denied: list[Path] = field(default_factory=list)     # foldery bez prawa odczytu
    searched: list[Path] = field(default_factory=list)   # przeszukane miejsca
    files_seen: int = 0
    cancelled: bool = False

    @property
    def total_size(self) -> int:
        return sum(m.size for m in self.models)

    def with_copies(self) -> list[FoundModel]:
        """Lista z oznaczeniem kopii: ten sam model (nazwa i rozmiar) w innym miejscu
        zajmuje dysk drugi raz – warto wiedzieć, co można usunąć."""
        seen, result = set(), []
        for m in self.models:
            key = (m.name, m.quant, m.size)
            if key in seen and m.note is None:
                m = replace(m, note="scan.note.copy")
            seen.add(key)
            result.append(m)
        return result

    def merge(self, other: "ScanReport") -> None:
        seen = {(m.path, m.name) for m in self.models}
        self.models += [m for m in other.models if (m.path, m.name) not in seen]
        self.denied += [d for d in other.denied if d not in self.denied]
        self.searched += other.searched
        self.files_seen += other.files_seen
        self.cancelled = self.cancelled or other.cancelled
