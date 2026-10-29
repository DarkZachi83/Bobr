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
    token_pattern: Optional[str] = None  # GGUF: wzór końca słownika – po nim poznajemy modele mowy

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

    @property
    def partial_size(self) -> int:
        """Przerwane pobrania – miejsce do odzyskania bez utraty żadnego modelu."""
        return sum(m.size for m in self.models if m.note == "scan.note.partial")

    def with_copies(self) -> list[FoundModel]:
        """Lista z oznaczeniem kopii: ten sam model (nazwa i rozmiar) w kilku miejscach
        zajmuje dysk kilka razy. Oznaczamy WSZYSTKIE wystąpienia – program nie wie,
        z której kopii korzysta użytkownik (np. Ollama jako usługa i z terminala)."""
        key = lambda m: (m.name, m.quant, m.size)        # noqa: E731
        counts: dict = {}
        for m in self.models:
            counts[key(m)] = counts.get(key(m), 0) + 1
        return [replace(m, note="scan.note.copy") if counts[key(m)] > 1 and m.note is None else m
                for m in self.models]

    def merge(self, other: "ScanReport") -> None:
        seen = {(m.path, m.name) for m in self.models}
        self.models += [m for m in other.models if (m.path, m.name) not in seen]
        self.denied += [d for d in other.denied if d not in self.denied]
        self.searched += other.searched
        self.files_seen += other.files_seen
        self.cancelled = self.cancelled or other.cancelled
