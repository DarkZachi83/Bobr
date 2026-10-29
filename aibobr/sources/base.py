"""Wspólne typy źródeł modelu."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from aibobr.domain.enums import ModelSource
from aibobr.domain.models import AudioInfo, ModelSpec


class SourceError(Exception):
    """Błąd źródła z kluczem i18n (np. "source.not_found") i szczegółem do logu.

    GUI pokazuje tekst spod klucza, a nie surowy wyjątek (zasada z NIUCHACZA)."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ArchInfo:
    """Architektura modelu – wspólna dla wszystkich kwantyzacji tego samego modelu."""

    params_total: Optional[int] = None
    n_layers: Optional[int] = None
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_context: Optional[int] = None
    n_experts: Optional[int] = None
    n_experts_used: Optional[int] = None
    params_active: Optional[int] = None
    architecture: Optional[str] = None       # np. "llama", "qwen3moe"
    sliding_window: Optional[int] = None
    swa_layers: Optional[int] = None
    embd_params: Optional[int] = None        # token_embd: słowa × wymiar
    tied_embeddings: Optional[bool] = None   # brak osobnej warstwy wyjściowej
    # Ślady modelu dźwięku (etap 9) – surowe dane ze źródła, interpretuje knowledge/audio.py:
    token_pattern: Optional[str] = None      # najczęstszy wzór końca słownika, np. "<custom_token_#>"
    task: Optional[str] = None               # pipeline_tag z Hugging Face, np. "text-to-speech"
    audio: Optional[AudioInfo] = None        # wynik rozpoznania (po wczytaniu modelu)

    def to_spec(self, *, name: str, source: ModelSource, quant: str,
                weights_bytes: Optional[int], origin: Optional[str]) -> ModelSpec:
        return ModelSpec(
            name=name, source=source, quant=quant, weights_bytes=weights_bytes,
            params_total=self.params_total, n_layers=self.n_layers,
            n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
            max_context=self.max_context, n_experts=self.n_experts,
            n_experts_used=self.n_experts_used, params_active=self.params_active,
            sliding_window=self.sliding_window, swa_layers=self.swa_layers,
            embd_params=self.embd_params, tied_embeddings=self.tied_embeddings,
            audio=self.audio, origin=origin,
        )


@dataclass(frozen=True)
class Variant:
    """Jedna kwantyzacja do wyboru. size=None → rozmiar zostanie oszacowany z tabeli."""

    quant: str
    size: Optional[int]
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedModel:
    """Model wczytany ze źródła: nazwa, architektura i warianty do wyboru."""

    name: str
    source: ModelSource
    arch: ArchInfo
    variants: tuple[Variant, ...]
    origin: Optional[str] = None
    preferred_quant: Optional[str] = None    # np. z linku "hf.co/…:Q4_K_M" albo z pliku

    def variant(self, quant: str) -> Variant:
        for v in self.variants:
            if v.quant.upper() == quant.upper():
                return v
        raise SourceError("source.unknown_quant", quant)

    def default_variant(self) -> Variant:
        """Wariant z linku/pliku, inaczej Q4_K_M (najpopularniejszy), inaczej pierwszy."""
        for wanted in (self.preferred_quant, "Q4_K_M"):
            if wanted:
                try:
                    return self.variant(wanted)
                except SourceError:
                    pass
        return self.variants[0]

    def spec(self, quant: Optional[str] = None) -> ModelSpec:
        v = self.variant(quant) if quant else self.default_variant()
        return self.arch.to_spec(name=self.name, source=self.source, quant=v.quant,
                                 weights_bytes=v.size, origin=self.origin)


def count_swa_layers(n_layers: int, pattern) -> int:
    """Ile warstw używa okna przesuwnego – reguła z llama.cpp (llama_hparams::set_swa_pattern).

    pattern: tablica (prawda = warstwa z oknem) albo liczba n: w każdej grupie n warstw
    pierwsze n-1 mają okno, ostatnia pełny kontekst (0 = wszystkie z oknem)."""
    if isinstance(pattern, (list, tuple)):
        return sum(1 for x in list(pattern)[:n_layers] if x)
    n = int(pattern)
    return sum(1 for il in range(n_layers) if n == 0 or il % n < n - 1)


# Nazwy kwantyzacji w nazwach plików GGUF, od najdłuższych (IQ4_XS przed Q4…).
_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    r"IQ[1-4]_(?:XXS|XS|NL|S|M)|Q[2-8]_K_(?:XL|S|M|L)|Q[2-8]_K|Q[2-8]_[01]|"
    r"MXFP4(?:_MOE)?|TQ[12]_0|BF16|F16|F32"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Przestarzałe formaty ARM usunięte z llama.cpp – obecne runtime'y ich nie wczytają,
# więc nie pokazujemy ich jako wariantów do wyboru.
OBSOLETE_QUANT_RE = re.compile(r"(?<![A-Za-z0-9])Q4_0_(?:4_4|4_8|8_8)(?![A-Za-z0-9])", re.I)
SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)


def quant_from_filename(name: str) -> Optional[str]:
    """'Llama-3.1-8B-Instruct-Q4_K_M.gguf' → 'Q4_K_M'. Bierze ostatnie trafienie."""
    found = _QUANT_RE.findall(name)
    return found[-1].upper() if found else None
