"""Modele mowy i muzyki typu LLM (etap 9): rozpoznanie i dekoder dźwięku.

Kolejność sygnałów (decyzja autora, 28.10.2026):
  1. tokeny dźwięku na końcu słownika GGUF – działa dla pliku, linku HF i Ollamy,
  2. pipeline_tag z Hugging Face („text-to-speech”…),
  3. nazwa – tylko gdy dwa pierwsze nic nie dały, z komunikatem „Rozpoznano z nazwy pliku”.
Rodziny (wzory, dekoder, tokeny na sekundę) są w data/audio_models.json – nową dopisuje
się bez zmiany kodu.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Optional

from aibobr.domain.models import AudioInfo
from aibobr.domain.units import gib
from aibobr.sources.base import LoadedModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Family:
    id: str
    name: str
    kind: str
    token_patterns: tuple[str, ...]
    name_patterns: tuple[re.Pattern, ...]
    decoder: Optional[str]
    decoder_bytes: int
    decoder_measured: bool
    tokens_per_second: Optional[float]

    def info(self, by: str) -> AudioInfo:
        return AudioInfo(kind=self.kind, by=by, family=self.id, name=self.name, decoder=self.decoder,
                         decoder_bytes=self.decoder_bytes, decoder_estimated=not self.decoder_measured,
                         tokens_per_second=self.tokens_per_second)


class AudioCatalog:
    def __init__(self, families=(), hints=(), tags=None):
        self.families: tuple[Family, ...] = tuple(families)
        self.hints: tuple[tuple[re.Pattern, str], ...] = tuple(hints)
        self.tags: dict[str, str] = dict(tags or {})

    @classmethod
    def load(cls, path: Path) -> "AudioCatalog":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("audio catalog %s ignored: %s", path, e)
            return cls()
        families = []
        for f in data.get("families") or []:
            try:
                families.append(Family(
                    id=f["id"], name=f["name"], kind=f["kind"],
                    token_patterns=tuple(f.get("token_patterns") or ()),
                    name_patterns=tuple(re.compile(p, re.I) for p in f.get("name_patterns") or ()),
                    decoder=f.get("decoder"),
                    decoder_bytes=gib(float(f["decoder_gb"])) if f.get("decoder_gb") else 0,
                    decoder_measured=bool(f.get("decoder_measured")),
                    tokens_per_second=f.get("tokens_per_second")))
            except (KeyError, TypeError, ValueError, re.error) as e:
                log.warning("audio family skipped: %s", e)
        hints = [(re.compile(h["pattern"], re.I), h["kind"]) for h in data.get("name_hints") or ()
                 if isinstance(h, dict) and "pattern" in h and "kind" in h]
        return cls(families, hints, data.get("tags"))

    # ------------------------------------------------------------------ rozpoznanie
    @staticmethod
    def _names(model: LoadedModel) -> str:
        # Nazwa modelu i pliku (albo repozytorium) – bez folderów nad nim, żeby folder
        # „tts” z innymi plikami nie robił z każdego modelu syntezatora mowy.
        parts = [model.name or ""]
        if model.origin:
            parts.append(PurePath(model.origin.rstrip("/")).name)
        return " ".join(parts)

    def recognize(self, model: LoadedModel) -> Optional[AudioInfo]:
        arch = model.arch
        if arch.token_pattern:
            for f in self.families:
                if arch.token_pattern in f.token_patterns:
                    return f.info("tokens")
        names = self._names(model)
        by_name = next((f for f in self.families if any(p.search(names) for p in f.name_patterns)), None)
        kind = self.tags.get(arch.task or "")
        if kind:
            if by_name:
                return by_name.info("tag")
            return AudioInfo(kind=kind, by="tag")
        if by_name:
            return by_name.info("name")
        for pattern, hint in self.hints:
            if pattern.search(names):
                return AudioInfo(kind=hint, by="name")
        return None

    def annotate(self, model: LoadedModel) -> LoadedModel:
        audio = self.recognize(model)
        return replace(model, arch=replace(model.arch, audio=audio)) if audio else model


def seconds(audio: Optional[AudioInfo], context_tokens: int) -> Optional[float]:
    """Ile sekund nagrania mieści kontekst (bez tekstu i próbki głosu)."""
    if not audio or not audio.tokens_per_second:
        return None
    return context_tokens / audio.tokens_per_second
