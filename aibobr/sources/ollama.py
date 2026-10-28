"""Model zainstalowany w Ollamie, odczytany przez jej lokalne API (bez szukania plików).

* GET  /api/tags              – lista modeli z rozmiarem pliku i kwantyzacją,
* POST /api/show (verbose)    – metadane GGUF (model_info) i lista tensorów,
* GET  /api/ps                – co jest załadowane: rozmiar łącznie i w VRAM (do kalibracji).
Metadane mają te same klucze co nagłówek GGUF, więc architekturę liczy ten sam kod
co dla pliku .gguf (gguf_reader.arch_from_headers).

Na razie używane przez tools/calibrate.py; pole „Wskaż model” w programie jeszcze go nie zna.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from aibobr.domain.enums import ModelSource
from aibobr.sources.base import LoadedModel, SourceError, Variant
from aibobr.sources.gguf_reader import FILE_TYPES, GgufHeader, TensorInfo, arch_from_headers

DEFAULT_HOST = "http://127.0.0.1:11434"


def _same(a: str, b: str) -> bool:
    """„llama3.1:8b” == „llama3.1:8b”, „qwen3” == „qwen3:latest”."""
    norm = lambda n: n if ":" in n else n + ":latest"   # noqa: E731
    return norm(a.strip()) == norm(b.strip())


class OllamaSource:
    def __init__(self, client, host: str = DEFAULT_HOST):
        self.client = client
        self.host = host.rstrip("/")

    def _get(self, path: str) -> Any:
        try:
            return self.client.get_json(self.host + path)
        except SourceError as e:
            if e.code == "source.network":
                raise SourceError("ollama.not_running", self.host) from e
            raise

    def _post(self, path: str, payload: dict, timeout: Optional[float] = None) -> Any:
        try:
            return self.client.post_json(self.host + path, payload, timeout)
        except SourceError as e:
            if e.code == "source.network":
                raise SourceError("ollama.not_running", self.host) from e
            raise

    # ------------------------------------------------------------------ odczyt
    def version(self) -> str:
        return str((self._get("/api/version") or {}).get("version", "?"))

    def installed(self) -> list[dict]:
        return list((self._get("/api/tags") or {}).get("models") or [])

    def running(self) -> list[dict]:
        return list((self._get("/api/ps") or {}).get("models") or [])

    def load(self, name: str) -> LoadedModel:
        entry = next((m for m in self.installed()
                      if _same(name, m.get("name", "")) or _same(name, m.get("model", ""))), None)
        if entry is None:
            raise SourceError("ollama.not_found", name)
        show = self._post("/api/show", {"model": entry["name"], "verbose": True}, timeout=60)
        info = show.get("model_info") or {}
        if not info.get("general.architecture"):
            raise SourceError("ollama.no_metadata", name)
        tensors = tuple(TensorInfo(t["name"], tuple(int(d) for d in t.get("shape") or ()))
                        for t in show.get("tensors") or () if "name" in t)
        arch = arch_from_headers([GgufHeader(3, info, tensors)])
        if not arch.params_total and isinstance(info.get("general.parameter_count"), int):
            arch = replace(arch, params_total=info["general.parameter_count"])
        details = show.get("details") or entry.get("details") or {}
        quant = str(details.get("quantization_level") or "").upper() \
            or FILE_TYPES.get(info.get("general.file_type"), "?")
        return LoadedModel(
            name=entry["name"], source=ModelSource.GGUF_FILE, arch=arch,
            variants=(Variant(quant, int(entry.get("size") or 0) or None),),
            origin=f"ollama:{entry['name']}", preferred_quant=quant,
        )

    # ------------------------------------------------------------------ ładowanie (kalibracja)
    def load_into_memory(self, name: str, num_ctx: int, keep_alive: str = "10m") -> None:
        """Pusty prompt = Ollama tylko ładuje model (dokumentacja /api/generate)."""
        self._post("/api/generate", {"model": name, "prompt": "", "keep_alive": keep_alive,
                                     "options": {"num_ctx": num_ctx}}, timeout=900)

    def unload(self, name: str) -> None:
        self._post("/api/generate", {"model": name, "prompt": "", "keep_alive": 0}, timeout=120)

    def unload_all(self) -> None:
        for m in self.running():
            self.unload(m.get("name") or m.get("model"))
