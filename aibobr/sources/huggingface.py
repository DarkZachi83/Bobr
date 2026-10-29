"""Model z Hugging Face (SPEC §4).

Obsługujemy dwa rodzaje repozytoriów:
  * GGUF (np. bartowski/…-GGUF): prawdziwe rozmiary plików z listy, architektura z nagłówka
    jednego pliku czytanego przez HTTP Range – pobieramy kilka MB, nie gigabajty.
  * zwykłe (transformers): architektura z config.json, liczba parametrów z rozmiaru plików
    safetensors; rozmiary kwantyzacji GGUF szacowane z tabeli (oznaczone jako szacunek).

Endpointy: api/models/<repo>/tree/<rev>?recursive=true (lista plików z rozmiarami)
oraz <repo>/resolve/<rev>/<plik> (zawartość pliku).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Optional
from urllib.parse import quote

from aibobr.domain.enums import ModelSource
from aibobr.sources.base import (
    OBSOLETE_QUANT_RE, SPLIT_RE, ArchInfo, count_swa_layers, LoadedModel, SourceError, Variant, quant_from_filename,
)
from aibobr.sources import gguf_reader
from aibobr.sources.gguf_reader import arch_from_headers, display_name, read_header
from aibobr.sources.http import HttpClient

HF = "https://huggingface.co"
_NOT_MODELS = {"datasets", "spaces", "docs", "models", "collections", "blog", "papers", "settings"}
_ID = r"[A-Za-z0-9][\w.-]*"
_QUANT_TAG = re.compile(r":([A-Za-z0-9_]+)$")

# Typy zadań HF spoza obecnego zakresu (pipeline_tag) – osobny komunikat zamiast „brak plików”.
IMAGE_TAGS = {"text-to-image", "image-to-image", "unconditional-image-generation",
              "text-to-video", "image-to-video", "image-text-to-image", "text-to-3d", "image-to-3d"}
AUDIO_TAGS = {"text-to-audio", "text-to-speech", "audio-to-audio", "text-to-music"}

# Warianty pokazywane dla repozytoriów bez plików GGUF (rozmiary z tabeli kwantyzacji).
ESTIMATED_QUANTS = ("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


@dataclass(frozen=True)
class HfRef:
    repo_id: str
    revision: str = "main"
    path: Optional[str] = None       # konkretny plik z linku …/blob/main/<plik>
    quant: Optional[str] = None      # znacznik jak w Ollama: hf.co/<repo>:Q4_K_M


def parse_reference(text: str) -> Optional[HfRef]:
    """Link albo identyfikator HF → HfRef; None, jeśli to nie wygląda na HF.

    Przyjmuje: https://huggingface.co/<o>/<r>[/tree|blob|resolve/<rev>/<plik>],
    hf.co/<o>/<r>[:QUANT] (forma z `ollama run`), oraz samo <o>/<r>."""
    s = text.strip().split("#")[0].split("?")[0].rstrip("/")
    s = re.sub(r"^https?://", "", s, flags=re.I)
    s = re.sub(r"^www\.", "", s, flags=re.I)
    quant = None
    m = _QUANT_TAG.search(s)
    if m:
        quant, s = m.group(1).upper(), s[: m.start()]
    host_m = re.match(r"^(?:huggingface\.co|hf\.co)/(.+)$", s, flags=re.I)
    if host_m:
        s = host_m.group(1)
    elif not re.fullmatch(f"{_ID}/{_ID}", s):
        return None
    parts = s.split("/")
    if len(parts) < 2 or parts[0].lower() in _NOT_MODELS:
        return None
    if not (re.fullmatch(_ID, parts[0]) and re.fullmatch(_ID, parts[1])):
        return None
    revision, path = "main", None
    if len(parts) >= 4 and parts[2] in ("tree", "blob", "resolve"):
        revision = parts[3]
        path = "/".join(parts[4:]) or None
    return HfRef(f"{parts[0]}/{parts[1]}", revision, path, quant)


@dataclass(frozen=True)
class _File:
    path: str
    size: Optional[int]


def _preferred(ref: HfRef, variants) -> Optional[str]:
    """Kwantyzacja wskazana w linku: znacznik :QUANT albo konkretny plik."""
    if ref.path:
        for v in variants:
            if ref.path in v.files:
                return v.quant
    return ref.quant


class HuggingFaceSource:
    def __init__(self, client: HttpClient, cache=None):
        self.client = client
        self.cache = cache          # ModelCache albo None

    # ------------------------------------------------------------------ adresy
    def _tree_url(self, ref: HfRef) -> str:
        return f"{HF}/api/models/{ref.repo_id}/tree/{quote(ref.revision, safe='')}?recursive=true"

    def _file_url(self, ref: HfRef, path: str) -> str:
        return f"{HF}/{ref.repo_id}/resolve/{quote(ref.revision, safe='')}/{quote(path, safe='/')}"

    # ------------------------------------------------------------------ wczytanie
    def load(self, ref: HfRef) -> LoadedModel:
        key = f"{ref.repo_id}@{ref.revision}"
        cached = self.cache.get(key) if self.cache else None
        if cached is None:
            cached = self._fetch(ref)
            if self.cache:
                self.cache.put(key, cached)
        return replace(cached, preferred_quant=_preferred(ref, cached.variants))

    def _fetch(self, ref: HfRef) -> LoadedModel:
        entries = self.client.get_json(self._tree_url(ref))
        if not isinstance(entries, list):
            raise SourceError("source.http_error", "unexpected tree response")
        files = [
            _File(e["path"], (e.get("lfs") or {}).get("size") or e.get("size"))
            for e in entries if isinstance(e, dict) and e.get("type") == "file" and "path" in e
        ]
        try:
            gguf = [f for f in files if f.path.lower().endswith(".gguf")
                    and "mmproj" not in PurePosixPath(f.path).name.lower()
                    and not OBSOLETE_QUANT_RE.search(PurePosixPath(f.path).name)]
            if gguf:
                return self._load_gguf_repo(ref, gguf)
            if any(f.path == "config.json" for f in files):
                return self._load_transformers_repo(ref, files)
            raise SourceError(self._out_of_scope(ref, files), ref.repo_id)
        except SourceError as e:
            # Lista plików jest dostępna, a plik nie: repozytorium wymaga akceptacji licencji.
            if e.code == "source.access_denied":
                raise SourceError("source.gated", ref.repo_id) from e
            raise
        raise SourceError("source.no_model_files", ref.repo_id)

    def _out_of_scope(self, ref: HfRef, files: list[_File]) -> str:
        """Brak modelu językowego: obrazy / dźwięk (etapy 9–11) mówimy wprost (SPEC §15)."""
        tag = ""
        try:
            info = self.client.get_json(f"{HF}/api/models/{ref.repo_id}")
            tag = str((info or {}).get("pipeline_tag") or "") if isinstance(info, dict) else ""
        except SourceError:
            pass                                       # bez typu też coś powiemy
        if tag in IMAGE_TAGS or (not tag and any(f.path == "model_index.json" for f in files)):
            return "source.image_model"
        if tag in AUDIO_TAGS:
            return "source.audio_model"
        return "source.no_model_files"

    # ------------------------------------------------------------------ repo GGUF
    def _load_gguf_repo(self, ref: HfRef, files: list[_File]) -> LoadedModel:
        groups: dict[tuple[str, str], list[_File]] = defaultdict(list)
        for f in files:
            p = PurePosixPath(f.path)
            m = SPLIT_RE.match(p.name)
            groups[(str(p.parent), m.group("base") if m else p.name[:-5])].append(f)

        variants, seen = [], set()
        for (_, base), parts in groups.items():
            parts.sort(key=lambda f: f.path)
            m = SPLIT_RE.match(PurePosixPath(parts[0].path).name)
            if m and len(parts) != int(m.group("count")):
                continue                                    # niekompletny plik dzielony
            quant = quant_from_filename(base) or base
            if quant in seen:                               # dwa pliki tej samej kwantyzacji
                quant = base
            seen.add(quant)
            sizes = [f.size for f in parts]
            variants.append(Variant(quant, None if None in sizes else sum(sizes),
                                    tuple(f.path for f in parts)))
        if not variants:
            raise SourceError("source.no_model_files", ref.repo_id)
        variants.sort(key=lambda v: v.size or 0, reverse=True)

        preferred = _preferred(ref, variants)
        model = LoadedModel(name="", source=ModelSource.HUGGINGFACE, arch=ArchInfo(),
                            variants=tuple(variants), origin=f"{HF}/{ref.repo_id}",
                            preferred_quant=preferred)

        # Architektura jest wspólna dla wszystkich kwantyzacji – czytamy nagłówek jednej.
        probe = model.default_variant()
        headers = [read_header(self.client.open_remote(self._file_url(ref, path)))
                   for path in probe.files]
        return LoadedModel(
            name=display_name(headers[0], PurePosixPath(probe.files[0]).name),
            source=model.source, arch=arch_from_headers(headers), variants=model.variants,
            origin=model.origin, preferred_quant=preferred,
        )

    # ------------------------------------------------------------------ repo transformers
    def _load_transformers_repo(self, ref: HfRef, files: list[_File]) -> LoadedModel:
        config = self.client.get_json(self._file_url(ref, "config.json"))
        arch = arch_from_config(config, files)
        return LoadedModel(
            name=ref.repo_id.split("/", 1)[1], source=ModelSource.HUGGINGFACE, arch=arch,
            variants=tuple(Variant(q, None) for q in ESTIMATED_QUANTS),
            origin=f"{HF}/{ref.repo_id}", preferred_quant=ref.quant,
        )


def _first(c: dict, *keys: str) -> Any:
    for k in keys:
        if isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool):
            return int(c[k])
    return None


def arch_from_config(config: Any, files: list[_File] = ()) -> ArchInfo:
    """config.json (transformers) → ArchInfo. Modele multimodalne trzymają część
    tekstową w "text_config"; jej wartości mają pierwszeństwo."""
    if not isinstance(config, dict):
        raise SourceError("source.bad_file", "config.json is not an object")
    c = dict(config)
    if isinstance(config.get("text_config"), dict):
        c.update(config["text_config"])

    layers = _first(c, "num_hidden_layers", "n_layer", "num_layers")
    heads = _first(c, "num_attention_heads", "n_head")
    kv_heads = _first(c, "num_key_value_heads", "num_kv_heads") or heads
    hidden = _first(c, "hidden_size", "n_embd", "d_model")
    head_dim = _first(c, "head_dim") or (hidden // heads if hidden and heads else None)
    kv_lora = _first(c, "kv_lora_rank")
    if kv_lora:                                   # MLA, jak w gguf_reader
        kv_heads, head_dim = 1, math.ceil((kv_lora + (_first(c, "qk_rope_head_dim") or 0)) / 2)
    experts = _first(c, "num_local_experts", "num_experts", "n_routed_experts")
    used = _first(c, "num_experts_per_tok", "num_experts_per_token", "moe_topk")

    # Parametry z rozmiaru wag; modele już skwantyzowane (AWQ, GPTQ…) pomijamy – nie da się.
    total = None
    dtype = str(c.get("torch_dtype") or c.get("dtype") or "bfloat16")
    weight_files = [f for f in files if f.path.endswith(".safetensors")] or \
                   [f for f in files if f.path.endswith(".bin") and "pytorch_model" in f.path]
    if weight_files and "quantization_config" not in config and dtype in _DTYPE_BYTES:
        if all(f.size for f in weight_files):
            total = sum(f.size for f in weight_files) // _DTYPE_BYTES[dtype]

    active = None
    is_moe = bool(experts and experts > 1 and used)
    if is_moe and total and layers and hidden:
        inter = _first(c, "moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
        if inter:
            expert_params = layers * experts * 3 * hidden * inter
            if expert_params < total:
                active = total - expert_params * (1 - used / experts)
                vocab = _first(c, "vocab_size")
                if vocab and c.get("tie_word_embeddings") is False:   # jak w gguf_reader
                    active -= vocab * hidden
                active = int(active)

    # Tablica słownika. Bez pola tie_word_embeddings przyjmujemy wspólne wejście i wyjście
    # (domyślna wartość transformers) – wtedy nic nie odejmujemy od karty, czyli ostrożnie.
    vocab = _first(c, "vocab_size")
    embd = vocab * hidden if vocab and hidden else None
    tied = (c.get("tie_word_embeddings") is not False) if embd else None

    # Okno przesuwne tylko z jednoznacznych pól: layer_types (np. gpt-oss, Gemma 3)
    # albo sliding_window_pattern. Samo "sliding_window" bywa wyłączone (Qwen:
    # use_sliding_window = false) – wtedy liczymy pełny kontekst.
    window, swa_layers = _first(c, "sliding_window"), None
    if window and layers and c.get("use_sliding_window") is not False:
        types = c.get("layer_types")
        if isinstance(types, list) and types:
            swa_layers = count_swa_layers(layers, [t == "sliding_attention" for t in types])
        elif isinstance(c.get("sliding_window_pattern"), int):
            swa_layers = count_swa_layers(layers, c["sliding_window_pattern"])
    if not swa_layers:
        window = None

    return ArchInfo(
        sliding_window=window, swa_layers=swa_layers,
        params_total=total, n_layers=layers, n_kv_heads=kv_heads, head_dim=head_dim,
        max_context=_first(c, "max_position_embeddings", "max_sequence_length", "n_positions"),
        n_experts=experts if is_moe else None, n_experts_used=used if is_moe else None,
        params_active=active, architecture=(config.get("architectures") or [None])[0],
        embd_params=embd, tied_embeddings=tied,
    )


# --------------------------------------------------------------------------- folder lokalny

def load_transformers_folder(path) -> LoadedModel:
    """Folder transformers na dysku (config.json + wagi), np. z pamięci podręcznej HF.

    Liczymy jak repozytorium HF bez plików GGUF: architektura z config.json, warianty
    kwantyzacji szacowane z tabeli. Nic nie pobieramy z internetu."""
    folder = Path(path)
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SourceError("source.is_directory", str(folder)) from None
    except (OSError, ValueError) as e:
        raise SourceError("source.bad_file", str(e)) from e
    files = [_File(p.name, gguf_reader.file_size(p)) for p in sorted(folder.iterdir())
             if p.is_file() and (p.suffix == ".safetensors"
                                 or (p.name.startswith("pytorch_model") and p.suffix == ".bin"))]
    # Pamięć podręczna HF: …/models--org--nazwa/snapshots/<wersja>/ → „org/nazwa”
    name = folder.name
    if folder.parent.name == "snapshots" and folder.parent.parent.name.startswith("models--"):
        name = folder.parent.parent.name[len("models--"):].replace("--", "/")
    return LoadedModel(name=name, source=ModelSource.GGUF_FILE, arch=arch_from_config(config, files),
                       variants=tuple(Variant(q, None) for q in ESTIMATED_QUANTS),
                       origin=str(folder))
