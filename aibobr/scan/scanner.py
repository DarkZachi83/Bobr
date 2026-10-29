"""Przeszukiwanie folderów: znane miejsca programów albo folder wskazany przez użytkownika.

Rozpoznajemy po drodze całe „magazyny” zamiast pojedynczych plików:
* folder Ollamy (manifests/ + blobs/) – nazwy z manifestów, nie skróty sha256,
* repozytorium w pamięci podręcznej Hugging Face (models--org--nazwa) – nazwa org/nazwa,
* folder transformers (config.json + wagi) i diffusers (model_index.json) – jedna pozycja.
Ten sam plik widziany dwa razy (dowiązania w pamięci HF) liczymy raz.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from aibobr.scan import ollama_store
from aibobr.scan.files import MIN_SIZE, classify_file, classify_gguf
from aibobr.scan.locations import known_locations
from aibobr.scan.model import FoundModel, Kind, ScanReport
from aibobr.sources import gguf_reader

# Foldery, do których nie zaglądamy przy przeszukiwaniu folderu wskazanego ręcznie.
_SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "proc", "sys", "dev", "run",
              "$recycle.bin", "system volume information", "windows", "program files",
              "program files (x86)", "programdata"}
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _size(path: Path) -> int:
    try:
        return gguf_reader.file_size(path)
    except OSError:
        return 0


def _is_weight_file(name: str) -> bool:
    n = name.lower()
    return n.endswith(".safetensors") or (n.startswith("pytorch_model") and n.endswith(".bin"))


class Scanner:
    def __init__(self, stop: Optional[threading.Event] = None,
                 progress: Optional[Callable[[int], None]] = None):
        self.stop = stop or threading.Event()
        self.progress = progress or (lambda n: None)
        self._seen: set = set()

    # ------------------------------------------------------------------ wejścia
    def scan_known(self, platform: str, env=None, home: Optional[Path] = None) -> ScanReport:
        report = ScanReport()
        for app, path in known_locations(platform, env, home):
            report.merge(self.scan_folder(path, app))
            if self.stop.is_set():
                break
        return report

    def scan_folder(self, root: Path, app: str = "folder") -> ScanReport:
        report = ScanReport(searched=[Path(root)])
        root = Path(root)
        if not root.is_dir():
            return report

        def denied(err: OSError) -> None:
            if isinstance(err, PermissionError) and err.filename:
                report.denied.append(Path(err.filename))

        for dirpath, dirnames, filenames in os.walk(root, onerror=denied):
            if self.stop.is_set():
                report.cancelled = True
                break
            d = Path(dirpath)
            report.files_seen += len(filenames)
            self.progress(report.files_seen)

            if ollama_store.is_store(d):
                report.models += self._ollama(d)
                dirnames[:] = []
                continue
            if d.name.startswith("models--") and (d / "snapshots").is_dir():
                report.models += self._hf_repo(d)
                dirnames[:] = []
                continue
            if "model_index.json" in filenames:
                found = self._diffusers(d, app)
                if found:                       # same pliki konfiguracji (Forge, SD.Next) pomijamy
                    report.models.append(found)
                dirnames[:] = []
                continue
            grouped = False
            if "config.json" in filenames and any(_is_weight_file(f) for f in filenames):
                found = self._transformers(d, filenames, app)
                if found:
                    report.models.append(found)
                    grouped = True

            dirnames[:] = sorted(n for n in dirnames
                                 if not n.startswith(".") and n.lower() not in _SKIP_DIRS)
            for name in sorted(filenames):
                if grouped and _is_weight_file(name):
                    continue
                found = self._file(d / name, app)
                if found:
                    report.models.append(found)
        return report

    # ------------------------------------------------------------------ pojedyncze rodzaje
    def _first_time(self, path: Path) -> bool:
        try:
            st = path.stat()
        except OSError:
            return False
        key = (st.st_dev, st.st_ino)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def _file(self, path: Path, app: str, name: Optional[str] = None) -> Optional[FoundModel]:
        size = _size(path)
        if size < MIN_SIZE or not self._first_time(path):
            return None
        found = classify_file(path, size, app)
        if found and name:
            found = FoundModel(name, found.kind, found.fmt, found.size, found.path, found.app,
                               found.quant, found.note, found.target)
        return found

    def _ollama(self, store: Path) -> list[FoundModel]:
        result = []
        for name, blob, size in ollama_store.entries(store):
            if not blob.is_file() or not self._first_time(blob):
                continue
            found = classify_gguf(blob, size or _size(blob), "Ollama", name=name)
            if found and found.kind == Kind.TEXT:
                found = FoundModel(found.name, found.kind, found.fmt, found.size, found.path,
                                   "Ollama", found.quant, found.note, f"ollama:{name}")
            if found:
                result.append(found)
        # Same bloby leżą w blobs/ – do nich nie schodzimy, żeby nie liczyć ich drugi raz.
        return result

    def _hf_repo(self, repo_dir: Path) -> list[FoundModel]:
        repo_id = repo_dir.name[len("models--"):].replace("--", "/")
        snapshots = sorted((p for p in (repo_dir / "snapshots").iterdir() if p.is_dir()),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshots:
            return []
        sub = Scanner(self.stop)
        sub._seen = self._seen
        report = sub.scan_folder(snapshots[0], "Hugging Face")
        result = []
        for m in report.models:
            name = repo_id if m.fmt in ("transformers", "diffusers") else f"{repo_id} · {m.path.name}"
            result.append(FoundModel(name, m.kind, m.fmt, m.size, m.path, "Hugging Face",
                                     m.quant, m.note, m.target))
        return result

    def _folder_size(self, d: Path) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(d):
            for f in filenames:
                p = Path(dirpath) / f
                if self._first_time(p):
                    total += _size(p)
        return total

    def _diffusers(self, d: Path, app: str) -> Optional[FoundModel]:
        """Folder diffusers. Forge i SD.Next trzymają też „szkielety” – same pliki JSON
        z konfiguracją, bez wag (backend/huggingface, configs). To nie są modele."""
        size = self._folder_size(d)
        if size < MIN_SIZE:
            return None
        return FoundModel(d.name, Kind.IMAGE, "diffusers", size, d, app)

    def _transformers(self, d: Path, filenames: list[str], app: str) -> Optional[FoundModel]:
        weights = [d / f for f in filenames if _is_weight_file(f)]
        size = sum(_size(p) for p in weights if self._first_time(p))
        if size < MIN_SIZE:
            return None
        try:
            config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        archs = config.get("architectures") if isinstance(config, dict) else None
        archs = [a for a in archs if isinstance(a, str)] if isinstance(archs, list) else []
        # Modele dźwięku w formacie transformers (Whisper, MusicGen…) – etap 9; na razie „inne”.
        audio = any(w in a.lower() for a in archs for w in ("whisper", "speech", "musicgen", "wav2vec", "audio"))
        language = not audio and any(
            a.endswith(("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")) for a in archs)
        kind = Kind.TEXT if language else Kind.OTHER
        return FoundModel(d.name, kind, "transformers", size, d, app,
                          target=str(d) if language else None)
