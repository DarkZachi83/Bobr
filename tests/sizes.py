"""Udawane rozmiary plików modeli w testach (bez wielogigabajtowych plików na dysku)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

from aibobr.sources import gguf_reader

_real = gguf_reader.file_size


def fake_sizes(sizes: dict) -> mock._patch:
    """Łatka: plik o nazwie z `sizes` ma podany rozmiar, reszta – prawdziwy.

    Użycie: self.addCleanup(fake_sizes({"model.gguf": 4_920_000_000}).stop) po .start()."""
    table = {Path(k).name: v for k, v in sizes.items()}
    return mock.patch.object(gguf_reader, "file_size",
                             lambda path: table.get(Path(path).name) or _real(Path(path)))


def write_model(path: Path, header: bytes, size: int) -> Path:
    """Nagłówek GGUF na dysku + udawany rozmiar całego pliku."""
    path.write_bytes(header)
    patch = fake_sizes({path.name: size})
    patch.start()
    _active.append(patch)
    return path


_active: list = []


def stop_all() -> None:
    while _active:
        _active.pop().stop()
