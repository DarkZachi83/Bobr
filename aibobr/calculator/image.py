"""Pamięć modelu obrazów (etap 10, SPEC §5a): szczyt największego etapu, nie suma."""
from __future__ import annotations

from dataclasses import replace

from aibobr.domain.models import ImageNeed, ModelSpec, RunSettings
from aibobr.domain.units import gib

# Dekodowanie kafelkami (tiled VAE): VAE liczy obraz po kawałkach, więc pamięć nie rośnie
# z rozdzielczością. ComfyUI i Forge przechodzą na nie same, gdy zwykłe dekodowanie się nie
# mieści. Przyjmujemy pamięć jak dla ok. 1 megapiksela.
TILE_MPX = 1.0


def resolution(spec: ModelSpec, settings: RunSettings) -> tuple[int, int]:
    """Bok × bok: z ustawień (tryb Zaawansowany) albo natywny dla rodziny."""
    native = spec.image.native if spec.image else 1024
    if not settings.image_width:
        return native, native
    return settings.image_width, settings.image_height or settings.image_width


def image_need(spec: ModelSpec, settings: RunSettings) -> ImageNeed:
    info = spec.image
    w, h = resolution(spec, settings)
    batch = max(1, settings.image_batch)
    mpx = w * h / 1e6 * batch
    # GGUF i warianty z HF: części głównej nie ma w opisie – to cały plik wariantu.
    main = info.part("main") or (spec.weights_bytes or 0)
    return ImageNeed(
        main=main, encoders=info.part("encoders"), vae=info.part("vae"),
        main_compute=gib(info.main_gb_per_mpx * mpx), vae_compute=gib(info.vae_gb_per_mpx * mpx),
        overhead=gib(info.overhead_gb), width=w, height=h, batch=batch,
        estimated=not info.measured or bool(info.assumed))


def tiled(spec: ModelSpec, need: ImageNeed) -> ImageNeed:
    per_image = need.width * need.height / 1e6
    if per_image <= TILE_MPX:
        return need
    return replace(need, vae_compute=gib(spec.image.vae_gb_per_mpx * TILE_MPX), vae_tiled=True)
