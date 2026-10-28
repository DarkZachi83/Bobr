"""Ile pamięci potrzebuje model: wagi + cache KV + narzut (SPEC §5)."""
from __future__ import annotations

from aibobr.calculator.quants import QuantTable
from aibobr.domain.models import MemoryNeed, ModelSpec, RunSettings

# Ostrożny szacunek cache KV, gdy nie znamy architektury (wpis ręczny, poziom podstawowy):
# bajty na token na miliard parametrów przy f16. Współczesne modele z GQA mają ok. 4–16 KiB
# (Llama 3.1 8B: 16 KiB, Qwen 2.5 32B: 8 KiB, Llama 3.3 70B: 4,6 KiB), więc 32 KiB celowo
# zawyża wynik. Starsze modele bez GQA (np. Llama 2 7B: ~73 KiB) mogą go przekroczyć –
# dlatego wynik jest oznaczony jako szacunek. Do kalibracji na katalogu (SPEC §16).
KV_ESTIMATE_PER_BILLION_F16 = 32 * 1024


class MissingModelData(ValueError):
    """Brakuje danych, bez których nie da się nic policzyć (np. ani pliku, ani parametrów)."""


def weights_bytes(model: ModelSpec, quants: QuantTable) -> tuple[int, bool]:
    """Rozmiar wag i informacja, czy to szacunek.

    Rozmiar pliku GGUF ma pierwszeństwo – to dokładnie to, co runtime załaduje."""
    if model.weights_bytes is not None:
        return model.weights_bytes, False
    if model.params_total is None:
        raise MissingModelData("params_total")
    bits = quants.bits(model.quant)
    if bits is None:
        raise MissingModelData(f"quant:{model.quant}")
    return int(model.params_total * bits / 8), True


# llama.cpp trzyma dla warstw z oknem przesuwnym min(kontekst, okno + n_ubatch) komórek,
# zaokrąglone w górę do 256 (llama-kv-cache-iswa.cpp; n_ubatch domyślnie 512).
SWA_UBATCH = 512


def swa_cells(window: int, context: int) -> int:
    return min(context, -(-(window + SWA_UBATCH) // 256) * 256)


def kv_bytes(model: ModelSpec, settings: RunSettings) -> tuple[int, bool, bool]:
    """Cache KV dla całego kontekstu: (bajty, czy szacunek, czy z oknem przesuwnym)."""
    element = settings.kv_cache_type.bytes_per_element
    ctx = settings.context_tokens
    if model.has_architecture:
        # K i V (stąd 2) × głowice KV × wymiar głowicy × bajty na element – na warstwę i token
        per_layer_token = 2 * model.n_kv_heads * model.head_dim * element
        swa = min(model.swa_layers or 0, model.n_layers) if model.sliding_window else 0
        cells = swa_cells(model.sliding_window, ctx) if swa else ctx
        total = per_layer_token * ((model.n_layers - swa) * ctx + swa * cells)
        return int(total), False, bool(swa) and cells < ctx
    if model.params_total is None:
        raise MissingModelData("params_total")
    per_token_f16 = KV_ESTIMATE_PER_BILLION_F16 * model.params_total / 1e9
    return int(per_token_f16 * element / 2.0 * ctx), True, False


def memory_need(model: ModelSpec, settings: RunSettings, quants: QuantTable) -> MemoryNeed:
    weights, weights_estimated = weights_bytes(model, quants)
    kv, kv_estimated, sliding = kv_bytes(model, settings)
    return MemoryNeed(
        weights=weights,
        kv_cache=kv,
        overhead=settings.overhead,
        weights_estimated=weights_estimated,
        kv_estimated=kv_estimated,
        kv_sliding=sliding,
    )
