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


# Tablica słownika (token_embd). llama.cpp (a więc i Ollama, LM Studio) trzyma ją w RAM,
# bo z wejścia czyta tylko jeden wiersz na token (pomiar autora: Gemma 3 4B –
# „CPU_Mapped model buffer size = 525 MiB”). Dwa przypadki:
#  * osobna warstwa wyjściowa (Llama 3.1, Qwen 3): tablica w ogóle nie trafia do karty;
#    jej rozmiar liczymy z dolnym szacunkiem bitów (≤ 4,5), żeby nie odjąć od karty za dużo,
#  * wspólne wejście i wyjście (Gemma, Llama 3.2): karta trzyma kopię jako wyjście,
#    a w RAM jest druga – tę liczymy z bitami Q6_K (tak llama.cpp kwantyzuje wyjście).
# Bez karty (sam procesor) kopii nie ma, więc nic nie zmieniamy.
EMBD_UNTIED_MAX_BITS = 4.5
EMBD_TIED_MIN_BITS = 6.56


def embd_bytes(model: ModelSpec, quants: QuantTable) -> tuple[int, int]:
    """(bajty tablicy słownika w RAM, dodatkowa kopia) – (0, 0), gdy nie znamy tablicy."""
    bits = quants.bits(model.quant)
    if not model.embd_params or model.tied_embeddings is None or bits is None:
        return 0, 0
    if model.tied_embeddings:
        size = int(model.embd_params * max(bits, EMBD_TIED_MIN_BITS) / 8)
        return size, size
    return int(model.embd_params * min(bits, EMBD_UNTIED_MAX_BITS) / 8), 0


def memory_need(model: ModelSpec, settings: RunSettings, quants: QuantTable,
                gpu: bool = True) -> MemoryNeed:
    """gpu=False: liczy dla samego procesora (bez podziału tablicy słownika)."""
    weights, weights_estimated = weights_bytes(model, quants)
    kv, kv_estimated, sliding = kv_bytes(model, settings)
    in_ram, copy = embd_bytes(model, quants) if gpu else (0, 0)
    decoder = model.audio.decoder_bytes if model.audio else 0
    return MemoryNeed(
        decoder=decoder,
        decoder_in_ram=not (gpu and settings.decoder_on_gpu),
        embd_in_ram=in_ram,
        embd_copy=copy,
        weights=weights,
        kv_cache=kv,
        overhead=settings.overhead,
        weights_estimated=weights_estimated,
        kv_estimated=kv_estimated,
        kv_sliding=sliding,
    )
