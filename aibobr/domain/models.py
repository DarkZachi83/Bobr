"""Modele danych: sprzęt, model językowy, ustawienia uruchomienia i wynik.

Wszystkie obiekty są niezmienne (frozen). Wariant „co by było, gdyby” (tryb
Zaawansowany, podpowiedzi) tworzymy przez dataclasses.replace(), więc obliczenia
pozostają czystymi funkcjami. Rozmiary pamięci zawsze w bajtach (zob. units.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from aibobr.domain.enums import FitLevel, GpuVendor, KvCacheType, ModelSource
from aibobr.domain.units import gib


# --------------------------------------------------------------------------- sprzęt

@dataclass(frozen=True)
class GpuInfo:
    """Jedna karta graficzna.

    vram_total = None oznacza „nie udało się odczytać” – to co innego niż 0
    (SPEC §10: nie wolno po cichu przyjąć 0 GB)."""

    name: str
    vendor: GpuVendor
    vram_total: Optional[int]
    vram_used: Optional[int] = None          # tylko informacyjnie, nie wpływa na werdykt
    integrated: bool = False                 # grafika zintegrowana: pamięć = RAM, nie liczymy
    selected: bool = True                    # odznaczenie w trybie Zaawansowanym
    compute_capability: Optional[tuple[int, int]] = None  # tylko NVIDIA
    driver_version: Optional[str] = None     # tylko NVIDIA, np. "570.86.16"
    pci_bus: Optional[str] = None            # np. "0000:01:00.0" – do łączenia wyników


@dataclass(frozen=True)
class HardwareProfile:
    """Wynik skanu (albo ręcznie poprawione / fałszywe dane z --fake-hw)."""

    platform: str                            # "linux", "windows", "darwin", …
    cpu_name: Optional[str]
    ram_total: Optional[int]
    ram_used: Optional[int] = None           # tylko informacyjnie
    ram_description: Optional[str] = None    # np. "DDR4-3200", jeśli system to podaje
    gpus: tuple[GpuInfo, ...] = ()


# --------------------------------------------------------------------------- model

@dataclass(frozen=True)
class ImageInfo:
    """Model obrazów (etap 10, SPEC §5a): części i współczynniki pamięci obliczeń.

    parts: rola → bajty; role: "main" (UNet / DiT), "encoders", "vae". Części dobrane
    z tabeli rodziny (bo nie ma ich w pliku) są w `assumed`."""

    family: Optional[str]                    # "sdxl", "flux"…; None = nie rozpoznano
    name: Optional[str]                      # do wyświetlenia, np. "SDXL"
    by: str                                  # skąd: "keys", "index" (diffusers), "gguf"
    parts: tuple[tuple[str, int], ...]
    assumed: tuple[str, ...] = ()            # role wzięte z tabeli (szacunek)
    native: int = 1024                       # bok obrazu natywny dla rodziny (px)
    main_gb_per_mpx: float = 1.5             # obliczenia rysowania na megapiksel i obraz
    vae_gb_per_mpx: float = 3.0              # dekodowanie VAE na megapiksel i obraz
    overhead_gb: float = 0.6                 # PyTorch, cuDNN
    measured: bool = False                   # współczynniki z pomiaru (inaczej szacunek)

    def part(self, role: str) -> int:
        return sum(size for r, size in self.parts if r == role)


@dataclass(frozen=True)
class AudioInfo:
    """Model mowy albo muzyki typu LLM (etap 9): część językowa + dekoder dźwięku.

    Liczba tokenów na sekundę nagrania i pamięć dekodera pochodzą z data/audio_models.json."""

    kind: str                                # "speech" albo "music"
    by: str                                  # skąd wiemy: "tokens", "tag" (HF) albo "name"
    family: Optional[str] = None             # np. "orpheus"; None = model dźwięku, ale nieznany
    name: Optional[str] = None               # do wyświetlenia, np. "Orpheus"
    decoder: Optional[str] = None            # np. "SNAC 24 kHz"
    decoder_bytes: int = 0                   # 0 = nie znamy
    decoder_estimated: bool = True           # True do czasu pomiaru na prawdziwym sprzęcie
    tokens_per_second: Optional[float] = None


@dataclass(frozen=True)
class ModelSpec:
    """Konkretny model w konkretnej kwantyzacji (SPEC §4).

    Pola architektury (warstwy, głowice KV, wymiar głowicy) są opcjonalne: przy
    ręcznym wpisie na poziomie podstawowym ich nie ma i cache KV jest szacowany."""

    name: str
    source: ModelSource
    quant: str                               # np. "Q4_K_M", "F16"
    params_total: Optional[int] = None       # liczba parametrów (nie w miliardach)
    weights_bytes: Optional[int] = None      # rozmiar pliku GGUF, jeśli znany – ma pierwszeństwo
    n_layers: Optional[int] = None
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_context: Optional[int] = None
    n_experts: Optional[int] = None          # MoE: wszystkich ekspertów
    n_experts_used: Optional[int] = None     # MoE: aktywnych na token
    params_active: Optional[int] = None      # MoE: aktywnych parametrów na token
    sliding_window: Optional[int] = None     # okno przesuwne (tokeny), np. Gemma 3: 1024
    swa_layers: Optional[int] = None         # ile warstw używa okna (reszta: pełny kontekst)
    embd_params: Optional[int] = None        # tablica słownika (token_embd): słowa × wymiar
    tied_embeddings: Optional[bool] = None   # True = ta sama tablica jest też wyjściem modelu
    # Hybrydy (Qwen 3.5+, Qwen3-Next, Granite 4, Nemotron-H, Jamba, LFM2) i modele SSM
    # (Mamba, RWKV): tylko część warstw ma pamięć KV, reszta trzyma stały stan.
    kv_layers: Optional[int] = None          # warstwy z pamięcią KV; None = wszystkie
    recurrent_bytes: Optional[int] = None    # stan warstw liniowych (nie rośnie z kontekstem)
    audio: Optional[AudioInfo] = None        # model mowy / muzyki (etap 9)
    image: Optional[ImageInfo] = None        # model obrazów (etap 10)
    origin: Optional[str] = None             # link HF albo ścieżka pliku – do raportu

    @property
    def has_architecture(self) -> bool:
        return None not in (self.n_layers, self.n_kv_heads, self.head_dim)

    @property
    def is_moe(self) -> bool:
        return bool(self.n_experts and self.n_experts > 1)


# --------------------------------------------------------------------------- ustawienia

@dataclass(frozen=True)
class RunSettings:
    """Parametry uruchomienia i progi (SPEC §3, §5, §6). Wartości domyślne = tryb Automatyczny."""

    context_tokens: int = 8192
    kv_cache_type: KvCacheType = KvCacheType.F16
    vram_reserve: int = gib(1.0)             # na każdą kartę
    ram_reserve: int = gib(2.0)              # dla systemu; Windows: 4 GB (ustawia config)
    margin_ratio: float = 0.10               # próg ostrzeżenia o małym zapasie
    margin_min: int = gib(1.0)               # …ale nie mniej niż tyle
    overhead: int = gib(0.5)                 # bufor obliczeń + kontekst CUDA; kalibracja: SPEC §5
    decoder_on_gpu: bool = True              # modele mowy: dekoder dźwięku w karcie (domyślnie)
    image_width: Optional[int] = None        # modele obrazów: None = natywna rozdzielczość rodziny
    image_height: Optional[int] = None
    image_batch: int = 1                     # obrazów naraz


# --------------------------------------------------------------------------- wynik

@dataclass(frozen=True)
class MemoryNeed:
    """Ile pamięci potrzebuje model – trzy części paska modelu."""

    weights: int
    kv_cache: int
    overhead: int
    weights_estimated: bool                  # z bitów na wagę, nie z rozmiaru pliku
    kv_estimated: bool                       # bez danych architektury
    kv_sliding: bool = False                 # część warstw pamięta tylko okno przesuwne
    kv_layers: Optional[int] = None          # hybryda: tyle warstw pamięta cały kontekst…
    n_layers: Optional[int] = None           # …z tylu
    recurrent: int = 0                       # stan warstw liniowych (wliczony w kv_cache)
    # Tablica słownika (token_embd) – llama.cpp/Ollama trzyma ją w RAM, nie w karcie.
    # Gdy model ma wspólne wejście i wyjście, w karcie zostaje druga kopia (embd_copy).
    embd_in_ram: int = 0
    embd_copy: int = 0
    decoder: int = 0                         # dekoder dźwięku (modele mowy / muzyki)
    decoder_in_ram: bool = False             # dekoder uruchomiony na procesorze

    @property
    def total(self) -> int:
        """Cała pamięć modelu (karta + RAM)."""
        return self.weights + self.embd_copy + self.kv_cache + self.overhead + self.decoder

    @property
    def weights_total(self) -> int:
        """Wagi razem z kopią tablicy słownika – to pokazuje pasek i wyjaśnienie."""
        return self.weights + self.embd_copy

    @property
    def vram_part(self) -> int:
        """Ile z tego może trafić do karty – wszystko poza tablicą słownika w RAM."""
        return self.total - self.embd_in_ram - (self.decoder if self.decoder_in_ram else 0)

    @property
    def ram_fixed(self) -> int:
        """Co zawsze leży w RAM, niezależnie od miejsca w karcie."""
        return self.total - self.vram_part

    @property
    def is_estimate(self) -> bool:
        return self.weights_estimated or self.kv_estimated


@dataclass(frozen=True)
class Capacity:
    """Odcinki paska systemu: VRAM użytkowy (szybki) i RAM użytkowy (wolny)."""

    vram_usable: int
    ram_usable: int
    gpus_used: tuple[GpuInfo, ...]
    cpu_only: bool                           # brak osobnej karty branej pod uwagę
    missing: tuple[str, ...] = ()            # pola, których nie udało się odczytać

    @property
    def total(self) -> int:
        return self.vram_usable + self.ram_usable

    @property
    def complete(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class ImageNeed:
    """Modele obrazów: szczyty trzech etapów (SPEC §5a). Bajty."""

    main: int
    encoders: int
    vae: int
    main_compute: int
    vae_compute: int
    overhead: int
    width: int
    height: int
    batch: int
    estimated: bool                          # współczynniki albo części z tabeli
    encoders_on_cpu: bool = False            # enkoder nie mieści się w karcie – liczy procesor
    vae_tiled: bool = False                  # dekodowanie kafelkami (tiled VAE) – mniej pamięci

    @property
    def stages(self) -> dict[str, int]:
        return {"encode": self.encoders + self.overhead,
                "sample": self.main + self.main_compute + self.overhead,
                "decode": self.vae + self.vae_compute + self.overhead}

    @property
    def gpu_stages(self) -> dict[str, int]:
        s = self.stages
        if self.encoders_on_cpu:
            s.pop("encode")
        return s

    @property
    def peak_stage(self) -> str:
        s = self.gpu_stages
        return max(s, key=s.get)

    @property
    def peak(self) -> int:
        return max(self.gpu_stages.values())

    @property
    def weights(self) -> int:
        return self.main + self.encoders + self.vae

    @property
    def all_at_once(self) -> int:
        return self.weights + max(self.main_compute, self.vae_compute) + self.overhead

    # Zgodność z paskiem i doradcą modeli językowych (total = co musi się zmieścić łącznie)
    @property
    def total(self) -> int:
        return self.all_at_once

    @property
    def is_estimate(self) -> bool:
        return self.estimated

    @property
    def vram_part(self) -> int:
        return self.peak

    @property
    def ram_fixed(self) -> int:
        return 0


@dataclass(frozen=True)
class Verdict:
    """Werdykt dla jednej pary model + komputer."""

    level: FitLevel
    need: "MemoryNeed | ImageNeed"
    capacity: Capacity
    in_vram: int                             # ile bajtów modelu trafia do VRAM
    in_ram: int                              # ile do RAM
    shortfall: int                           # ile brakuje (tylko RED, inaczej 0)
    margin: Optional[int]                    # zapas do granicy strefy (None dla RED)
    low_margin: bool
    context_exceeds_model: bool              # kontekst większy niż model obsługuje
    warnings: tuple[str, ...] = field(default=())  # klucze i18n, np. "warn.nvidia_legacy"
    all_at_once: bool = False                # modele obrazów: wszystkie części naraz w karcie

    @property
    def spill(self) -> int:
        """Ile modelu wypchnął brak miejsca w karcie (bez tego, co i tak jest w RAM:
        tablicy słownika i dekodera uruchomionego na procesorze)."""
        return max(0, self.in_ram - self.need.ram_fixed)

    @property
    def vram_fraction(self) -> float:
        """Jaka część modelu jest w VRAM (0–1), nie licząc tablicy słownika, która i tak
        zostaje w RAM (nie spowalnia). Dla RED liczona względem całej potrzeby."""
        part = self.need.vram_part
        return self.in_vram / part if part else 0.0
