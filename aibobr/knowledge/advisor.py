"""Werdykt w jednym zdaniu i wyjaśnienie w trzech sekcjach (SPEC §8).

Nic tu nie jest gotowym zdaniem: każda linia to klucz i18n + pola z surowymi
liczbami (Size, Context…), formatowanymi dopiero przy wyświetleniu w wybranym
języku (render()). Dzięki temu GUI może przełączyć język bez przeliczania.

„Co możesz zrobić” nie zgaduje: przelicza werdykt dla innych kwantyzacji, krótszego
kontekstu, cache KV w q8_0 i modeli z katalogu i pokazuje tylko to, co zmienia wynik.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

from aibobr.calculator.memory import MissingModelData
from aibobr.calculator.quants import QuantTable
from aibobr.calculator.verdict import evaluate
from aibobr.domain.enums import FitLevel, KvCacheType
from aibobr.domain.models import HardwareProfile, ModelSpec, RunSettings, Verdict
from aibobr.i18n import fmt_context, fmt_gb, fmt_number, fmt_params, t
from aibobr.sources.base import LoadedModel

# Poniżej ok. 3,4 bita na wagę jakość modeli wyraźnie spada – takich wariantów nie polecamy.
QUALITY_MIN_BITS = 3.4
# Gdy w karcie jest co najmniej tyle modelu, do RAM trafia kilka warstw i spowolnienie
# jest umiarkowane – nie piszemy wtedy „kilka razy wolniej”.
MOSTLY_IN_VRAM = 0.9
LEVEL_RANK = {FitLevel.RED: 0, FitLevel.YELLOW: 1, FitLevel.GREEN: 2}


# --------------------------------------------------------------------------- wartości pól

class Size(int):
    """Bajty – wyświetlane jako „6,1 GB”."""


class Context(int):
    """Tokeny – wyświetlane jako „8k”."""


class AudioContext(Context):
    """Kontekst modelu mowy – „16k (≈ 3 min 17 s nagrania)”."""

    def __new__(cls, value: int, tokens_per_second: float):
        obj = super().__new__(cls, value)
        obj.tokens_per_second = tokens_per_second
        return obj


def _ctx(spec: ModelSpec, value: int) -> Context:
    a = spec.audio
    return AudioContext(value, a.tokens_per_second) if a and a.tokens_per_second else Context(value)


class Params(int):
    """Liczba parametrów – wyświetlana jako „3,3B”."""


class Pct(float):
    """Ułamek 0–1 – wyświetlany jako „52”."""


class Text(str):
    """Klucz i18n wstawiany jako pole (np. „w całości w karcie”)."""


@dataclass(frozen=True)
class Line:
    key: str
    fields: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Explanation:
    summary: Line
    need: tuple[Line, ...]
    computer: tuple[Line, ...]
    actions_title: str               # "expl.act.title" albo "expl.spare.title"
    actions: tuple[Line, ...]
    notes: tuple[Line, ...]


def _format(value, lang: Optional[str]):
    if isinstance(value, Size):
        return fmt_gb(int(value), lang)
    if isinstance(value, AudioContext):
        duration = fmt_duration(int(value) / value.tokens_per_second)
        return f"{fmt_context(int(value))} ({t('audio.seconds', lang, duration=duration)})"
    if isinstance(value, Context):
        return fmt_context(int(value))
    if isinstance(value, Params):
        return fmt_params(int(value), lang)
    if isinstance(value, Pct):
        return fmt_number(float(value) * 100, 0, lang)
    if isinstance(value, Text):
        return t(str(value), lang)
    return value


def render(line: Line, lang: Optional[str] = None) -> str:
    return t(line.key, lang, **{k: _format(v, lang) for k, v in line.fields.items()})


def render_list(lines: Sequence[Line], lang: Optional[str] = None) -> list[str]:
    """Punkty listy kończą się przecinkiem, ostatni – kropką (jak w makiecie wyjaśnienia)."""
    texts = [render(line, lang) for line in lines]
    if texts and texts[-1].endswith(","):
        texts[-1] = texts[-1][:-1] + "."
    return texts


# --------------------------------------------------------------------------- pomocnicze

def _bits_per_weight(spec: ModelSpec, quants: QuantTable) -> Optional[float]:
    """Klasa jakości wariantu w bitach na wagę – z NAZWY kwantyzacji, nie z rozmiaru pliku.

    Warianty _L / _XL (np. Q3_K_XL od bartowskiego, UD-Q4_K_XL od unsloth) to ta sama
    kwantyzacja z dokładniejszymi warstwami wejścia/wyjścia: plik większy, ale klasa
    jakości bazowej (Q3_K_XL ≈ Q3_K_M, a nie lepsze od Q4_K_S). Rozmiar pliku jest
    ostatnią deską ratunku dla nazw spoza tabeli."""
    name = spec.quant.upper()
    bits = quants.bits(name)
    if bits is None:
        m = re.fullmatch(r"(Q\d_K)(?:_(?:XL|L|M|S))?", name)
        if m:
            bits = quants.bits(m.group(1)) or quants.bits(m.group(1) + "_M")
    if bits is None and spec.weights_bytes and spec.params_total:
        bits = spec.weights_bytes * 8 / spec.params_total
    return bits


_NOMINAL = re.compile(r"^I?Q(\d)")


def _quality(spec: ModelSpec, quants: QuantTable) -> Optional[tuple[int, float]]:
    """Klucz jakości: (klasa bitowa z nazwy, bity na wagę).

    Klasa z nazwy rozstrzyga pierwsza: Q3_K_L (4,27 bita na wagę, bo część warstw trzyma
    wyżej) to wciąż klasa 3-bitowa i jest gorsza od IQ4_XS (4,25) – tak wychodzi w pomiarach
    jakości llama.cpp. Bity na wagę porządkują warianty wewnątrz klasy."""
    bits = _bits_per_weight(spec, quants)
    if bits is None:
        return None
    name = spec.quant.upper()
    m = _NOMINAL.match(name)
    nominal = int(m.group(1)) if m else {"F32": 32, "F16": 16, "BF16": 16}.get(name, int(bits))
    return nominal, bits


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s} s" if s < 60 else f"{s // 60} min {s % 60:02d} s"


def audio_label(spec: ModelSpec, settings: RunSettings) -> Optional[str]:
    """„model mowy Orpheus (dekoder SNAC 24 kHz) · ≈ 1 min 39 s nagrania” albo None."""
    a = spec.audio
    if a is None:
        return None
    text = t(f"audio.kind.{a.kind}")
    if a.name:
        text += f" {a.name}"
    if a.decoder:
        text += " " + t("audio.decoder", decoder=a.decoder)
    if a.tokens_per_second:
        # Kontekst jest już w nagłówku obok – tu tylko, ile to sekund nagrania.
        text += " · " + t("audio.seconds",
                          duration=fmt_duration(settings.context_tokens / a.tokens_per_second))
    return text


def image_label(spec: ModelSpec, settings: RunSettings) -> Optional[str]:
    """„model obrazów SDXL · 1024×1024” (· „2 obrazy naraz”) albo None."""
    info = spec.image
    if info is None:
        return None
    from aibobr.calculator.image import resolution
    w, h = resolution(spec, settings)
    text = t("image.kind") + (f" {info.name}" if info.name else "") + f" · {w}×{h}"
    if settings.image_batch > 1:
        text += " · " + t("image.batch", n=settings.image_batch)
    return text


def _where(level: FitLevel, cpu_only: bool = False) -> Text:
    if level == FitLevel.GREEN:
        return Text("expl.where.vram")
    return Text("expl.where.ram_only" if cpu_only else "expl.where.ram")


class Advisor:
    def __init__(self, quants: QuantTable, catalog: Sequence[ModelSpec] = ()):
        self.quants = quants
        self.catalog = catalog

    def _eval(self, spec: ModelSpec, hw: HardwareProfile, settings: RunSettings) -> Optional[Verdict]:
        try:
            return evaluate(spec, hw, settings, self.quants)
        except MissingModelData:
            return None

    def _variants(self, model: LoadedModel) -> list[ModelSpec]:
        specs = []
        for v in model.variants:
            spec = model.spec(v.quant)
            q = _quality(spec, self.quants)
            if q is not None:
                specs.append((q, spec.weights_bytes or 0, spec))
        return [s for *_, s in sorted(specs, key=lambda p: (p[0], p[1]), reverse=True)]

    # ------------------------------------------------------------------ werdykt
    @staticmethod
    def summary(v: Verdict, spec: ModelSpec) -> Line:
        if v.level == FitLevel.GREEN:
            return Line("verdict.green")
        if v.level == FitLevel.RED:
            return Line("verdict.red", {"short": Size(v.shortfall)})
        if v.capacity.cpu_only:
            return Line("verdict.yellow_cpu")
        if spec.is_moe:
            return Line("verdict.yellow_moe", {"pct": Pct(v.vram_fraction)})
        if v.vram_fraction >= MOSTLY_IN_VRAM:
            return Line("verdict.yellow_mostly", {"pct": Pct(v.vram_fraction)})
        return Line("verdict.yellow", {"pct": Pct(v.vram_fraction)})

    # ------------------------------------------------------------------ sekcje
    @staticmethod
    def _need(v: Verdict, settings: RunSettings, spec: Optional[ModelSpec] = None) -> list[Line]:
        n = v.need
        decoder_name = (spec.audio.decoder if spec and spec.audio else None) or "?"
        lines = [
            Line("expl.need.title", {"total": Size(n.total)}),
            Line("expl.need.weights_est" if n.weights_estimated else "expl.need.weights",
                 {"size": Size(n.weights_total)}),
            Line("expl.need.kv_est" if n.kv_estimated else
                 "expl.need.kv_swa" if n.kv_sliding else "expl.need.kv",
                 {"size": Size(n.kv_cache), "ctx": Context(settings.context_tokens)}),
        ]
        if n.decoder:
            lines.append(Line("expl.need.decoder_cpu" if n.decoder_in_ram else "expl.need.decoder",
                              {"size": Size(n.decoder), "decoder": decoder_name}))
        lines.append(Line("expl.need.overhead", {"size": Size(n.overhead)}))
        if n.embd_in_ram:
            lines.append(Line("expl.need.embd_ram", {"size": Size(n.embd_in_ram)}))
        return lines

    @staticmethod
    def _computer(v: Verdict, spec: ModelSpec, hw: HardwareProfile,
                  settings: RunSettings) -> list[Line]:
        cap, lines = v.capacity, []
        if cap.cpu_only:
            lines.append(Line("expl.pc.no_gpu"))
        for g in cap.gpus_used:
            lines.append(Line("expl.pc.gpu", {
                "name": g.name, "usable": Size(max(0, (g.vram_total or 0) - settings.vram_reserve))}))
        image = spec.image is not None
        if v.level == FitLevel.GREEN:
            lines.append(Line("expl.pc.img_fits" if image else "expl.pc.fits", {"margin": Size(v.margin or 0)}))
        elif v.level == FitLevel.YELLOW and image:
            lines.append(Line("expl.pc.img_slow_cpu" if cap.cpu_only else "expl.pc.img_stream",
                              {"cpu": hw.cpu_name or "?"}))
        elif v.level == FitLevel.YELLOW:
            if cap.cpu_only:
                lines.append(Line("expl.pc.all_in_ram", {"size": Size(v.in_ram)}))
            else:
                lines.append(Line("expl.pc.spill", {"size": Size(v.spill)}))
            if not cap.cpu_only and v.vram_fraction >= MOSTLY_IN_VRAM:
                lines.append(Line("expl.pc.slow_little"))
            elif spec.is_moe and spec.params_active and spec.params_total:
                lines.append(Line("expl.pc.slow_moe", {
                    "active": Params(spec.params_active), "total": Params(spec.params_total)}))
            else:
                ram = f" ({hw.ram_description})" if hw.ram_description else ""
                lines.append(Line("expl.pc.slow" if not cap.cpu_only else "expl.pc.slow_cpu",
                                  {"cpu": hw.cpu_name or "?", "ram": ram}))
            if hw.ram_used is not None and hw.ram_total is not None \
                    and v.in_ram > hw.ram_total - hw.ram_used:
                lines.append(Line("expl.pc.ram_busy", {"used": Size(hw.ram_used)}))
        else:
            lines.append(Line("expl.pc.short", {
                "short": Size(v.shortfall), "capacity": Size(cap.total)}))
        return lines

    def _spare(self, model: LoadedModel, spec: ModelSpec, hw: HardwareProfile,
               settings: RunSettings) -> list[Line]:
        lines = []
        limit = spec.max_context or 131072
        best_ctx = None
        ctx = settings.context_tokens * 2
        while ctx <= limit:
            v = self._eval(spec, hw, replace(settings, context_tokens=ctx))
            if not v or v.level != FitLevel.GREEN or v.low_margin:
                break
            best_ctx, ctx = ctx, ctx * 2
        if best_ctx:
            lines.append(Line("expl.spare.context", {"ctx": _ctx(spec, best_ctx)}))

        current = _quality(spec, self.quants) or (0, 0.0)
        for other in self._variants(model):                 # od najdokładniejszej
            if (_quality(other, self.quants) or (0, 0.0)) <= current:
                break
            v = self._eval(other, hw, settings)
            # „Zapas” nie może prowadzić do wariantu, przy którym sami ostrzegamy o małym zapasie.
            if v and v.level == FitLevel.GREEN and not v.low_margin:
                lines.append(Line("expl.spare.quant", {
                    "quant": other.quant, "size": Size(v.need.weights)}))
                break
        return lines or [Line("expl.spare.none")]

    def _actions(self, model: LoadedModel, spec: ModelSpec, v: Verdict, hw: HardwareProfile,
                 settings: RunSettings) -> list[Line]:
        current = _quality(spec, self.quants) or (99, 99.0)
        smaller = [s for s in self._variants(model)
                   if (_quality(s, self.quants) or (0, 0.0)) < current
                   and (_bits_per_weight(s, self.quants) or 0) >= QUALITY_MIN_BITS]
        contexts = [c for c in (16384, 8192, 4096, 2048) if c < settings.context_tokens]
        kv = replace(settings, kv_cache_type=KvCacheType.Q8_0) \
            if settings.kv_cache_type == KvCacheType.F16 else None

        cpu = v.capacity.cpu_only
        # Model mowy: dekoder dźwięku jest mały i na procesorze działa znośnie (SPEC §5).
        decoder_tip = bool(spec.audio and spec.audio.decoder_bytes and not cpu and settings.decoder_on_gpu)
        # MoE w strefie żółtej jest używalny (SPEC §7) – zamiast „potrzebny mniejszy model”
        # podpowiadamy opcję runtime'u, która trzyma ekspertów w RAM, a resztę na karcie.
        moe_tip = [Line("expl.act.moe_offload")] \
            if spec.is_moe and v.level == FitLevel.YELLOW and not cpu else []
        if cpu and v.level == FitLevel.YELLOW:
            # Bez karty zielony jest niemożliwy; mniejszy model = szybsze odpowiedzi.
            return [Line("expl.act.cpu_only")] + self._catalog(spec, hw, settings)
        if cpu:
            targets = [FitLevel.YELLOW]
        elif v.level == FitLevel.YELLOW:
            targets = [FitLevel.GREEN]
        else:
            targets = [FitLevel.GREEN, FitLevel.YELLOW]
        for target in targets:
            lines = []
            for s in smaller:                                       # najlepsza jakość, która wystarczy
                r = self._eval(s, hw, settings)
                if r and LEVEL_RANK[r.level] >= LEVEL_RANK[target]:
                    lines.append(Line("expl.act.quant", {
                        "quant": s.quant, "size": Size(r.need.weights), "where": _where(target, cpu)}))
                    break
            for c in contexts:                                      # najdłuższy, który wystarczy
                r = self._eval(spec, hw, replace(settings, context_tokens=c))
                if r and LEVEL_RANK[r.level] >= LEVEL_RANK[target]:
                    lines.append(Line("expl.act.context", {"ctx": _ctx(spec, c), "where": _where(target, cpu)}))
                    break
            if kv:
                r = self._eval(spec, hw, kv)
                if r and LEVEL_RANK[r.level] >= LEVEL_RANK[target]:
                    lines.append(Line("expl.act.kv", {"where": _where(target, cpu)}))
            if decoder_tip:                                         # bez utraty jakości – na początek
                r = self._eval(spec, hw, replace(settings, decoder_on_gpu=False))
                if r and LEVEL_RANK[r.level] >= LEVEL_RANK[target]:
                    lines.insert(0, Line("expl.act.decoder_cpu", {
                        "size": Size(spec.audio.decoder_bytes), "where": _where(target, cpu)}))
            if lines:
                return moe_tip + lines + self._catalog(spec, hw, settings)
        if moe_tip:
            return moe_tip + self._catalog(spec, hw, settings)

        # Żadna pojedyncza zmiana nie wystarcza – sprawdzamy wszystkie naraz.
        lowest = smaller[-1] if smaller else spec
        combo_ctx = min([4096, settings.context_tokens])
        combo = replace(kv or settings, context_tokens=combo_ctx)
        r = self._eval(lowest, hw, combo)
        if r and LEVEL_RANK[r.level] > LEVEL_RANK[v.level]:
            lines = [Line("expl.act.combo", {"quant": lowest.quant, "ctx": Context(combo_ctx),
                                             "where": _where(r.level, cpu)})]
        else:
            key = ("expl.act.nothing_cpu" if cpu else
                   "expl.act.nothing_red" if v.level == FitLevel.RED else "expl.act.nothing")
            lines = [Line(key,
                          {"quant": lowest.quant, "ctx": Context(combo_ctx)})]
        return lines + self._catalog(spec, hw, settings)

    def _catalog(self, spec: ModelSpec, hw: HardwareProfile, settings: RunSettings) -> list[Line]:
        """Największy model z katalogu, który zmieści się w całości w karcie (etap 6).
        Bez karty nic nie jest zielone, więc lista jest pusta."""
        best = None
        for other in self.catalog:
            if other.name == spec.name or (spec.origin and other.origin == spec.origin):
                continue                                   # ten sam model
            r = self._eval(other, hw, settings)
            if r and r.level == FitLevel.GREEN and not r.low_margin and (
                    best is None or (other.params_total or 0) > (best.params_total or 0)):
                best = other
        return [Line("expl.act.model", {"name": best.name, "quant": best.quant})] if best else []

    @staticmethod
    def _notes(v: Verdict, spec: ModelSpec, settings: RunSettings) -> list[Line]:
        notes = []
        if v.low_margin:
            notes.append(Line("note.low_margin"))
        if v.context_exceeds_model:
            notes.append(Line("note.context_exceeds", {
                "ctx": Context(settings.context_tokens), "max": Context(spec.max_context or 0)}))
        if v.need.is_estimate:
            notes.append(Line("note.estimate"))
        a = spec.audio
        if a is not None:
            kind = Text(f"audio.kind.{a.kind}")
            if a.by == "name":                              # decyzja autora: wyraźny komunikat
                notes.append(Line("note.audio_by_name", {"kind": kind}))
            if not a.decoder_bytes:
                notes.append(Line("note.audio_decoder_unknown", {"kind": kind}))
            elif a.decoder_estimated:
                notes.append(Line("note.audio_decoder_estimate", {"size": Size(a.decoder_bytes)}))
        notes += [Line(w) for w in v.warnings]
        return notes

    # ------------------------------------------------------------------ całość
    # ------------------------------------------------------------------ modele obrazów
    def _image_summary(self, v: Verdict) -> Line:
        if v.level == FitLevel.GREEN:
            return Line("verdict.image_green_all" if v.all_at_once else "verdict.image_green_stages")
        if v.level == FitLevel.RED:
            return Line("verdict.red", {"short": Size(v.shortfall)})
        return Line("verdict.image_cpu" if v.capacity.cpu_only else "verdict.image_yellow")

    @staticmethod
    def _image_need(v: Verdict) -> list[Line]:
        n = v.need
        stages = n.stages
        lines = [Line("expl.img.title", {"peak": Size(n.peak), "stage": Text(f"expl.img.stage.{n.peak_stage}")})]
        lines.append(Line("expl.img.encode_cpu" if n.encoders_on_cpu else "expl.img.encode",
                          {"size": Size(stages["encode"]), "enc": Size(n.encoders)}))
        lines.append(Line("expl.img.sample", {"size": Size(stages["sample"]), "main": Size(n.main),
                                              "comp": Size(n.main_compute)}))
        lines.append(Line("expl.img.decode_tiled" if n.vae_tiled else "expl.img.decode",
                          {"size": Size(stages["decode"]), "vae": Size(n.vae), "comp": Size(n.vae_compute)}))
        lines.append(Line("expl.img.overhead", {"size": Size(n.overhead)}))
        lines.append(Line("expl.img.all", {"size": Size(n.all_at_once)}))
        return lines

    def _image_variant(self, spec, hw, settings, side: int, batch: int) -> Optional[Verdict]:
        return self._eval(spec, hw, replace(settings, image_width=side, image_height=side, image_batch=batch))

    def _image_actions(self, spec: ModelSpec, v: Verdict, hw: HardwareProfile,
                       settings: RunSettings) -> list[Line]:
        cpu = v.capacity.cpu_only
        n = v.need
        target = FitLevel.YELLOW if cpu or v.level == FitLevel.RED else FitLevel.GREEN
        where = _where(target, cpu)
        lines = []
        if n.batch > 1:
            r = self._image_variant(spec, hw, settings, n.width, 1)
            if r and LEVEL_RANK[r.level] >= LEVEL_RANK[target]:
                lines.append(Line("expl.act.img_batch", {"where": where}))
        for side in (1024, 768, 512):
            if side >= max(n.width, n.height):
                continue
            r = self._image_variant(spec, hw, settings, side, n.batch)
            if r and LEVEL_RANK[r.level] >= LEVEL_RANK[target]:
                lines.append(Line("expl.act.img_res", {"res": f"{side}×{side}", "where": where}))
                break
        if not lines:
            lines.append(Line("expl.act.img_nothing"))
        return lines

    def _image_spare(self, spec: ModelSpec, v: Verdict, hw: HardwareProfile,
                     settings: RunSettings) -> list[Line]:
        n, best = v.need, None
        for side in (1024, 1280, 1536, 2048):
            if side <= max(n.width, n.height):
                continue
            r = self._image_variant(spec, hw, settings, side, n.batch)
            if not r or r.level != FitLevel.GREEN or r.low_margin:
                break
            best = side
        if best:
            return [Line("expl.spare.img_res", {"res": f"{best}×{best}"})]
        return [Line("expl.spare.none")]

    @staticmethod
    def _image_notes(v: Verdict, spec: ModelSpec) -> list[Line]:
        n, info, notes = v.need, spec.image, []
        if v.low_margin:
            notes.append(Line("note.img_low_margin"))
        if info.family is None:
            notes.append(Line("note.img_unknown"))
        if info.assumed:
            size = sum(info.part(r) for r in info.assumed)
            notes.append(Line("note.img_assumed", {"family": info.name or "?", "size": Size(size)}))
        if n.encoders_on_cpu:
            notes.append(Line("note.img_encoders_cpu", {"size": Size(n.encoders)}))
        if n.vae_tiled:
            notes.append(Line("note.img_tiled"))
        if not info.measured:
            notes.append(Line("note.img_estimate"))
        notes += [Line(w) for w in v.warnings]
        return notes

    def _explain_image(self, spec, hw, settings) -> tuple[Verdict, Explanation]:
        v = evaluate(spec, hw, settings, self.quants)
        if v.level == FitLevel.GREEN:
            title, actions = "expl.spare.title", self._image_spare(spec, v, hw, settings)
        else:
            title, actions = "expl.act.title", self._image_actions(spec, v, hw, settings)
        return v, Explanation(
            summary=self._image_summary(v), need=tuple(self._image_need(v)),
            computer=tuple(self._computer(v, spec, hw, settings)),
            actions_title=title, actions=tuple(actions), notes=tuple(self._image_notes(v, spec)))

    # ------------------------------------------------------------------ całość
    def explain(self, model: LoadedModel, spec: ModelSpec, hw: HardwareProfile,
                settings: RunSettings) -> tuple[Verdict, Explanation]:
        if spec.image is not None:
            return self._explain_image(spec, hw, settings)
        v = evaluate(spec, hw, settings, self.quants)   # IncompleteProfileError leci wyżej
        if v.level == FitLevel.GREEN:
            title, actions = "expl.spare.title", self._spare(model, spec, hw, settings)
        else:
            title, actions = "expl.act.title", self._actions(model, spec, v, hw, settings)
        return v, Explanation(
            summary=self.summary(v, spec),
            need=tuple(self._need(v, settings, spec)),
            computer=tuple(self._computer(v, spec, hw, settings)),
            actions_title=title, actions=tuple(actions),
            notes=tuple(self._notes(v, spec, settings)),
        )


def all_keys(explanation: Explanation) -> Iterable[str]:
    """Klucze użyte w wyjaśnieniu – do testów kompletności tłumaczeń."""
    yield explanation.summary.key
    yield explanation.actions_title
    for section in (explanation.need, explanation.computer, explanation.actions, explanation.notes):
        for line in section:
            yield line.key
            yield from (str(v) for v in line.fields.values() if isinstance(v, Text))
