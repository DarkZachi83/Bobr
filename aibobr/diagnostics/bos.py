"""Token początku (BOS): ile tokenów początku model faktycznie dostaje w Ollamie.

Skąd: eksperyment z 7.11.2026 (tests/experiments/bos) – szablon Ollamy dla Bielika-Minitrona z hf.co/… nie ma
tokenu początku, a szablon z pliku GGUF ma „{{bos_token}}”; z tokenem początku model zaliczył polskie znaki
3/3 zamiast 0/3, a odpowiedź zmieniła się w 6 z 15 zadań. Wcześniejsza diagnoza „podwójnego BOS” była błędna –
dlatego tu niczego nie zakładamy, tylko mierzymy.

Trzy źródła:
1. czego model oczekuje – opis z pliku GGUF w /api/show (tokenizer.ggml.add_bos_token, bos_token_id,
   tekst tokenu ze słownika) **i szablon czatu z pliku GGUF** (`tokenizer.chat_template`; gdy Ollama go nie
   podaje – z nagłówka pliku modelu, aibobr/diagnostics/gguf.py). Pomiar u autora 12.11.2026: Minitron z hf.co
   ma add_bos_token = false, a jego szablon GGUF zaczyna się od „{{bos_token}}” – token początku dodaje
   szablon, nie tokenizer; sama flaga dała fałszywe „model nie używa tokenu początku”;
2. czy szablon Ollamy zawiera ten token (tekst szablonu);
3. pomiar – dwa zapytania w trybie raw (bez szablonu), każde po zwolnieniu modelu (pamięć promptu nie zaniży
   liczby): „a” i „<tekst tokenu początku>a”. prompt_eval_count − 1 to liczba tokenów początku, które model
   dostaje bez tokenu w tekście (dodaje je sama Ollama) i z tokenem w tekście. Sam brak tokenu w szablonie
   niczego nie dowodzi – dla części modeli Ollama dodaje go sama.

Wynik: ok (jeden), missing (zero, a model oczekuje), double (dwa), not_used (model nie używa, nie dostaje),
unknown (nie da się ustalić – bez ostrzeżenia).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from aibobr.diagnostics import gguf

OK, MISSING, DOUBLE, NOT_USED, UNKNOWN = "ok", "missing", "double", "not_used", "unknown"
WARN = (MISSING, DOUBLE)
PROBE = "a"                          # jeden token w praktycznie każdym słowniku; sprawdzane: wynik musi być 0 albo 1
PROBE_CTX = 512
# Teksty tokenu początku, gdy słownik niedostępny – i znaczniki w szablonach (Go i Jinja).
KNOWN_BOS = ("<s>", "<bos>", "<|begin_of_text|>", "<|startoftext|>", "[BOS]")
TEMPLATE_MARKERS = ("bos_token", ".BOS")


@dataclass(frozen=True)
class BosCheck:
    model: str
    digest: Optional[str]
    status: str                      # ok / missing / double / not_used / unknown
    reason: str                      # szczegół, np. measured, no_model_info, not_released, cached, bos_not_special
    expects: Optional[bool]          # czy model oczekuje tokenu początku (flaga albo szablon GGUF); None – nie wiadomo
    bos_text: Optional[str]          # np. „<s>”
    template_has: Optional[bool]     # czy szablon Ollamy zawiera token początku
    without: Optional[int]           # tokeny początku bez tokenu w tekście (dodaje Ollama)
    with_text: Optional[int]         # tokeny początku, gdy tekst zaczyna się od tokenu początku
    effective: Optional[int]         # ile dostaje przy zwykłym zapytaniu (przez szablon)
    add_bos: Optional[bool] = None   # tokenizer.ggml.add_bos_token z pliku GGUF
    gguf_template_has: Optional[bool] = None   # czy szablon czatu z pliku GGUF podaje bos_token
    gguf_template_source: Optional[str] = None  # model_info / file / None (niedostępny)

    @property
    def warning(self) -> bool:
        return self.status in WARN

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BosCheck":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


def _bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def model_expectation(info: dict) -> tuple[Optional[bool], Optional[int]]:
    """(czy model oczekuje tokenu początku, jego id) z model_info."""
    info = info or {}
    return _bool(info.get("tokenizer.ggml.add_bos_token")), info.get("tokenizer.ggml.bos_token_id")


def bos_text_from(info: dict, bos_id: Optional[int]) -> Optional[str]:
    tokens = (info or {}).get("tokenizer.ggml.tokens")
    if isinstance(tokens, list) and isinstance(bos_id, int) and 0 <= bos_id < len(tokens):
        text = tokens[bos_id]
        return text if isinstance(text, str) and text else None
    return None


def template_has_bos(template: Optional[str], bos_text: Optional[str]) -> Optional[bool]:
    """Czy szablon Ollamy podaje token początku: jego tekst albo zmienna (bos_token, .BOS)."""
    if not template:
        return None
    markers = ((bos_text,) if bos_text else KNOWN_BOS) + TEMPLATE_MARKERS
    return any(m in template for m in markers)


def expectation(add_bos: Optional[bool], gguf_has: Optional[bool]) -> Optional[bool]:
    """Model oczekuje tokenu początku, gdy dodaje go tokenizer (flaga) albo szablon z pliku GGUF.
    Nie oczekuje – tylko gdy wiadomo to z szablonu GGUF; sama flaga false nie wystarcza (Minitron)."""
    if add_bos is True or gguf_has is True:
        return True
    if gguf_has is False:
        return False
    return None


def verdict(expects: Optional[bool], effective: Optional[int]) -> str:
    if effective is None:
        return UNKNOWN
    if effective >= 2:
        return DOUBLE
    if expects is False:
        return NOT_USED if effective == 0 else OK
    if effective == 1:
        return OK
    return MISSING if expects else UNKNOWN


class BosProbe:
    """Pomiar dla jednego modelu. bench – OllamaBench (show, digest, generate(raw=True), unload, is_loaded)."""

    def __init__(self, bench, clock: Callable[[], float], sleep: Callable[[float], None],
                 unload_timeout: float = 30.0):
        self.bench, self.clock, self.sleep, self.unload_timeout = bench, clock, sleep, unload_timeout

    def release(self, model: str) -> bool:
        start = self.clock()
        try:
            self.bench.unload(model)
            while self.bench.is_loaded(model):
                if self.clock() - start >= self.unload_timeout:
                    return False
                self.sleep(0.25)
        except Exception:
            return False
        return True

    def count(self, model: str, text: str) -> tuple[Optional[int], str]:
        """Liczba tokenów promptu od zera (po zwolnieniu). Przy pamięci promptu – jedna powtórka."""
        for _ in range(2):
            if not self.release(model):
                return None, "not_released"
            g = self.bench.generate(model, text, PROBE_CTX, 1, raw=True)
            if not g.prompt_cached:
                return g.prompt_tokens, "measured"
        return None, "cached"

    def run(self, model: str) -> BosCheck:
        try:
            digest = self.bench.digest(model)
        except Exception:
            digest = None
        try:
            show = self.bench.show(model)
        except Exception:
            show = {}
        info = show.get("model_info") or {}
        add_bos, bos_id = model_expectation(info)
        verbose: dict = {}

        def full_info() -> dict:                      # słownik tokenów tylko w trybie verbose – duży, raz
            if not verbose:
                try:
                    verbose.update(self.bench.show(model, verbose=True).get("model_info") or {})
                except Exception:
                    verbose["_failed"] = True
            return verbose

        bos_text = bos_text_from(info, bos_id)
        if bos_text is None and isinstance(bos_id, int):
            bos_text = bos_text_from(full_info(), bos_id)
        chat, source = info.get("tokenizer.chat_template"), "model_info"
        if not isinstance(chat, str) and show:
            chat = full_info().get("tokenizer.chat_template")
        if not isinstance(chat, str):
            chat, source = None, None
            path = gguf.model_path(show)
            if path is not None:
                try:
                    chat = gguf.read_keys(path, ["tokenizer.chat_template"]).get("tokenizer.chat_template")
                    source = "file" if isinstance(chat, str) else None
                except (OSError, gguf.GgufError):
                    chat = None
        gguf_has = ("bos_token" in chat) if isinstance(chat, str) else None
        expects = expectation(add_bos, gguf_has)
        template = show.get("template")
        has = template_has_bos(template, bos_text)

        def result(status, reason, without=None, with_text=None, effective=None):
            return BosCheck(model, digest, status, reason, expects, bos_text, has, without, with_text, effective,
                            add_bos, gguf_has, source)

        if not show:
            return result(UNKNOWN, "no_model_info")
        try:
            plain, reason = self.count(model, PROBE)
            if plain is None:
                return result(UNKNOWN, reason)
            without = plain - 1
            if without not in (0, 1):                      # „a” nie jest jednym tokenem – pomiar nic nie mówi
                return result(UNKNOWN, "probe_not_single_token")
            with_text = None
            if bos_text:
                prefixed, reason = self.count(model, bos_text + PROBE)
                if prefixed is None:
                    return result(UNKNOWN, reason, without)
                with_text = prefixed - 1
                if with_text - without not in (0, 1):      # tekst nie jest rozpoznawany jako token specjalny
                    return result(UNKNOWN, "bos_not_special", without, with_text)
        finally:
            try:
                self.release(model)                        # po pomiarze karta wolna, jak po benchmarku
            except Exception:
                pass
        if has is None:
            return result(UNKNOWN, "no_template", without, with_text)
        if has and with_text is None:
            return result(UNKNOWN, "no_bos_text", without, with_text)
        effective = with_text if has else without
        return result(verdict(expects, effective), "measured", without, with_text, effective)
