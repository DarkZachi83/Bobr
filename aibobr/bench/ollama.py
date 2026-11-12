"""Ollama jako silnik pomiarów (etap 12a): Bóbr nie uruchamia modeli sam, tylko pyta Ollamę.

Odpowiedź /api/generate niesie gotowe czasy w nanosekundach: prompt_eval_count/_duration
(przetwarzanie promptu), eval_count/_duration (generowanie), load_duration (wczytanie modelu).
/api/ps mówi, ile wczytanego modelu siedzi w karcie (size_vram z size).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from aibobr.sources.http import HttpClient

TIMEOUT = 900                 # długi prompt na samym procesorze potrafi liczyć się minutami


@dataclass(frozen=True)
class Generation:
    prompt_tokens: int
    prompt_seconds: float
    gen_tokens: int
    gen_seconds: float
    load_seconds: float
    text: str
    prompt_cached: Optional[int] = None   # prompt_eval_cached_count: tokeny promptu z pamięci; None = Ollama nie podaje
    total_seconds: Optional[float] = None  # total_duration – cały czas zapytania po stronie silnika
    done_reason: Optional[str] = None      # „stop” (model skończył) / „length” (limit tokenów)

    @property
    def pp_tps(self) -> Optional[float]:
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds > 0 else None

    @property
    def tg_tps(self) -> Optional[float]:
        return self.gen_tokens / self.gen_seconds if self.gen_seconds > 0 else None

    @property
    def ttft_ms(self) -> float:
        """Czas do pierwszego tokenu bez wczytywania modelu: przetworzenie promptu."""
        return self.prompt_seconds * 1000


class OllamaBench:
    def __init__(self, host: str, http=None):
        self.host = host.rstrip("/")
        self.http = http or HttpClient(timeout=TIMEOUT)

    def version(self) -> Optional[str]:
        data = self.http.get_json(f"{self.host}/api/version")
        return str(data.get("version")) if isinstance(data, dict) and data.get("version") else None

    def generate(self, model: str, prompt: str, num_ctx: int, num_predict: int,
                 temperature: float = 0, seed: int = 42,
                 on_chunk: Optional[Callable[[str], None]] = None, raw: bool = False) -> Generation:
        """on_chunk(tekst) – podgląd na żywo (etap 12c): Ollama wysyła odpowiedź kawałkami
        (stream). Czasy i tak przychodzą w ostatnim kawałku i liczy je serwer – pomiar się
        nie zmienia."""
        stream = on_chunk is not None and hasattr(self.http, "post_lines")
        payload = {"model": model, "prompt": prompt, "stream": stream, "think": False, "keep_alive": "5m",
                   "options": {"num_ctx": num_ctx, "num_predict": num_predict,
                               "temperature": temperature, "seed": seed}}
        if raw:                       # bez szablonu – tekst trafia do modelu tak, jak jest (diagnostyka wejścia)
            payload["raw"] = True
        url = f"{self.host}/api/generate"
        if stream:
            parts, r = [], {}
            for line in self.http.post_lines(url, payload, timeout=TIMEOUT):
                piece = line.get("response") or ""
                if piece:
                    parts.append(piece)
                    on_chunk(piece)
                if line.get("done"):
                    r = line
            text = "".join(parts)
        else:
            r = self.http.post_json(url, payload, timeout=TIMEOUT)
            text = str(r.get("response") or "")
            if on_chunk is not None and text:
                on_chunk(text)
        ns = 1e9
        return Generation(
            prompt_tokens=int(r.get("prompt_eval_count") or 0),
            prompt_seconds=float(r.get("prompt_eval_duration") or 0) / ns,
            gen_tokens=int(r.get("eval_count") or 0),
            gen_seconds=float(r.get("eval_duration") or 0) / ns,
            load_seconds=float(r.get("load_duration") or 0) / ns,
            text=text,
            prompt_cached=None if r.get("prompt_eval_cached_count") is None else int(r["prompt_eval_cached_count"]),
            total_seconds=float(r["total_duration"]) / ns if r.get("total_duration") is not None else None,
            done_reason=str(r["done_reason"]) if r.get("done_reason") else None)

    def show(self, model: str, verbose: bool = False) -> dict:
        """/api/show: szablon Ollamy i opis modelu z pliku GGUF (model_info). verbose dokłada słownik tokenów –
        duży, więc tylko gdy potrzebny tekst tokenu początku."""
        data = self.http.post_json(f"{self.host}/api/show", {"model": model, "verbose": verbose}, timeout=120)
        return data if isinstance(data, dict) else {}

    def digest(self, model: str) -> Optional[str]:
        """Skrót pliku modelu z /api/tags – ta sama nazwa po ponownym pobraniu może być innym modelem."""
        data = self.http.get_json(f"{self.host}/api/tags")
        for m in (data.get("models") or []) if isinstance(data, dict) else []:
            if m.get("name") == model or m.get("model") == model:
                return m.get("digest") or None
        return None

    def load(self, model: str, num_ctx: int) -> Optional[float]:
        """Wczytuje model bez liczenia promptu (puste zapytanie) z tym samym num_ctx, co zadanie –
        inny num_ctx to ponowne wczytanie. Zwraca load_duration silnika w s (None – brak pola)."""
        r = self.http.post_json(f"{self.host}/api/generate",
                                {"model": model, "keep_alive": "5m", "options": {"num_ctx": num_ctx}},
                                timeout=TIMEOUT)
        value = r.get("load_duration") if isinstance(r, dict) else None
        return float(value) / 1e9 if value is not None else None

    def unload(self, model: str) -> None:
        """Zwalnia model z pamięci od razu (keep_alive 0) – po teście karta jest wolna dla innych programów."""
        self.http.post_json(f"{self.host}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)

    def is_loaded(self, model: str) -> bool:
        """Czy Ollama ma model w pamięci (/api/ps) – potwierdzenie zwolnienia po keep_alive 0."""
        data = self.http.get_json(f"{self.host}/api/ps")
        return any(m.get("name") == model or m.get("model") == model
                   for m in (data.get("models") or []) if isinstance(data, dict))

    def gpu_share(self, model: str) -> Optional[float]:
        """Jaka część wczytanego modelu jest w karcie (0–1); None, gdy Ollama go nie pokazuje."""
        data = self.http.get_json(f"{self.host}/api/ps")
        for m in (data.get("models") or []) if isinstance(data, dict) else []:
            if m.get("name") == model or m.get("model") == model:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                return vram / size if size else None
        return None
