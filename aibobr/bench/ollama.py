"""Ollama jako silnik pomiarów (etap 12a): Bóbr nie uruchamia modeli sam, tylko pyta Ollamę.

Odpowiedź /api/generate niesie gotowe czasy w nanosekundach: prompt_eval_count/_duration
(przetwarzanie promptu), eval_count/_duration (generowanie), load_duration (wczytanie modelu).
/api/ps mówi, ile wczytanego modelu siedzi w karcie (size_vram z size).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
                 temperature: float = 0, seed: int = 42) -> Generation:
        payload = {"model": model, "prompt": prompt, "stream": False, "think": False, "keep_alive": "5m",
                   "options": {"num_ctx": num_ctx, "num_predict": num_predict,
                               "temperature": temperature, "seed": seed}}
        r = self.http.post_json(f"{self.host}/api/generate", payload, timeout=TIMEOUT)
        ns = 1e9
        return Generation(
            prompt_tokens=int(r.get("prompt_eval_count") or 0),
            prompt_seconds=float(r.get("prompt_eval_duration") or 0) / ns,
            gen_tokens=int(r.get("eval_count") or 0),
            gen_seconds=float(r.get("eval_duration") or 0) / ns,
            load_seconds=float(r.get("load_duration") or 0) / ns,
            text=str(r.get("response") or ""))

    def unload(self, model: str) -> None:
        """Zwalnia model z pamięci od razu (keep_alive 0) – po teście karta jest wolna dla innych programów."""
        self.http.post_json(f"{self.host}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)

    def gpu_share(self, model: str) -> Optional[float]:
        """Jaka część wczytanego modelu jest w karcie (0–1); None, gdy Ollama go nie pokazuje."""
        data = self.http.get_json(f"{self.host}/api/ps")
        for m in (data.get("models") or []) if isinstance(data, dict) else []:
            if m.get("name") == model or m.get("model") == model:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                return vram / size if size else None
        return None
