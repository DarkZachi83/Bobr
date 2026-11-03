"""Prompty do pomiarów wydajności o zadanej długości (etap 12a).

Każdy przebieg dostaje INNY początek promptu: Ollama pamięta przetworzony prompt (prompt cache)
i przy tym samym tekście drugi przebieg „przetwarzałby” go w ułamku sekundy – pomiar byłby fałszywy.
Długość liczymy z grubsza (ok. 4 znaki na token w angielskim); prawdziwą liczbę tokenów podaje Ollama.
"""
from __future__ import annotations

import random

CHARS_PER_TOKEN = 4.0
_WORDS = ("river beaver dam wood water stone forest bank current branch winter spring lodge family "
          "builder engineer valley meadow trail evening morning shadow light north south quiet").split()


def filler(tokens: int, run: int) -> str:
    """Tekst ok. `tokens` tokenów; `run` zmienia początek (inny prompt = bez pamięci podręcznej)."""
    rnd = random.Random(1000 + run)
    target = int(tokens * CHARS_PER_TOKEN)
    out = [f"Report {run}-{rnd.randint(100000, 999999)}."]
    length = len(out[0])
    while length < target:
        sentence = " ".join(rnd.choice(_WORDS) for _ in range(12)).capitalize() + "."
        out.append(sentence)
        length += len(sentence) + 1
    return " ".join(out)


def prompt_processing(context: int, run: int) -> str:
    """Prompt wypełniający ok. 85% kontekstu – zostaje miejsce na polecenie i odpowiedź."""
    body = filler(int(context * 0.85) - 40, run)
    return body + "\n\nIn one short sentence: what animal appears in the text above?"


def generation(run: int) -> str:
    return (f"Story {run}. Write a long, detailed story about a family of beavers building a dam "
            "through one whole year. Do not stop early.")
