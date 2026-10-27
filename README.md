# AI Bóbr (AI BEAVER)

**AI Hardware Checker**

🚧 **Work in progress / Prace trwają** – the core calculator works, no GUI yet. / Rdzeń liczy, GUI jeszcze nie ma.

**EN:** Will this local LLM run on my PC, and how fast? AI Bóbr reads a specific model
(a Hugging Face link or a local `.gguf` file), scans your RAM and VRAM, and shows a memory
bar: fits in VRAM, spills into RAM (slow), or does not fit at all – with a plain-language
explanation of why and what to change.

**PL:** Czy ten lokalny model językowy pójdzie na moim komputerze i jak szybko? AI Bóbr
czyta konkretny model (link do Hugging Face albo plik `.gguf`), sprawdza RAM i VRAM
i pokazuje pasek pamięci: mieści się w VRAM, wchodzi do RAM (wolno) albo się nie mieści –
z prostym wyjaśnieniem dlaczego i co zmienić.

Linux · Windows · Python 3.10+ · [specyfikacja / specification (PL)](docs/SPEC.md)

```
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

Designed and built by Rafał together with **Claude (Anthropic)** as an AI pair-engineer.
Zaprojektowane i zbudowane przez Rafała razem z **Claude (Anthropic)**.

[MIT](LICENSE)
