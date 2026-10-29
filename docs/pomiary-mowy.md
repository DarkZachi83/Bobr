# Pomiar modelu mowy (etap 9) – Orpheus 3B

Cel: sprawdzić, ile pamięci karty naprawdę zajmują **część językowa** (llama.cpp) i **dekoder
dźwięku** (SNAC w Pythonie), i porównać to z tym, co liczy AI Bóbr. Wynik trafi do testu
regresyjnego, a szacunek dekodera (0,5 GB) zastąpi pomiar.

Potrzebne: karta NVIDIA, ok. 2,5 GB na model, ok. 4 GB na PyTorch z CUDA. Na Archu pakiet
llama.cpp z CUDA dociąga też CUDA Toolkit (kilka GB) i kompiluje się kilkanaście minut.

## 1. Czy AI Bóbr rozpoznaje Orpheusa? (bez pobierania modelu)

```bash
python main.py --cli hf.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF --explain
```

W linii „Model:” powinno być `model mowy Orpheus (dekoder SNAC 24 kHz) · ≈ 1 min 39 s nagrania`.

## 2. Przygotowanie (raz)

```bash
# programy systemowe: llama.cpp z CUDA (AUR) i PortAudio (skrypt odtwarza dźwięk)
yay -S llama.cpp-cuda
sudo pacman -S --needed portaudio

# model i skrypt – na dysku z miejscem
mkdir -p /mnt/dane/orpheus && cd /mnt/dane/orpheus
curl -LO https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF/resolve/main/orpheus-3b-0.1-ft-q4_k_m.gguf
git clone https://github.com/isaiahbjork/orpheus-tts-local

# środowisko Pythona 3.12 (uv sam je pobierze) z PyTorchem dla CUDA 12.8 – karty RTX 50xx
# (Blackwell) potrzebują co najmniej tej wersji
uv venv --python 3.12 env
source env/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install snac numpy sounddevice requests
```

`requirements.txt` skryptu celowo pomijamy: wymienia pakiet `wave`, który jest częścią
Pythona, a `torch` bez wskazania wersji CUDA.

## 3. Pomiar – przebieg A (dekoder na karcie)

Trzy terminale:

**Terminal 1 – pomiar** (z folderu AI Bóbra, uruchom jako pierwszy):
```bash
python tools/measure_vram.py --label "orpheus Q4_K_M, dekoder na GPU" \
    --model /mnt/dane/orpheus/orpheus-3b-0.1-ft-q4_k_m.gguf --context 8192
```

**Terminal 2 – model językowy:**
```bash
llama-server -m /mnt/dane/orpheus/orpheus-3b-0.1-ft-q4_k_m.gguf -ngl 99 -c 8192 --port 1234
```
Poczekaj na `server is listening on http://127.0.0.1:1234`.

**Terminal 3 – mowa** (głosy są angielskie, więc tekst po angielsku):
```bash
cd /mnt/dane/orpheus && source env/bin/activate
time python orpheus-tts-local/gguf_orpheus.py --voice tara --output test-gpu.wav \
    --text "Hello, this is the beaver. We are checking how much memory speech really needs."
```

Gdy skrypt zapisze `test-gpu.wav`, naciśnij **Enter w terminalu 1**. Wydruk i plik
`measure-….json` to wynik. llama-server zostaw włączony.

## 4. Pomiar – przebieg B (dekoder na procesorze)

Terminal 1 – nowy pomiar z `--label "orpheus Q4_K_M, dekoder na CPU"`. Terminal 3:
```bash
time env CUDA_VISIBLE_DEVICES= python orpheus-tts-local/gguf_orpheus.py --voice tara --output test-cpu.wav \
    --text "Hello, this is the beaver. We are checking how much memory speech really needs."
```
(Pusta zmienna ukrywa kartę przed PyTorchem – SNAC liczy procesor; llama-server dalej
używa karty.) Enter w terminalu 1.

## 5. Co odesłać

Oba wydruki z terminala 1 (albo pliki `measure-….json`) i dwa czasy z `time` (wiersz `real`).
Dobrze też posłuchać `test-gpu.wav` – jeśli to szum albo cisza, napisz.

## Gdy coś nie działa

- **Skrypt kończy się bez dźwięku / „No tokens”** – llama-server mógł inaczej oddać tokeny
  `<custom_token_…>`. Wtedy uruchom model w LM Studio (Developer → Start Server, port 1234),
  zamiast llama-server – tak opisuje to autor skryptu.
- **`OSError: PortAudio library not found`** – brakuje `portaudio` (punkt 2).
- **`no kernel image is available`** – PyTorch bez wsparcia RTX 50xx; zainstaluj go z indeksu
  `cu128`, jak w punkcie 2.
