# Pomiar modelu mowy (etap 9) – Orpheus 3B

Cel: sprawdzić, ile pamięci karty naprawdę zajmują **część językowa** (llama.cpp) i **dekoder
dźwięku** (SNAC w Pythonie), i porównać to z tym, co liczy AI Bóbr. Wynik trafi do testu
regresyjnego, a szacunek dekodera (0,5 GB) zastąpi pomiar.

Potrzebne: karta NVIDIA, ok. 2,5 GB na model i ok. 3 GB na środowisko Pythona z PyTorchem.

## 1. Czy AI Bóbr rozpoznaje Orpheusa? (bez pobierania modelu)

```bash
python main.py --cli hf.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF --explain
```

Program czyta tylko nagłówek pliku. W linii „Model:” powinno być
`model mowy Orpheus (dekoder SNAC 24 kHz) · ≈ 1 min 39 s nagrania` (dla kontekstu 8k).

## 2. Model i program do uruchomienia

```bash
mkdir -p ~/orpheus && cd ~/orpheus
curl -LO https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF/resolve/main/orpheus-3b-0.1-ft-q4_k_m.gguf
git clone https://github.com/isaiahbjork/orpheus-tts-local
uv venv env && source env/bin/activate
uv pip install -r orpheus-tts-local/requirements.txt
```

Część językowa – **llama.cpp** z CUDA (Arch: pakiet AUR `llama.cpp-cuda`; nazwa pakietu
może się różnić – w razie czego zbudowanie ze źródeł opisuje repozytorium llama.cpp):

```bash
llama-server -m ~/orpheus/orpheus-3b-0.1-ft-q4_k_m.gguf -ngl 99 -c 8192 --port 1234
```

Skrypt `gguf_orpheus.py` łączy się z `http://127.0.0.1:1234/v1/completions` – tym samym
adresem, który ma LM Studio, więc działa z llama-server bez zmian. **Jeśli skrypt nie
wygeneruje dźwięku** (llama-server może inaczej oddawać tokeny `<custom_token_…>`), uruchom
model w LM Studio (zakładka Developer → Start Server, port 1234) – tak opisuje to autor skryptu.

## 3. Pomiary (dwa przebiegi)

W **drugim terminalu**, z folderu AI Bóbra, uruchom narzędzie pomiaru **przed** llama-server:

```bash
python tools/measure_vram.py --label "orpheus Q4_K_M, dekoder na GPU" \
    --model ~/orpheus/orpheus-3b-0.1-ft-q4_k_m.gguf --context 8192
```

Gdy narzędzie poprosi: uruchom llama-server (punkt 2), a potem w trzecim terminalu:

```bash
cd ~/orpheus && source env/bin/activate
python orpheus-tts-local/gguf_orpheus.py --text "Cześć, tu bóbr. Sprawdzamy, ile pamięci zjada mowa." --output test.wav
```

Po zapisaniu `test.wav` naciśnij Enter w terminalu z pomiarem.

**Przebieg B – dekoder na procesorze:** to samo, ale skrypt uruchom tak:

```bash
CUDA_VISIBLE_DEVICES= python orpheus-tts-local/gguf_orpheus.py --text "…" --output test-cpu.wav
```

(Pusta zmienna ukrywa kartę przed PyTorchem – SNAC liczy wtedy procesor, a llama-server
dalej używa karty.) Przy okazji zanotuj, ile trwało generowanie w obu przebiegach.

## 4. Co odesłać

Oba pliki `measure-RRRRMMDD-GGMM.json` (albo wydruk z terminala) i czasy generowania.
Narzędzie pokazuje osobno każdy proces, np.:

```
  2.94 GB  pid 2222  llama-server -m …orpheus… -ngl 99 -c 8192
  0.62 GB  pid 3333  python orpheus-tts-local/gguf_orpheus.py --text …
```
