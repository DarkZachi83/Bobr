# AI Bóbr (AI BEAVER) – specyfikacja

**AI Hardware Checker**

Wersja 0.2 (zatwierdzona) · 27.10.2026 · licencja MIT
Autor: Rafał Zacharski, współpraca: Claude (Anthropic)

Dokument zbiera decyzje podjęte przed rozpoczęciem pisania kodu. Każda sekcja
kończy się tym, co zostało ustalone. Otwarte kwestie są na końcu.

---

## 1. Cel

Program odpowiada na jedno pytanie: **czy ten konkretny model językowy pójdzie na
moim komputerze, i jak szybko?** Adresatem jest osoba, która chce uruchomić model
lokalnie (Ollama, LM Studio, Jan), ale gubi się w pojęciach VRAM, kwantyzacja,
kontekst.

Program pokazuje werdykt w formie graficznej (pasek), a na żądanie wyjaśnia go
prostym językiem i podpowiada, co zmienić.

## 2. Zakres

| Platforma | Status |
|---|---|
| Linux | obsługiwany |
| Windows | obsługiwany |
| macOS | **poza zakresem w wersji 1** (opcja A). Program uruchomi się, ale wykrywanie zgłosi nieobsługiwaną platformę i odeśle do trybu Zaawansowanego. Wsparcie eksperymentalne (opcja B) w planie rozwoju. |

Uzasadnienie: autor nie ma sprzętu Apple Silicon, więc nie może sprawdzić logiki
pamięci zunifikowanej. Lepiej, żeby program mówił „nie wiem” niż podawał
niesprawdzony werdykt. Interfejs detekcji od początku przewiduje trzecią platformę.

## 3. Tryby pracy

**Automatyczny:** wskaż model → skan systemu → werdykt.
Kontekst przyjmuje wartość domyślną 8192 tokenów; werdykt mówi to wprost
(„przy kontekście 8k”).

**Zaawansowany:** skan systemu (wyniki wpisane w pola do edycji) → parametry →
wskaż model → werdykt.

Tryb Zaawansowany:
- zaczyna od wyników skanu, a nie od pustych pól (poprawia się tylko błędną wartość),
- pozwala sprawdzić sprzęt, którego jeszcze nie ma („czy po zakupie karty 16 GB…”),
- jest jednocześnie ręczną korektą, gdy wykrywanie zawiedzie.

Parametry w trybie Zaawansowanym (pola liczbowe, **bez suwaków**):
- RAM całkowity, VRAM każdej karty, zaznaczenie kart branych pod uwagę,
- długość kontekstu (dowolna liczba, np. 12288),
- kwantyzacja cache KV (f16 / q8_0 / q4_0),
- rezerwa VRAM (domyślnie 1 GB na kartę), rezerwa RAM dla systemu
  (domyślnie 4 GB na Windowsie, 2 GB na Linuksie, liczona od RAM całkowitego),
- próg ostrzeżenia o małym zapasie (domyślnie 10%).

## 4. Źródła danych o modelu

Analizowany jest **zawsze konkretny model**, nie klasa rozmiaru („7B”).

| Źródło | Co daje | Uwagi |
|---|---|---|
| **Link do Hugging Face** | `config.json` (warstwy, głowice KV, wymiar głowicy, maks. kontekst, eksperci MoE) + lista plików z rozmiarami (prawdziwe rozmiary GGUF dla każdej kwantyzacji) | Modele „gated” (Llama, Gemma) wymagają tokenu HF: pole w ustawieniach albo przejście do trybu ręcznego. Linki do Ollama / LM Studio nie są obsługiwane (to strony, nie API). |
| **Lokalny plik .gguf** | metadane z nagłówka pliku: architektura, rozmiar, kwantyzacja | Działa bez internetu. Najdokładniejsze, bo to dokładnie ten plik, który zostanie uruchomiony. |
| **Wpisanie ręczne** | poziom podstawowy: parametry, kwantyzacja, kontekst. Poziom zaawansowany (rozwijany): warstwy, głowice KV, wymiar głowicy, aktywne parametry (MoE) | Na poziomie podstawowym cache KV jest szacowany ostrożnie, a wynik oznaczony jako „szacunek”. |

Pole „Wskaż model” przyjmuje: ścieżkę do pliku `.gguf` (także jedną z części pliku
dzielonego `…-00001-of-00003.gguf`), link do Hugging Face (także do konkretnego pliku),
skrót `hf.co/<repo>:Q4_K_M` oraz wklejone całe polecenie `ollama run hf.co/…`.
Z repozytorium GGUF program pobiera tylko nagłówek jednego pliku (zapytania HTTP Range,
kilka MB). Token HF trafia wyłącznie do huggingface.co, nigdy do serwerów CDN.

**Tryb Zaawansowany – modele z Ollamy** (decyzja autora, etap 8): przycisk „Z Ollamy…”
widoczny tylko w trybie Zaawansowanym otwiera listę modeli zainstalowanych w Ollamie
(`/api/tags`); wybór wpisuje `ollama:NAZWA` w pole modelu. Dane z `/api/show` (metadane
GGUF i tensory), bez szukania plików. W trybie tekstowym: `python main.py ollama:qwen3:8b`.
Adres Ollamy: ustawienie `ollama_host` (domyślnie `http://127.0.0.1:11434`). Tryb
Automatyczny zostaje prosty – bez tej opcji.

**Modele na dysku** (etap 9a, decyzja autora 28.10.2026): przycisk „Modele na dysku…”
w **obu trybach** („co ja w ogóle mam?” to pytanie początkującego), w trybie tekstowym
`--scan [FOLDER]`.
- **Zgoda:** przy pierwszym użyciu okno z informacją, co jest czytane (nazwy, rozmiary,
  nagłówki plików; nic nie idzie do internetu) – „Szukaj” / „Nie teraz”. Zgoda zapisana
  w ustawieniach (`scan_consent`). W trybie tekstowym zgodą jest sama flaga `--scan`.
- **Poziom 1 – znane miejsca** (sekundy): Ollama (`OLLAMA_MODELS`, `~/.ollama/models`,
  na Linuksie także foldery usługi w `/usr/share/ollama` i `/var/lib/ollama`), LM Studio
  (`~/.lmstudio/models`, starsze `~/.cache/lm-studio/models`), pamięć podręczna Hugging Face
  (`HF_HUB_CACHE`, `HF_HOME/hub`, `~/.cache/huggingface/hub`), llama.cpp (`LLAMA_CACHE`,
  `~/.cache/llama.cpp`, Windows `%LOCALAPPDATA%\llama.cpp`), GPT4All. Tylko istniejące foldery.
- **Poziom 2 – „Przeszukaj folder…”**: dowolny folder (np. z Stable Diffusion WebUI,
  ComfyUI). Bez pełnego skanu całego dysku. Pomijane: foldery ukryte i systemowe, pliki
  mniejsze niż 50 MB.
- **Rozpoznanie po zawartości:** GGUF po nagłówku (projektor obrazu `mmproj` jako dodatek);
  safetensors po nazwach tensorów (model obrazów, LoRA, VAE, model językowy); `.ckpt`,
  `.pt`, `.pth`, `.bin` to pickle – **nigdy nie są otwierane**, typ tylko z nazwy folderu
  (`Stable-diffusion`, `Lora`, `VAE`…). Całe magazyny jako jedna pozycja: Ollama (nazwy
  z `manifests/`, wagi w `blobs/`), repozytorium w pamięci HF (`models--org--nazwa` →
  `org/nazwa`), folder transformers (`config.json` + wagi), folder diffusers
  (`model_index.json`). Plik widziany dwa razy (dowiązania) liczony raz.
- **Poprawki po pierwszym skanie u autora** (29.10.2026): foldery diffusers bez wag
  (Forge `backend/huggingface`, SD.Next `configs` – same pliki JSON) są pomijane; pickle
  bez podpowiedzi z folderu pokazujemy tylko jako `.ckpt` (inaczej np. pamięć shaderów gier
  `*.bin` udawała model); nazwy folderów porównywane bez „-” i „_” (Stability Matrix:
  `StableDiffusion`); ten sam model w kilku miejscach – wszystkie wystąpienia oznaczone „ten sam model jest też
  w innym miejscu” (program nie wie, z której kopii korzysta użytkownik);
  dla Ollamy pokazujemy folder modeli zamiast pliku `sha256-…`.
- **Przerwane pobrania** (dysk autora: 6 plików `Unconfirmed …smdownload` w Stability Matrix,
  14,8 GB): `.smdownload`, `.crdownload`, `.part`, `.partial`, `.download`, `.incomplete`
  (pamięć HF, `blobs/`) i `sha256-…-partial` (Ollama). Na liście jako „przerwane pobieranie”
  z łącznym rozmiarem do odzyskania. Program niczego nie usuwa – pokazuje, gdzie leży plik.
- **Wynik:** lista (nazwa, typ, rozmiar, skąd) i łączny rozmiar. Modele tekstowe można od
  razu sprawdzić: Ollama → `ollama:NAZWA` (gdy Ollama jest wyłączona, program czyta jej
  plik z dysku), folder transformers → szacunek z `config.json` jak dla repozytorium HF
  bez GGUF. Modele obrazów i dodatki są tylko na liście – ocena od etapu 10. Modele dźwięku
  w formacie transformers (Whisper, MusicGen…) na razie jako „inne” – etap 9.

**Modele mowy i muzyki typu LLM** (etap 9, decyzje autora 28.10.2026). Orpheus i OuteTTS
to w środku zwykła Llama, więc sama architektura ich nie odróżni. Rozpoznanie, od
najpewniejszego:
1. **Tokeny dźwięku na końcu słownika GGUF** – czytnik zapamiętuje ostatnie 4096 tokenów
   i wyznacza najczęstszy wzór (cyfry → `#`), jeśli to wyraźnie tokeny specjalne: Orpheus
   `<custom_token_#>`, OuteTTS 1.0 `<|c#_#|>`. Działa dla pliku, linku HF i Ollamy.
2. **`pipeline_tag` z HF** (`text-to-speech` → mowa, `text-to-music` → muzyka,
   `text-to-audio` → dźwięk). Rodzinę dopasowujemy wtedy po nazwie.
3. **Nazwa** – tylko gdy 1 i 2 nic nie dały, z komunikatem „Rozpoznano z nazwy pliku –
   prawdopodobnie model mowy / model muzyczny” (nazwa modelu i pliku, bez folderów).
Rodziny są w `data/audio_models.json` (wzory tokenów i nazw, dekoder, pamięć dekodera,
tokeny na sekundę nagrania, źródła) – nową dopisuje się bez zmiany kodu. Model rozpoznany
bez znanej rodziny dostaje uwagę, że pamięci jego dekodera nie liczymy.

Wszystkie źródła tworzą ten sam obiekt `ModelSpec`. Kalkulator nie wie,
skąd przyszły dane.

## 5. Model pamięci

```
potrzeba = wagi + cache KV + narzut
```

**Wagi**
- z rozmiaru pliku GGUF, jeśli jest znany (HF lub plik lokalny),
- w przeciwnym razie `parametry × bity_na_wagę / 8`, z tabeli `data/quants.json`
  (orientacyjnie: Q4_K_M ≈ 4,8 · Q5_K_M ≈ 5,7 · Q6_K ≈ 6,6 · Q8_0 ≈ 8,5 · F16 = 16).

**Cache KV** (rośnie liniowo z kontekstem)
```
KV = 2 × warstwy × głowice_KV × wymiar_głowicy × kontekst × bajty_na_element
```
bajty_na_element: f16 = 2 · q8_0 ≈ 1,06 · q4_0 ≈ 0,56.
Przykład: Llama 3.1 8B ≈ 128 KB na token → ok. 1 GB przy 8k, ok. 4 GB przy 32k.

**Okno przesuwne** (etap 8): w modelach takich jak Gemma 3 i gpt-oss część warstw pamięta
tylko ostatnie tokeny. Jak w llama.cpp (`llama-kv-cache-iswa.cpp`) warstwy z oknem trzymają
`min(kontekst, okno + 512)` komórek, zaokrąglone w górę do 256:
```
KV = per_warstwę_token × (warstwy_pełne × kontekst + warstwy_z_oknem × komórki_okna)
```
Które warstwy mają okno: tablica albo liczba `{arch}.attention.sliding_window_pattern`
z pliku, a gdy jej brak – domyślny wzór architektury z llama.cpp (gemma2: 2, gemma3: 6,
gpt-oss: 2, cohere2: 4, exaone4: 4; w grupie n warstw pierwsze n−1 mają okno). Z config.json
tylko z jednoznacznych pól (`layer_types`, `sliding_window_pattern`); `use_sliding_window:
false` (Qwen) i samo `sliding_window` bez informacji o warstwach są ignorowane. Nieznana
architektura = pełny kontekst we wszystkich warstwach (wynik zawyżony, czyli bezpieczny).
Przykład: Gemma 3 12B przy 8k – 0,97 GB zamiast 3 GB.

**Hybrydy i modele SSM** (v0.3.1, zgłoszenie z Reddita): Qwen 3.5 / 3.6 / 3.8 i Qwen3-Next
(`qwen35`, `qwen35moe`, `qwen3next`), Granite 4, Nemotron-H, Jamba, LFM2 mają pamięć KV tylko
w warstwach uwagi; pozostałe warstwy (Gated DeltaNet, Mamba, splot) trzymają **stały stan**,
który nie rośnie z kontekstem. Mamba i RWKV nie mają KV wcale. Jak w llama.cpp
(`llama_memory_hybrid`, `llama_hparams::n_embd_r/n_embd_s`):
```
KV   = per_warstwę_token × warstwy_uwagi × kontekst
stan = warstwy_liniowe × (n_embd_r + n_embd_s) × 4 B       (F32, jedna rozmowa)
n_embd_r = (conv_kernel − 1) × (inner_size + 2 × group_count × state_size)
n_embd_s = state_size × inner_size
```
Które warstwy mają uwagę: `{arch}.full_attention_interval` (Qwen, domyślnie 4 – co czwarta),
tablica `attention.head_count_kv` na warstwę (0 = brak uwagi), z config.json `layer_types`
(`linear_attention`, `mamba`, `conv`), `full_attention_interval`, `hybrid_override_pattern`
(„*” = uwaga) albo `attn_layer_period/offset`. Stan jest wliczony w „pamięć rozmowy”.
Przykład: Qwen 3.8 27B przy 90k, KV q8_0 – 2,9 GB + 0,15 GB stanu zamiast 11,9 GB.

**Aktywne parametry MoE** liczymy bez tablicy embedding, gdy model ma osobną warstwę
wyjściową (na token czytany jest jeden wiersz tablicy) – jak OpenAI dla gpt-oss (3,6B).
Inni producenci podają liczbę z tablicą (Mistral: Mixtral 12,9B, u nas ~12,6B).

**Tablica słownika w RAM** (etap 8): llama.cpp – a więc Ollama i LM Studio – trzyma
tablicę wejściową słownika (`token_embd`, słowa × wymiar) w RAM, nie w karcie (z wejścia
czytany jest jeden wiersz na token). Potwierdzone logiem Ollamy u autora:
`CPU_Mapped model buffer size = 525.00 MiB` dla gemma3:4b (262 144 × 2560 w Q6_K).
- Model z osobną warstwą wyjściową (Llama 3.1, Qwen 3, Qwen 2.5): tablica nie trafia do
  karty. Od części dla karty odejmujemy ją z dolnym szacunkiem bitów (≤ 4,5), żeby nie
  obiecać za dużo miejsca.
- Model ze wspólnym wejściem i wyjściem (Gemma, Llama 3.2): karta trzyma tablicę jako
  wyjście, a w RAM jest druga kopia (liczona w Q6_K – tak llama.cpp kwantyzuje wyjście).
- Sam procesor: jedna kopia, bez podziału. Nieznana tablica (wpis ręczny): bez podziału.

Źródła: tensor `token_embd` i obecność `output.weight` w GGUF (plik, HF, Ollama);
`vocab_size × hidden_size` i `tie_word_embeddings` z config.json (brak pola = wspólne,
jak domyślnie w transformers – wtedy nic nie odejmujemy od karty).

```
karta  = wagi − tablica (tylko gdy wyjście osobne) + KV + narzut
RAM    = tablica + to, co nie zmieściło się w karcie
zielone, gdy karta ≤ VRAM_użyt i RAM ≤ RAM_użyt
```

**Dekoder dźwięku** (modele mowy / muzyki, etap 9): czwarta część paska modelu. Zamienia
tokeny na falę dźwięku i zwykle działa jako osobny program (Python + PyTorch), więc jego
koszt to wagi (SNAC 24 kHz: 19,8 mln parametrów) **i** zapas drugiego programu na karcie.
**Pomiar autora** (29.10.2026, laptop RTX 5050 8 GB, `tools/measure_vram.py`, instrukcja:
`docs/pomiary-mowy.md`; dane: `tests/calibration/orpheus_rtx5050_2026-10-29.json`):

| Orpheus 3B Q4_K_M, kontekst 8k | zmierzone | AI Bóbr |
|---|---|---|
| llama-server (część językowa) | 2,99 GB | 2,82 GB + narzut 0,5 GB = 3,32 GB |
| dekoder SNAC (Python, PyTorch cu128) | 0,22 GB | **0,25 GB** (tabela, z zapasem) |
| czas: dekoder w karcie / na procesorze | 11,1 s / 11,2 s | – |

Odsłuch autora (4 nagrania, subiektywnie): nagrania z dekoderem w karcie brzmiały odrobinę
szybciej (5,38 i 5,03 s wobec 5,46 i 5,55 s), bez trzasków i przesterowań w żadnym. Długość
wybiera losowanie tokenów w llama-server, nie dekoder – obiektywnej różnicy nie stwierdzono.
Dekoder na procesorze **nie spowolnił** mowy: tempo wyznacza część językowa (ok. 44 tokeny/s
na tym laptopie, czyli ok. 0,5 s nagrania na sekundę). 497 tokenów dało 5,46 s nagrania
(ok. 91/s – część tokenów skrypt pomija; tabela: 83). Dekoder domyślnie liczony w karcie
(bezpieczniej); podpowiedź „dekoder na procesorze zwolni X GB” pojawia się, gdy to zmienia
werdykt. Dla OuteTTS pamięć dekodera (DAC) to nadal szacunek 0,5 GB – do zmierzenia.
**Kontekst w sekundach nagrania:** przy werdykcie dopisek „≈ 1 min 39 s nagrania”
(dla kontekstu z nagłówka, np. 8k) (tokeny na sekundę z tabeli: Orpheus 83 – 7 tokenów SNAC na ramkę, ~12 ramek/s;
OuteTTS 1.0 150). Obliczenia zostają w tokenach; tekst i próbka głosu też zajmują kontekst.

**Narzut**: bufor obliczeń i kontekst CUDA – **0,5 GB**, zmierzone w etapie 8
(28.10.2026, Ollama 0.40.11, laptop RTX 5050 8 GB, 6 modeli × konteksty 2k/8k/16k,
`tools/calibrate.py`). Prawdziwy narzut (pomiar nvidia-smi − przewidywanie dla karty)
wyszedł 0,17–0,46 GB; 0,5 GB to najmniejsza okrągła wartość, która go zawsze pokrywa:

| | narzut 0,3 GB bez tablicy w RAM | 0,5 GB bez tablicy w RAM | **0,5 GB z tablicą w RAM** |
|---|---|---|---|
| trafione werdykty „całość w karcie” | 17/17 | 16/17 | **17/17** |
| pomiary poniżej nvidia-smi | 2 (Gemma 3 przy 8k, 16k) | 0 | **0** |

Bez tablicy w RAM żadna stała wartość narzutu nie przechodziła wszystkich pomiarów:
Gemma 3 4B (wspólna tablica) potrzebowała ≥ 0,45 GB, a Llama 3.1 8B przy 16k (osobna)
mieściła się w karcie tylko przy narzucie < 0,4 GB – bo jej 0,3 GB tablicy i tak było w RAM.
Pomiary są testem regresyjnym
(`tests/test_calibration_data.py`, dane w `tests/calibration/`): każda zmiana obliczeń musi
dalej trafiać werdykty i nigdy nie przewidywać mniej, niż zmierzył nvidia-smi.
Obserwacje z pomiarów:
- Ollama w `/api/ps` podaje o ~0,1 GB mniej niż nvidia-smi (kontekst CUDA) – narzut to obejmuje.
- Gdy model częściowo trafia do RAM, „size” z `/api/ps` jest zaniżony (qwen2.5-coder 14B:
  7,1 GB przy pliku 8,4 GB) – kalibracja używa tylko pomiarów z całością w VRAM.
- Narzut lekko rośnie z kontekstem (Gemma 3 4B: 0,29 → 0,46 GB od 2k do 16k) – 0,5 GB
  to pokrywa do 16k; przy dłuższych kontekstach do sprawdzenia.
- Zmierzone na modelach 3–14B; dla dużo większych narzut może rosnąć (bufor obliczeń
  zależy od rozmiaru warstw) – do sprawdzenia, gdy będą pomiary.

## 5a. Modele obrazów (etap 10, decyzje autora 29.10.2026)

**Co jest „modelem”** – trzy układy, od najdokładniejszego:
1. **Jeden plik „wszystko w jednym”** (`.safetensors`, np. SD 1.5, SDXL, Illustrious, Pony):
   nagłówek ma nazwy i przesunięcia tensorów, więc rozmiar dzielimy **dokładnie** na część
   główną (`model.diffusion_model.`), enkodery tekstu (`cond_stage_model.`,
   `conditioner.embedders.`) i VAE (`first_stage_model.`) – bez czytania wag.
2. **Folder diffusers / repozytorium HF** (`model_index.json`): rozmiar każdej części
   z jej podfolderu (`unet`/`transformer`, `text_encoder*`, `vae`); przy kilku wariantach
   tej samej części bierzemy fp16 (tak ładują programy).
3. **Części w osobnych plikach** (FLUX, SD 3, GGUF od city96): użytkownik wskazuje plik
   główny, enkodery i VAE dobieramy z tabeli rodziny – **szacunek**.

**Rodzina** po nazwach tensorów (albo klasie z `model_index.json`, albo architekturze GGUF):

| Rodzina | znak rozpoznawczy | natywnie | enkodery i VAE |
|---|---|---|---|
| SD 1.5 | `cond_stage_model.transformer.` | 512×512 | w pliku |
| SD 2.x | `cond_stage_model.model.` | 768×768 | w pliku |
| SDXL (Illustrious, Pony) | `conditioner.embedders.1.` | 1024×1024 | w pliku |
| FLUX | `double_blocks.` | 1024×1024 | osobno: T5-XXL, CLIP-L, VAE |
| SD 3 | `joint_blocks.` | 1024×1024 | zależnie od pliku |

Tabela: `data/image_models.json` (wzory, rozdzielczość, części domyślne, współczynniki).

**Pamięć – szczyt największego etapu, nie suma.** ComfyUI, Forge, SD.Next i diffusers
pracują etapami i między etapami przenoszą części do RAM:
```
kodowanie opisu = enkodery + narzut
rysowanie       = część główna + obliczenia(rozdzielczość × obrazy) + narzut
dekodowanie     = VAE + dekodowanie(rozdzielczość × obrazy) + narzut
szczyt          = max(etapów)            całość naraz = wszystkie części + większe z obliczeń + narzut
```
Obliczenia rosną z liczbą pikseli (część główna liczy na latencie 8× mniejszym, VAE w pełnej
rozdzielczości – dlatego przy dużych obrazach szczytem bywa dekodowanie). Współczynniki
(GB na megapiksel) i narzut: **szacunek do czasu pomiaru** w Forge-neo (`tools/measure_vram.py`).

**Światła:**
- 🟢 **całość naraz w karcie** – najszybciej,
- 🟢 **mieści się etapami** – części są przeładowywane (sekunda–dwie przy zmianie opisu),
  sam obraz liczy się w pełnym tempie,
- 🟡 część główna nie mieści się w karcie, program dociąga ją z RAM – **wolniej** (kilka razy),
- 🔴 nie mieszczą się nawet karta i RAM razem.
Enkoder, który nie mieści się w karcie, a mieści w RAM (np. T5-XXL przy FLUX), **nie zmienia
werdyktu** – dopisek „opis przetworzy procesor, kilka sekund dłużej”.

**Rozdzielczość zamiast kontekstu:** tryb Automatyczny – natywna rodziny; Zaawansowany –
własna rozdzielczość i liczba obrazów naraz. CLI: `--resolution 1024` albo `1024x768`,
`--batch 2`.

**Pasek – trzy wiersze** na wspólnej skali: **System**; **Szczyt** (kolor werdyktu,
z nazwą największego etapu); **Całość naraz** (jaśniejszy, informacyjnie – czy wszystko
zmieściłoby się jednocześnie). Wyjaśnienie wymienia każdy etap z jego szczytem.

## 6. Werdykt: pasek pamięci

Zamiast samych świateł: **dwie warstwy na wspólnej skali w GB**.

- **Warstwa systemu** ma trzy odcinki: VRAM użytkowy (szybki) → RAM użytkowy
  (wolny) → „brak pamięci”.
- **Warstwa modelu** ma trzy części: wagi, cache KV (jaśniejsza), narzut.
- Linie przerywane oznaczają koniec szybkiej pamięci i koniec całej pamięci.

```
VRAM_użyt = Σ (VRAM karty − rezerwa VRAM)   dla zaznaczonych kart
RAM_użyt  = RAM − rezerwa systemu

zielony  potrzeba ≤ VRAM_użyt                       „pójdzie na Twoim komputerze”
żółty    potrzeba ≤ VRAM_użyt + RAM_użyt            „będzie działał wolno”, + procent w VRAM
czerwony w przeciwnym razie                         „zapomnij o tym”, + ile brakuje
```

- Pasek modelu przyjmuje kolor strefy, w której się kończy.
- **Komputer bez osobnej karty** (także z grafiką zintegrowaną) ma pusty odcinek
  VRAM. Najlepszy możliwy wynik to żółty z dopiskiem „tryb CPU”.
- **Ostrzeżenie o małym zapasie**: gdy koniec paska leży bliżej granicy strefy niż
  `max(próg × pojemność strefy, 1 GB)`, program pokazuje „mały zapas, dłuższy
  kontekst może przekroczyć pamięć”. Próg domyślnie 10%, zmienny w trybie
  Zaawansowanym (np. 15%), zapisany w `config/settings.json`.
- Bieżące zajęcie pamięci przez inne programy jest pokazywane jako informacja,
  ale nie wpływa na werdykt (werdykt nie może skakać między skanami).

GUI: `tkinter.Canvas`. Tryb tekstowy: pasek ze znaków, np.
`[*********####------]`.

## 7. Modele MoE

Wykrywane automatycznie: z `config.json` (liczba ekspertów i aktywnych ekspertów)
albo z metadanych GGUF.

- **Pamięć i pasek bez zmian**: wszystkie parametry muszą się zmieścić.
- Przy nazwie modelu znacznik, np. „MoE · 3B aktywnych”.
- **Strefa żółta ma inny tekst**: „wolniej niż na samej karcie, ale używalnie,
  bo przy każdym słowie model używa tylko 3B z 30B parametrów”.

Do sprawdzenia przez autora na kilku modelach MoE; tekst może wymagać korekty.

## 8. Wyjaśnienie

Przycisk **„Wyjaśnienie”** przy każdym wyniku, **zwinięty domyślnie** (nie
przeszkadza doświadczonym). Treść składa się z liczb danego przypadku, nie ze
stałego tekstu. Układ:

```
Model potrzebuje 22,4 GB:
  → 19,9 GB wagi,
  → 2,0 GB pamięć rozmowy (kontekst 8k),
  → 0,5 GB narzut.

Twój komputer:
  → karta: NVIDIA RTX 3060, 11 GB użytkowe,
  → nie zmieści się: 11,4 GB trafi do RAM,
  → procesor <nazwa> i RAM (<typ, prędkość>) są kilka razy wolniejsze od karty,
    dlatego odpowiedzi będą wolne.

Co możesz zrobić:
  → kwantyzacja Q3_K_M i kontekst 4k nadal nie zmieszczą się w karcie,
  → model Qwen 2.5 14B w Q4_K_M zmieści się w całości.
```

- Dla wyniku zielonego trzecia sekcja to **„Masz jeszcze zapas:”** (np. „kontekst
  możesz zwiększyć do 32k”, „Q6_K też zmieści się w karcie”).
- „Co możesz zrobić” podaje **tylko warianty, które zmieniają werdykt**:
  inną kwantyzację, krótszy kontekst, kwantyzację cache KV i inne modele z katalogu.
- Nazwa i parametry RAM są opcjonalne (na Linuksie zwykle wymagają roota).
  Gdy ich brak, nawias znika.
- Zdania są szablonami w `i18n/pl.json` i `en.json`, liczby wstawiane przy
  wyświetleniu (zasady jak w NIUCHACZU: w stanie programu trzymamy klucze i pola,
  nie gotowe zdania).

Ta sekcja zastępuje osobny moduł rekomendacji.

Reguły dopracowane przy implementacji (etap 4, `knowledge/advisor.py`):
- Gdy w karcie mieści się co najmniej 90% modelu, werdykt brzmi łagodniej („pójdzie, ale
  nieco wolniej”) – do RAM trafia wtedy kilka warstw, a nie większość modelu.
- Nie polecamy kwantyzacji poniżej ok. 3,4 bita na wagę (np. Q2_K), bo jakość wyraźnie spada.
- „Masz jeszcze zapas” nie podpowiada wariantu, przy którym program sam ostrzegłby o małym zapasie.
- Gdy żadna pojedyncza zmiana nie wystarcza, program sprawdza je wszystkie naraz
  (najniższa sensowna kwantyzacja + kontekst 4k + cache KV q8_0) i mówi, czy to pomoże.
- Bez karty graficznej zielony jest niemożliwy, więc podpowiedź brzmi: mniejszy model
  albo niższa kwantyzacja odpowie szybciej.
- Ostatni punkt każdej listy kończy się kropką, pozostałe przecinkiem.

## 9. Katalog modeli

- `data/models/catalog.json` dołączony do programu. Wpis: repozytorium HF,
  parametry architektury, rozmiary GGUF dla kwantyzacji, data i źródło.
- **Używany tylko do podpowiedzi „inny model się zmieści”.** Analizowany model
  zawsze pochodzi z HF, pliku .gguf albo z ręcznego wpisu. Bez katalogu
  wyjaśnienie podpowiada tylko zmiany kwantyzacji i kontekstu.
- **Przycisk „Sprawdź aktualizacje”** pobiera nowszy katalog z repozytorium
  projektu na GitHubie. Program nie łączy się z siecią sam z siebie.
- **`tools/build_catalog.py`** (dla autora): lista repozytoriów HF → gotowy JSON.
  Aktualizacja = dopisanie linijki i jedno uruchomienie.

Szczegóły (etap 6, `knowledge/catalog.py`):
- Lista repozytoriów: `data/models/catalog_sources.txt` (jedno repo GGUF na linię).
- Format `catalog.json` (wersja 1): `updated` (data), `models[]` z nazwą, repozytorium,
  architekturą (pola `ArchInfo`) i prawdziwymi rozmiarami wariantów.
- Podpowiadany wariant: Q4_K_M (albo najbliższy: Q4_K_S, IQ4_XS…). Podpowiedź to
  największy model z katalogu, który mieści się w całości w karcie **bez małego zapasu**;
  pomijany jest analizowany model (ta sama nazwa albo repozytorium).
- „Sprawdź aktualizacje listy modeli” (menu Ustawienia; w CLI `--update-catalog`) pobiera
  `catalog.json` z gałęzi `main` repozytorium na GitHubie i zapisuje go w katalogu
  użytkownika tylko wtedy, gdy jest poprawny, niepusty i nowszy. Program używa nowszej
  z dwóch kopii (dołączonej albo pobranej).

## 10. Wykrywanie sprzętu

Wspólny interfejs, osobne implementacje dla systemów, wybór w `factory.py`.

| Dane | Linux | Windows |
|---|---|---|
| CPU, RAM | `psutil`, nazwa CPU z `/proc/cpuinfo` | `psutil`, nazwa CPU z rejestru |
| NVIDIA | `nvidia-ml-py`, zapasowo `nvidia-smi` | jak na Linuksie |
| AMD / Intel, VRAM całkowity | sysfs `mem_info_vram_total` | rejestr `HardwareInformation.qwMemorySize` (64-bit) |
| AMD / Intel, VRAM zajęty | sysfs `mem_info_vram_used` | pomijany (tylko informacja; liczniki wydajności są powolne i trudne do przypisania karcie) |
| Lista kart obecnych | `/sys/class/drm/cardN` | `Win32_VideoController` przez PowerShell (rejestr trzyma też karty dawno wyjęte) |
| Typ i prędkość RAM | brak (wymaga roota) | `Win32_PhysicalMemory`, np. „DDR5-6000” |

**Celowo nie używamy** `Win32_VideoController.AdapterRAM` (WMI): wartość 32-bitowa
obcina VRAM do 4 GB.

**Kilka kart:** program pokazuje wszystkie wykryte karty i sumuje VRAM kart
**jednego producenta** (rezerwa odejmowana na każdej karcie). Grafika
zintegrowana obok osobnej karty się nie liczy. W trybie Zaawansowanym kartę można
odznaczyć.

**Grafika zintegrowana:** Intel poza kartami Arc (identyfikatory 0x56xx, 0xE2xx) oraz AMD
z „VRAM” ≤ 2 GB (wydzielony kawałek RAM w procesorach APU). Karta Intel Arc na Linuksie ma
VRAM „?” (sterownik nie podaje go w prosty sposób) i wymaga wpisu w trybie Zaawansowanym.

**Karta, której runtime może nie obsłużyć:** pasek **liczy się z VRAM**,
a obok stoi ostrzeżenie (`calculator/runtime_support.py`, progi według dokumentacji Ollamy):
- NVIDIA compute capability < 5.0: nieobsługiwana przez CUDA (`warn.nvidia_unsupported`),
- NVIDIA 5.0–7.4 (Maxwell, Pascal, Volta): CUDA 13 ich nie obsługuje, działają kompilacje
  z CUDA 12 albo Vulkan (`warn.nvidia_legacy`),
- sterownik NVIDIA < 550, a dla kart ≤ 6.2 < 570 (`warn.nvidia_driver_old`),
- AMD / Intel: wsparcie zależy od runtime'u, często przez Vulkan (`warn.amd_runtime`,
  `warn.intel_runtime`).

**Gdy wykrycie się nie uda:** nie wolno po cichu przyjąć 0 GB. Pola nieodczytane
mają „?”, werdykt się nie pojawia, program proponuje tryb Zaawansowany z tym, co
wykrył, już wpisanym:

> Nie udało się odczytać pamięci karty graficznej. Uzupełnij brakujące dane,
> resztę już wpisaliśmy. [Ustaw parametry ręcznie]

Zewnętrzne polecenia (`nvidia-smi`, PowerShell) mają limit czasu.

## 11. Interfejs

- **GUI (tkinter)**: wersja główna, `python main.py` (bez `--cli`). Ten sam kod na Linuksie
  i Windowsie; autor testuje GUI na Linuksie. Na Windowsie włączona obsługa skalowania
  ekranu (DPI), żeby okno nie było rozmazane. Gdy okna nie da się otworzyć (brak tkintera
  albo ekranu, np. przez SSH), program sam przechodzi w tryb tekstowy.
  Układ okna: **Model** (pole na link/ścieżkę, „Plik…”, „Sprawdź”, lista kwantyzacji
  z rozmiarami, „Wpisz dane modelu ręcznie…”) → **Twój komputer** (Automatyczny /
  Zaawansowany) → **Wynik** (pasek, werdykt, uwagi, przycisk „Wyjaśnienie”).
  Pobieranie z HF i skan idą w tle, okno nie zamarza. Menu: Plik, Ustawienia (token HF,
  język – zmiana bez restartu, stan zostaje), Pomoc (O programie).
  W trybie Zaawansowanym niezmienione pola liczą się z dokładną wartością ze skanu,
  a nie z zaokrąglonego napisu – samo przełączenie trybu nie może zmienić wyniku.
  „Zapisz jako domyślne” zapisuje parametry do ustawień użytkownika (tryb Automatyczny).
- **Tryb tekstowy**: `python main.py --cli <model>` (do czasu GUI także bez `--cli`).
  Pasek ze znaków; przydatny w testach i przy zgłaszaniu błędów. Opcje: `--quant`,
  `--context 16k`, `--kv-cache q8_0`, `--explain` (wyjaśnienie, w GUI przycisk),
  `--ram GB` i `--vram GB[,GB]` (tryb Zaawansowany: poprawka skanu albo „co jeśli kupię
  kartę 16 GB”), `--no-color`. Bez podanego modelu program o niego pyta.
- **`--fake-hw <plik.json>`**: wczytuje fałszywy profil sprzętu zamiast skanu.
  Pozwala przeklikać na Linuksie scenariusze Windows/AMD, laptop bez karty,
  dwie karty, nieudane wykrycie. Te same profile służą testom automatycznym.
- **`--lang pl|en`**: język; domyślnie język systemu (jak w NIUCHACZU).

**Ustawienia i pamięć podręczna:** wartości domyślne leżą w `config/settings.json`
w repozytorium; zmiany użytkownika (w tym token HF) program zapisuje poza repozytorium
(`~/.config/ai-bobr/`, na Windowsie `%APPDATA%\AI-Bobr\`). Opis modelu z Hugging Face
jest zapamiętywany na 7 dni (`~/.cache/ai-bobr/models/`), więc ponowne sprawdzenie
tego samego modelu z inną kwantyzacją lub kontekstem jest natychmiastowe.

## 12. Dystrybucja

- **Windows:** gotowy `.exe` (PyInstaller), budowany automatycznie przez GitHub
  Actions po wypchnięciu taga i dołączany do wydania. Bez płatnego certyfikatu
  Windows SmartScreen pokaże ostrzeżenie przy pierwszym uruchomieniu; README
  wyjaśnia, jak je przejść.
- **Linux:** uruchamianie ze źródeł (`pip install psutil`, `python main.py`).

Szczegóły (etap 7):
- `tools/build_exe.py` (PyInstaller, jeden plik): `AI-Bobr.exe` – okno, bez konsoli;
  `AI-Bobr-cli.exe` – tryb tekstowy (program okienkowy na Windowsie nie ma konsoli).
  W paczce: teksty i18n, `data/quants.json`, katalog modeli, `config/settings.json`, ikona.
  Ścieżki zasobów: `aibobr/resources.py` (repozytorium albo `sys._MEIPASS`).
- `.github/workflows/tests.yml`: testy przy każdym wypchnięciu i PR – Ubuntu (Python 3.10
  i 3.13, GUI na Xvfb) oraz Windows (3.12) z wykrywaniem sprzętu na prawdziwym rejestrze.
- `.github/workflows/release.yml`: tag `v*` → testy, budowa, sprawdzenie spakowanej wersji
  i oba pliki `.exe` w wydaniu na GitHubie.
- Testy nie tworzą wielogigabajtowych plików (na NTFS plik „rzadki” zajmuje miejsce) –
  rozmiar pliku modelu jest podmieniany (`tests/sizes.py`).
- Ikona: `tools/make_icon.py` → `assets/aibobr.png` (okno) i `assets/aibobr.ico` (.exe).

## 13. Zależności

| Pakiet | Status | Po co |
|---|---|---|
| Python 3.10+, tkinter | wbudowane | GUI |
| `psutil` | **wymagany** | CPU, RAM |
| `nvidia-ml-py` | opcjonalny | VRAM i compute capability NVIDIA; bez niego `nvidia-smi`, potem tryb ręczny |
| pobieranie z HF | wbudowane (`urllib`) | bez `requests` |
| nagłówek .gguf | własny kod (`struct`) | pakiet `gguf` ciągnie `numpy`, a potrzebny jest tylko nagłówek |

## 14. Architektura plików

Wzorowana na NIUCHACZU: `main.py` tylko składa obiekty, `build_app()` działa bez
GUI (używane też w testach), logika w pakiecie, teksty w JSON. Pakiet nazywa się `aibobr`
(bez polskich znaków, bo to nazwa modułu Pythona).

```
main.py                     składanie obiektów, --lang, --cli, --fake-hw
requirements.txt
LICENSE                     MIT
config/
  settings.json             rezerwy, próg zapasu, kontekst domyślny, token HF
data/
  quants.json               bity na wagę dla kwantyzacji
  audio_models.json         rodziny modeli mowy / muzyki: wzory tokenów, dekoder, tokeny/s
  image_models.json         rodziny modeli obrazów: znaczniki, rozdzielczość, części, współczynniki
  models/catalog.json       katalog do podpowiedzi
aibobr/
  domain/      models.py, enums.py, units.py  HardwareProfile, GpuInfo, ModelSpec, Verdict; bajty ↔ GB
  detection/   base.py, factory.py, common.py, linux.py, windows.py, nvidia.py, profiles.py
  sources/     base.py, factory.py, http.py, huggingface.py, gguf_reader.py, manual.py,
               cache.py, ollama.py
  scan/        model.py, locations.py, files.py, ollama_store.py, scanner.py  modele na dysku (etap 9a)
  calculator/  quants.py, memory.py, verdict.py  tabela kwantyzacji; wagi + KV + narzut → strefy, zapas
  knowledge/   catalog.py, advisor.py, audio.py, images.py         warianty „co możesz zrobić”, aktualizacja katalogu
  app/         config.py, controller.py
  i18n/        __init__.py, pl.json, en.json
  cli/         cli.py                         pasek tekstowy
  gui/         main_window.py, theme.py, worker.py, bar_layout.py,
               widgets/ (memory_bar, explanation, hardware_panel, model_picker),
               dialogs/ (manual_model, about, ollama_models, found_models)
tools/
  build_catalog.py
  try_source.py             narzędzie deweloperskie: co program odczytał z linku / pliku
  try_detection.py          narzędzie deweloperskie: co program wykrył; --json zapisuje profil
  calibrate.py              kalibracja: przewidywania kontra pomiar w Ollamie (etap 8)
  measure_vram.py           szczyt VRAM każdego procesu (dowolny program; etap 9 i 10)
tests/
  fake_hardware/*.json      profile sprzętu (także dla --fake-hw)
  test_memory.py, test_verdict.py, test_gguf_reader.py, test_huggingface.py,
  test_detection_*.py, test_advisor.py, test_i18n.py
docs/
  SPEC.md, img/
assets/                     aibobr.png, aibobr.ico
packaging/cli_entry.py      punkt startowy AI-Bobr-cli.exe
.github/workflows/          tests.yml, release.yml
```

Z NIUCHACZA **nie przenosimy** `transport/`, `protocol/`, `reader_thread` ani
`event_bus`: istniały dla strumienia danych z ESP32, a tu skan jest jednorazowy.

## 15. Plan prac

1. **Rdzeń:** `domain/`, `calculator/`, testy. Bez GUI.
2. **Źródła modelu:** czytnik GGUF, klient HF, wpis ręczny.
3. **Wykrywanie:** Linux, Windows, profile `fake`.
4. **Wyjaśnienia i tryb tekstowy:** `advisor.py`, i18n, `--cli`.
5. **GUI:** pasek, wyjaśnienie, tryby Automatyczny i Zaawansowany.
6. **Katalog:** `build_catalog.py`, przycisk „Sprawdź aktualizacje”.
7. **Dystrybucja:** GitHub Actions dla Windows, README pl/en.
8. **Kalibracja:** porównanie werdyktów z rzeczywistym zużyciem pamięci na
   modelach autora (w tym MoE).

Stan: etapy 1–7 zakończone, **v0.1.0 wydane 27.10.2026** (`AI-Bobr.exe`, `AI-Bobr-cli.exe`).
Etap 8 (kalibracja) zakończony – **v0.2.0 wydane 28.10.2026**.
Etapy 9a (modele na dysku) i 9 (modele mowy i muzyki typu LLM) – **v0.3.0, 29.10.2026**.

### Plan po v0.1 (zaakceptowany 27.10.2026)

Program ma sprawdzać nie tylko modele językowe, ale też modele dźwięku, muzyki i obrazów.
Kolejność wynika z tego, jak te modele zużywają pamięć:

8. **Kalibracja modeli językowych** (lista w sekcji 16).
9a. **Modele na dysku** (dopisane 28.10.2026) – szukanie zainstalowanych modeli za zgodą
    użytkownika (SPEC §4); przyda się do pomiarów w etapach 9 i 10.
9. **Dźwięk i muzyka typu LLM** (w toku: rozpoznanie, dekoder, sekundy nagrania,
   `tools/measure_vram.py`; Orpheus zmierzony 29.10.2026, OuteTTS do zmierzenia) – modele generujące tokeny dźwięku jak LLM tokeny tekstu:
   syntezatory mowy na bazie LLM (np. Orpheus na Llamie 3B, OuteTTS, Dia, CSM), MusicGen,
   modele rozmowy głosem. Obecny model pamięci pasuje; dochodzi: rozpoznanie typu z HF
   (`pipeline_tag`, architektura), znacznik „model mowy / muzyki”, dekoder dźwięku
   (np. SNAC) w narzucie, kilka takich modeli w katalogu.
10. **Obrazy** (w toku, szczegóły: §5a) – drugi model pamięci: kilka części (główny model, enkodery tekstu, VAE)
    + pamięć obliczeń zależna od rozdzielczości, bez cache KV. Inne znaczenie świateł:
    ComfyUI / Forge / diffusers ładują części po kolei i przenoszą do RAM, więc model
    większy od VRAM zwykle **zadziała wolniej**; czerwony dopiero, gdy największa część
    nie mieści się w VRAM + RAM. W trybie Zaawansowanym rozdzielczość zamiast kontekstu.
    Źródła: diffusers (`model_index.json` + rozmiary folderów), GGUF (np. FLUX od city96),
    pojedyncze `.safetensors` z ComfyUI (tylko łączny rozmiar → szacunek).
11. **Dyfuzyjne audio** (Stable Audio Open, ACE-Step…) na mechanizmie z etapu 10;
    długość utworu zamiast rozdzielczości.

Każdy etap zaczyna się od omówienia uwag i dopisania szczegółów do tej specyfikacji.

## 16. Otwarte kwestie i rzeczy do sprawdzenia

Etap 8 – kalibracja:
- ✅ **Okno przesuwne** (Gemma 3, gpt-oss) – SPEC §5.
- ✅ **Hybrydy i SSM** (Qwen 3.5+, Granite 4, Nemotron-H, Jamba, LFM2, Mamba, RWKV) – SPEC §5.
- ⏳ **Narzut przy długim kontekście** (≥ 64k): bufor obliczeń rośnie z kontekstem; Swift
  27B przy 90k – przewidywanie 16,9 GB, pomiar 18,4 GB. Kalibracja z logu llama-server.
- ✅ **gpt-oss-20b:** 3,6B aktywnych zamiast 4,2B – SPEC §5.
- ✅ **Repozytoria spoza zakresu:** `model_index.json` albo `pipeline_tag` z HF →
  „to model obrazów / dźwięku” zamiast „brak plików modelu”.
- ✅ **Narzut runtime'u:** 0,5 GB i tablica słownika w RAM (SPEC §5). Pomiar – `tools/calibrate.py`
  ładuje modele z Ollamy z kilkoma kontekstami (pusty prompt, nic nie jest generowane),
  porównuje przewidywanie z `/api/ps` (łącznie i w VRAM) i przyrostem VRAM z `nvidia-smi`,
  zapisuje raport JSON. Źródło `sources/ollama.py` czyta modele z API Ollamy (metadane GGUF
  i tensory z `/api/show`).
- ✅ **Ranking jakości kwantyzacji:** najpierw klasa bitowa z nazwy (Q3_K_L to klasa 3,
  gorsza od IQ4_XS mimo 4,27 > 4,25 bita na wagę), potem bity na wagę.
- ⏳ **Szybkość nagłówka z HF:** 5–12 s na repozytorium (głównie słownik tokenizera).
- ⏳ **Skan na Windowsie:** 3,6 s na serwerze GitHuba, prawie w całości start PowerShella.

- **Narzut runtime'u:** 0,5 GB to wartość startowa; do skalibrowania (etap 8).
- **Szacunek KV bez danych architektury** (ręczny poziom podstawowy): reguła
  ostrożna, do skalibrowania na katalogu.
- **macOS (opcja B):** po wersji 1.
- **Nazwa repozytorium na GitHubie.**
