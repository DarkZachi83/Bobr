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

**Aktywne parametry MoE** liczymy bez tablicy embedding, gdy model ma osobną warstwę
wyjściową (na token czytany jest jeden wiersz tablicy) – jak OpenAI dla gpt-oss (3,6B).
Inni producenci podają liczbę z tablicą (Mistral: Mixtral 12,9B, u nas ~12,6B).

**Narzut**: bufor obliczeń i kontekst CUDA – **0,3 GB**, zmierzone w etapie 8
(28.10.2026, Ollama 0.40.11, laptop RTX 5050 8 GB, 5 modeli × konteksty 2k/8k/16k,
`tools/calibrate.py`):

| | narzut 0,5 GB (wartość startowa) | **0,3 GB** | 0,25 GB |
|---|---|---|---|
| trafione werdykty „całość w karcie” | 13/14 | **14/14** | 14/14 |
| najgorsze niedoszacowanie względem nvidia-smi | 0 | **0** | +0,02 GB |

Przy 0,5 GB jedyny błąd: Llama 3.1 8B przy 16k – program mówił „częściowo w RAM”, a model
zmieścił się w karcie (6,5 GB). Pomiary są testem regresyjnym
(`tests/test_calibration_data.py`, dane w `tests/calibration/`): każda zmiana obliczeń musi
dalej trafiać werdykty i nigdy nie przewidywać mniej, niż zmierzył nvidia-smi.
Obserwacje z pomiarów:
- Ollama w `/api/ps` podaje o ~0,1 GB mniej niż nvidia-smi (kontekst CUDA) – narzut to obejmuje.
- Gdy model częściowo trafia do RAM, „size” z `/api/ps` jest zaniżony (qwen2.5-coder 14B:
  7,1 GB przy pliku 8,4 GB) – kalibracja używa tylko pomiarów z całością w VRAM.
- Zmierzone na modelach 3–14B; dla dużo większych narzut może rosnąć (bufor obliczeń
  zależy od rozmiaru warstw) – do sprawdzenia, gdy będą pomiary.

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
  models/catalog.json       katalog do podpowiedzi
aibobr/
  domain/      models.py, enums.py, units.py  HardwareProfile, GpuInfo, ModelSpec, Verdict; bajty ↔ GB
  detection/   base.py, factory.py, common.py, linux.py, windows.py, nvidia.py, profiles.py
  sources/     base.py, factory.py, http.py, huggingface.py, gguf_reader.py, manual.py,
               cache.py, ollama.py
  calculator/  quants.py, memory.py, verdict.py  tabela kwantyzacji; wagi + KV + narzut → strefy, zapas
  knowledge/   catalog.py, advisor.py         warianty „co możesz zrobić”, aktualizacja katalogu
  app/         config.py, controller.py
  i18n/        __init__.py, pl.json, en.json
  cli/         cli.py                         pasek tekstowy
  gui/         main_window.py, theme.py, worker.py, bar_layout.py,
               widgets/ (memory_bar, explanation, hardware_panel, model_picker),
               dialogs/ (manual_model, about, ollama_models)
tools/
  build_catalog.py
  try_source.py             narzędzie deweloperskie: co program odczytał z linku / pliku
  try_detection.py          narzędzie deweloperskie: co program wykrył; --json zapisuje profil
  calibrate.py              kalibracja: przewidywania kontra pomiar w Ollamie (etap 8)
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

### Plan po v0.1 (zaakceptowany 27.10.2026)

Program ma sprawdzać nie tylko modele językowe, ale też modele dźwięku, muzyki i obrazów.
Kolejność wynika z tego, jak te modele zużywają pamięć:

8. **Kalibracja modeli językowych** (lista w sekcji 16).
9. **Dźwięk i muzyka typu LLM** – modele generujące tokeny dźwięku jak LLM tokeny tekstu:
   syntezatory mowy na bazie LLM (np. Orpheus na Llamie 3B, OuteTTS, Dia, CSM), MusicGen,
   modele rozmowy głosem. Obecny model pamięci pasuje; dochodzi: rozpoznanie typu z HF
   (`pipeline_tag`, architektura), znacznik „model mowy / muzyki”, dekoder dźwięku
   (np. SNAC) w narzucie, kilka takich modeli w katalogu.
10. **Obrazy** – drugi model pamięci: kilka części (główny model, enkodery tekstu, VAE)
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
- ✅ **gpt-oss-20b:** 3,6B aktywnych zamiast 4,2B – SPEC §5.
- ✅ **Repozytoria spoza zakresu:** `model_index.json` albo `pipeline_tag` z HF →
  „to model obrazów / dźwięku” zamiast „brak plików modelu”.
- ✅ **Narzut runtime'u:** 0,3 GB zamiast 0,5 GB (SPEC §5). Pomiar – `tools/calibrate.py`
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
