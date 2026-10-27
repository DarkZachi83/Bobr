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

Wszystkie trzy źródła tworzą ten sam obiekt `ModelSpec`. Kalkulator nie wie,
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

**Narzut**: bufor obliczeń i kontekst runtime'u. Wartość startowa 0,5 GB,
do skalibrowania na prawdziwych pomiarach (zob. sekcja 15).

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

- **GUI (tkinter)**: wersja główna. Ten sam kod na Linuksie i Windowsie; autor
  testuje GUI na Linuksie. Na Windowsie włączona obsługa skalowania ekranu (DPI),
  żeby okno nie było rozmazane.
- **Tryb tekstowy**: `python main.py --cli`. Pasek ze znaków; przydatny w testach
  i przy zgłaszaniu błędów.
- **`--fake-hw <plik.json>`**: wczytuje fałszywy profil sprzętu zamiast skanu.
  Pozwala przeklikać na Linuksie scenariusze Windows/AMD, laptop bez karty,
  dwie karty, nieudane wykrycie. Te same profile służą testom automatycznym.
- **`--lang pl|en`**: język; domyślnie język systemu (jak w NIUCHACZU).

## 12. Dystrybucja

- **Windows:** gotowy `.exe` (PyInstaller), budowany automatycznie przez GitHub
  Actions po wypchnięciu taga i dołączany do wydania. Bez płatnego certyfikatu
  Windows SmartScreen pokaże ostrzeżenie przy pierwszym uruchomieniu; README
  wyjaśnia, jak je przejść.
- **Linux:** uruchamianie ze źródeł (`pip install psutil`, `python main.py`).

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
  sources/     base.py, factory.py, http.py, huggingface.py, gguf_reader.py, manual.py
  calculator/  quants.py, memory.py, verdict.py  tabela kwantyzacji; wagi + KV + narzut → strefy, zapas
  knowledge/   catalog.py, advisor.py         warianty „co możesz zrobić”, aktualizacja katalogu
  app/         config.py, controller.py
  i18n/        __init__.py, pl.json, en.json
  cli/         cli.py                         pasek tekstowy
  gui/         main_window.py, theme.py, widgets/ (memory_bar, explanation, hardware_panel, model_picker)
tools/
  build_catalog.py
  try_source.py             narzędzie deweloperskie: co program odczytał z linku / pliku
  try_detection.py          narzędzie deweloperskie: co program wykrył; --json zapisuje profil
tests/
  fake_hardware/*.json      profile sprzętu (także dla --fake-hw)
  test_memory.py, test_verdict.py, test_gguf_reader.py, test_huggingface.py,
  test_detection_*.py, test_advisor.py, test_i18n.py
docs/
  SPEC.md
.github/workflows/build-windows.yml
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

## 16. Otwarte kwestie i rzeczy do sprawdzenia

- **Narzut runtime'u:** 0,5 GB to wartość startowa; do skalibrowania (etap 8).
- **Szacunek KV bez danych architektury** (ręczny poziom podstawowy): reguła
  ostrożna, do skalibrowania na katalogu.
- **macOS (opcja B):** po wersji 1.
- **Nazwa repozytorium na GitHubie.**
