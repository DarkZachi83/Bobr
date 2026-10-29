# Pomiar modeli obrazów (etap 10) – Forge-neo

Cel: zmierzyć, ile karty naprawdę zajmuje generowanie obrazu przy kilku rozdzielczościach,
i zastąpić szacunkowe współczynniki AI Bóbra (pamięć obliczeń na megapiksel, narzut) pomiarem.
Wynik trafi do testu regresyjnego, tak jak kalibracja modeli językowych i Orpheusa.

Modele z dysku autora:
- SDXL: `/mnt/dane/ai-graphics/Data/Packages/forge-neo/models/Stable-diffusion/realismIllustriousBy_v50FP16.safetensors`
- SD 1.5: `/mnt/dane/ai-graphics/Data/Models/StableDiffusion/v1-5-pruned-fp16-emaonly.safetensors`

## 1. Czy AI Bóbr rozpoznaje modele? (nic nie trzeba uruchamiać)

```bash
python main.py --cli "/mnt/dane/ai-graphics/Data/Packages/forge-neo/models/Stable-diffusion/realismIllustriousBy_v50FP16.safetensors" --explain
python main.py --cli "/mnt/dane/ai-graphics/Data/Models/StableDiffusion/v1-5-pruned-fp16-emaonly.safetensors" --explain
```
W linii „Model:” powinno być `model obrazów SDXL · 1024×1024` i `model obrazów SD 1.5 · 512×512`.

## 2. Przygotowanie Forge-neo

1. Uruchom Forge-neo ze Stability Matrix, wybierz model SDXL.
2. Ustawienia wspólne dla wszystkich pomiarów: **sampler Euler a, 20 kroków, CFG 7,
   1 obraz (Batch count 1, Batch size 1), bez Hires fix, bez ADetailer i bez ControlNet**.
   Zanotuj wartość suwaka **GPU Weights** u góry, jeśli jest (nie zmieniaj go między pomiarami).
3. Wygeneruj **jeden obraz na rozgrzewkę** – model wczyta się do pamięci, a pomiar nie
   złapie samego ładowania.

## 3. Pomiar – jeden na każdą rozdzielczość

W terminalu z folderu AI Bóbra (Forge działa w tle):

```bash
python tools/measure_vram.py --label "forge sdxl 1024" --resolution 1024 \
    --model "/mnt/dane/ai-graphics/Data/Packages/forge-neo/models/Stable-diffusion/realismIllustriousBy_v50FP16.safetensors"
```

Ustaw w Forge szerokość i wysokość **1024**, kliknij Generate, poczekaj na gotowy obraz
i naciśnij **Enter** w terminalu. Zanotuj czas z Forge („Time taken”).

Powtórz dla:

| model | rozdzielczości |
|---|---|
| SDXL (`realismIllustrious…`) | 768, 1024, 1280 |
| SD 1.5 (`v1-5-pruned…`) | 512, 768 |

Po zmianie modelu w Forge – znowu jeden obraz na rozgrzewkę. Za każdym razem zmieniaj
`--label`, `--resolution` i `--model`.

Narzędzie pokaże proces Forge z jego szczytem, np.:
```
  6.64 GB  pid 4444  (przed 2.83 GB)  python launch.py …
AI Bóbr dla realismIllustriousBy_v50FP16 · 1024×1024: szczyt 6.43 GB (sample), wszystko naraz 9.68 GB (szacunek)
```

## 4. Co odesłać

Wydruki z terminala (albo pliki `measure-….json`), czasy z Forge i wartość GPU Weights.

## Gdy coś nie działa

- **Na liście nie ma procesu Forge** – pomiar zaczął się po wygenerowaniu obrazu albo Forge
  nie liczy na karcie NVIDIA; uruchom pomiar przed kliknięciem Generate.
- **„Żaden nowy proces…” przy zerowym przyroście** – obraz był gotowy, zanim pomiar ruszył.
