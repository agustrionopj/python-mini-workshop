# CCD Preprocessing Mini-Workshop

[![Open In Colab — Citra ilmiah sebagai data](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agustrionopj/python-mini-workshop/blob/main/scientific_images_as_data.ipynb?forcePlay=true)

Notebook untuk workshop Python, ditujukan untuk peserta
**tanpa latar belakang astronomi**,  5 modul, $±$ 3 jam: citra sebagai data,
statistik citra, kalibrasi, reduksi derau & warna, pengukuran + ekspor.

Klik pada badge di atas untuk membuka notebook dalam mode **playground**: semua cell bisa dijalankan secara bebas, namun segala perubahan hanya disimpan di versi salinan pada Drive masing-masing.

## Struktur repository

```
python-mini-workshop/
├── README.md
├── scientific_images_as_data.ipynb   ← workshop notebook, non-astronomer (buka dengan badge)
├── scripts/
│   ├── generate_mock_data.py         ← menghasilkan mock CCD FITS data (maintainer only)
│   └── generate_mock_data_basics.py  ← menghasilkan data/mock_basics/ (maintainer only)
└── data/
    ├── mock_nooverscan/              # 512×512 image
    │   ├── bias/  dark/  flat/  science/
    ├── mock_overscan/                # 512×512 + 16 overscan columns
    │   ├── bias/  dark/  flat/  science/
    └── mock_basics/                  # image-basics data 
        ├── color_photo.png           # prosedural RGB "foto sehari-hari" (Pillow demo)
        ├── problem_001.fit           # NaNs + saturated px + hot cluster + outliers
        ├── clean_001.fit             # frame pelengkap dengan background berbeda
        ├── field-r/g/b-001.fit       # latihan 3 mono frame → RGB composite 
        └── testpattern_001.fit       # frame star-free untuk reduksi hot-pixel
```

Notebook akan mengunduh semua file FITS secara otomatis. Para peserta tidak perlu melakukan instalasi ataupun clone repository selain menggunakan browser.

## Dataset

Data sintetik ("mock") yang dihasilkan oleh `scripts/generate_mock_data.py`
(seed 20261031). Simulasi kamera: gain 1 e⁻/ADU,
read noise 5 e⁻, −10 °C. Kamera yang sama disimulasikan dua kali: sebagai frame $512 \times  512$ dan sebagai frame dengan daerah overscan 16-column
(tambahan `DATASEC` / `BIASSEC` / `TRIMSEC` headers pada `mock_overscan`
dataset).

Main set (`mock_nooverscan/`, identical copies in `mock_overscan/`):

| Frame type | File pattern | Count | Exposure | Notes |
|---|---|---|---|---|
| Bias | `bias/bias-mock-*.fit` | 10 | 0 s | bias level ~500 ADU + row drift |
| Dark | `dark/dark-mock-0002s-*.fit` | 5 | 2 s | |
| Dark | `dark/dark-mock-0120s-*.fit` | 5 | 120 s | dark current + hot pixels |
| Flat | `flat/flat-mock-v-2s-*.fit` | 10 | 2 s | filter V, ~25 000 e⁻ signal, vignetting + dust |
| Science | `science/field-mock-v-120s-*.fit` | 5 | 120 s | star field + cosmic rays |

`mock_basics/` (notebook 2 — image basics, generator:
`scripts/generate_mock_data_basics.py`, seed 20261101):

| File | Purpose |
|---|---|
| `color_photo.png` | procedural RGB photo (sunset/lake) — Pillow & RGB-channel demos |
| `problem_001.fit` | float32 frame with 1680 NaN, 3 saturated, hot cluster, 6 outliers |
| `clean_001.fit` | uint16 partner frame (background ~80 ADU) — statistical comparison |
| `field-r/g/b-001.fit` | 3 mono 256×256 frames — RGB composite (`make_lupton_rgb`) |
| `testpattern_001.fit` | star-free frame with 60 isolated hot pixels — median-filter demo |

## How to run (for participants)

1. Click the **Open In Colab** badge above.
2. Run the cells in order (Runtime → Run before, section by section is better
   for following along).
3. If you want to keep your edits / plots: **File → Save a copy in Drive**.
