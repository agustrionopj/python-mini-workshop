"""Generate mock "image basics" data for the scientific_images_as_data notebook.

Creates data/mock_basics/:
  color_photo.png   - procedural everyday-like RGB photo (Pillow demo)
  problem_001.fit   - float32 frame with planted NaNs, saturated pixels,
                      hot-pixel cluster, and single-pixel outliers
  clean_001.fit     - uint16 partner frame (different background, no defects)
  field-r-001.fit / field-g-001.fit / field-b-001.fit
                    - 256x256 mono frames for an RGB composite exercise
  testpattern_001.fit - star-free frame for the hot-pixel reduction demo
  overscan/         - compact set with a 16-column overscan strip:
                      5 bias, 5 dark (120s), 5 flat (2s V), 2 science (120s V)

Detector constants are defined inline (same simulated camera as
generate_mock_data.py), so this script is self-sufficient.

Usage:
    python scripts/generate_mock_data_basics.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits

# detector constants (inlined; same simulated camera as generate_mock_data.py)
GAIN = 1.0                  # e-/ADU (kept at 1 so ADU == e- for teaching)
READ_NOISE = 5.0            # e- rms
BIAS_LEVEL = 500.0          # ADU nominal bias level
SET_TEMP = -10.0            # C
OBS_START = datetime(2026, 10, 30, 21, 0, 0)

# overscan set behavior (16 extra columns on the right side of the frame)
N_OVERSCAN = 16             # overscan columns
OVERSCAN_COLSIGMA = 1.5     # ADU sigma, per-column pattern in the strip
BIAS_JITTER = 2.0           # ADU sigma, frame-to-frame bias level shift
BIAS_ROW_DRIFT = 3.0        # ADU, smooth drift across rows within a frame
OVERSCAN_OFFSET = 3.0       # ADU, strip level vs image-area bias level
DARK_CURRENT = 0.3          # e-/s/pix
FLAT_ILLUM = 25000.0        # e- total signal at frame centre for flats
SKY_RATE = 2.0              # e-/s/pix sky background

SEED = 20261101

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT_DIR = os.path.join(DATA_ROOT, 'mock_basics')

SATURATION = 65535.0          # 16-bit saturation level
EXPTIME = 120.0

# problem frame layout (rows, cols) - fixed so the notebook can talk about them
NAN_BOX = (slice(200, 240), slice(300, 340))   # detector dropout region
HOT_BOX = (slice(100, 115), slice(100, 112))   # hot-pixel cluster
STAR_BOX = ((300, 350), (100, 160))            # bright stars -> saturation


def make_color_photo(path):
    """Procedural 'everyday photo': sunset over mountains with a lake."""
    from PIL import Image

    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]

    # --- sky gradient (top ~55%): deep blue -> warm orange at horizon
    horizon = 0.55 * h
    t = np.clip(yy / horizon, 0, 1)
    sky_top = np.array([30, 45, 110], dtype=float)
    sky_low = np.array([250, 160, 80], dtype=float)
    sky = (1 - t[..., None]) * sky_top + t[..., None] * sky_low

    img = np.broadcast_to(sky, (h, w, 3)).copy()

    # --- sun with glow, near the horizon
    sun_x, sun_y, sun_r = 470, int(0.50 * h), 22
    d = np.hypot(xx - sun_x, yy - sun_y)
    glow = np.exp(-d / 120.0)
    for c, strength in enumerate((1.0, 0.75, 0.5)):
        img[..., c] += strength * 200 * glow
    disk = d < sun_r
    img[disk] = np.array([255, 245, 220])

    # --- mountains: two silhouette layers
    ridge_back = (0.45 * h + 28 * np.sin(xx / 53.0) + 14 * np.sin(xx / 23.0 + 2))
    ridge_front = (0.52 * h + 20 * np.sin(xx / 37.0 + 1) + 10 * np.sin(xx / 15.0))
    back = yy > ridge_back
    front = yy > ridge_front
    img[back] = np.array([70, 55, 75])
    img[front] = np.array([25, 28, 38])

    # --- lake (bottom ~25%): reflect sky gradient + sun glitter
    lake = yy > 0.75 * h
    refl_y = 2 * int(0.75 * h) - yy          # mirror position above waterline
    tt = np.clip(refl_y / horizon, 0, 1)
    refl = (1 - tt[..., None]) * sky_top + tt[..., None] * sky_low
    img[lake] = refl[lake]
    # sun glitter column
    glitter = lake & (np.abs(xx - sun_x) < 14) & (yy > 0.75 * h)
    img[glitter] = img[glitter] * 0.4 + np.array([255, 230, 180]) * 0.6
    # gentle horizontal ripple noise
    ripple = np.random.default_rng(SEED + 1).normal(0, 4, (h, w, 1))
    img[lake] = np.clip(img[lake] + ripple[lake], 0, 255)

    arr = np.clip(np.round(img), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='RGB').save(path)
    return arr


def base_noise(rng, shape, level):
    """Background + read noise, electrons == ADU (GAIN = 1)."""
    return np.maximum(rng.normal(level, READ_NOISE, shape), 0.0)


def add_stars(img, rng, n_stars, peak_range):
    ny, nx = img.shape
    for _ in range(n_stars):
        x = rng.uniform(10, nx - 10)
        y = rng.uniform(10, ny - 10)
        sigma = rng.uniform(1.0, 2.0)
        peak = rng.uniform(*peak_range)
        half = int(np.ceil(4 * sigma))
        x0, x1 = max(int(x) - half, 0), min(int(x) + half + 1, nx)
        y0, y1 = max(int(y) - half, 0), min(int(y) + half + 1, ny)
        gx, gy = np.arange(x0, x1), np.arange(y0, y1)
        stamp = peak * np.exp(-((gy[:, None] - y) ** 2 +
                                (gx[None, :] - x) ** 2) / (2 * sigma ** 2))
        img[y0:y1, x0:x1] += stamp


def header_for(imagetyp, filt, bunit, date_idx=0):
    h = fits.Header()
    h['IMAGETYP'] = imagetyp
    h['EXPTIME'] = EXPTIME
    h['FILTER'] = filt
    h['OBJECT'] = 'MOCK FIELD'
    h['GAIN'] = GAIN
    h['RDNOISE'] = READ_NOISE
    h['SET-TEMP'] = SET_TEMP
    h['BUNIT'] = bunit
    h['DATE-OBS'] = (OBS_START + timedelta(seconds=date_idx * 31)).isoformat()
    h['TELESCOP'] = 'MOCK-0.5m'
    h['INSTRUME'] = 'MOCK-CCD'
    h['HISTORY'] = 'Synthetic data for scientific_images_as_data notebook'
    return h


def write_fits(arr, header, name):
    path = os.path.join(OUT_DIR, name)
    fits.PrimaryHDU(data=arr, header=header).writeto(path, overwrite=True)
    print(f'  wrote {name:20s} shape={arr.shape} dtype={arr.dtype} '
          f'median={np.nanmedian(arr.astype(float)):.1f}')


def make_problem_frame(rng):
    """float32 frame: NaN dropout, saturation plateau, hot cluster, outliers."""
    img = base_noise(rng, (512, 512), 150.0)

    # smooth sky-glow structure so statistics are non-trivial
    yy, xx = np.mgrid[0:512, 0:512]
    img += 20 * np.exp(-((xx - 380) ** 2 + (yy - 120) ** 2) / (2 * 150.0 ** 2))

    # faint stars
    add_stars(img, rng, 40, (300, 3000))

    # very bright stars -> will saturate at 65535
    y0, x0 = STAR_BOX[0][0], STAR_BOX[1][0]
    for (dy, dx, peak) in ((20, 30, 90000), (35, 90, 70000), (10, 60, 80000)):
        img[y0 + dy, x0 + dx] = peak

    # hot-pixel cluster
    img[HOT_BOX] += rng.uniform(1000, 5000, (15, 12))

    # single-pixel outliers (high, but NOT saturated)
    for _ in range(6):
        y = rng.integers(0, 512)
        x = rng.integers(0, 512)
        img[y, x] = rng.uniform(40000, 50000)

    # saturate: clip everything to the 16-bit ceiling
    img = np.clip(img, 0, SATURATION)

    # detector dropout -> NaN
    img[NAN_BOX] = np.nan
    for _ in range(80):                       # scattered dead pixels
        y = rng.integers(0, 512)
        x = rng.integers(0, 512)
        img[y, x] = np.nan

    return img.astype(np.float32), 1


def make_clean_frame(rng):
    """uint16 frame: different background level, no defects."""
    img = base_noise(rng, (512, 512), 80.0)
    add_stars(img, rng, 25, (200, 1500))
    return np.clip(np.round(img), 0, SATURATION).astype(np.uint16), 2


# ---------------------------------------------------------------------------
# Overscan set: 512x512 image + 16 overscan columns (strip = electronics only)
# ---------------------------------------------------------------------------
def overscan_header(imagetyp, filt, exptime, date_idx=0):
    h = header_for(imagetyp, filt, 'adu', date_idx)
    h['EXPTIME'] = exptime
    h['DATASEC'] = '[1:512,1:512]'      # area citra
    h['BIASSEC'] = '[513:528,1:512]'    # strip overscan (16 kolom)
    h['TRIMSEC'] = '[1:512,1:512]'
    h.add_history('strip overscan 16 kolom; level bias bergeser antar frame')
    return h


def make_overscan_maps(rng):
    """Properti detektor yang tetap antar frame (kamera mock yang sama)."""
    ny, nx = 512, 512
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    vign = 1.0 - 0.15 * (r / r.max()) ** 2              # vignetting di sudut
    prnu = rng.normal(1.0, 0.02, (ny, nx))              # sensitivitas piksel
    dust = np.ones((ny, nx))
    for _ in range(3):                                  # 3 donat debu
        dy, dx = rng.integers(40, ny - 40), rng.integers(40, nx - 40)
        rad, depth = rng.uniform(10, 25), rng.uniform(0.02, 0.08)
        rr = np.hypot(yy - dy, xx - dx)
        dust *= 1.0 - depth * np.exp(-0.5 * (rr / (rad / 2)) ** 2)
    hot = rng.random((ny, nx)) < 0.001                  # 0.1% piksel panas
    hot_rate = np.where(hot, rng.uniform(10.0, 100.0, (ny, nx)), 0.0)
    return {'response': vign * prnu * dust,
            'hot_rate': hot_rate,
            'col_pattern': rng.normal(0.0, OVERSCAN_COLSIGMA, nx)}


def make_star_catalog(rng, n_stars=80):
    """Lapangan bintang tetap, dipakai bersama semua frame science."""
    return [(rng.uniform(15, 497), rng.uniform(15, 497),
             rng.uniform(1.2, 2.0), rng.uniform(2000.0, 20000.0))
            for _ in range(n_stars)]


def stamp_stars(img, stars, response):
    """Bintang PSF Gaussian (elektron), teredam respons piksel."""
    for x, y, sigma, peak in stars:
        half = int(np.ceil(4 * sigma))
        x0, x1 = max(int(x) - half, 0), min(int(x) + half + 1, 512)
        y0, y1 = max(int(y) - half, 0), min(int(y) + half + 1, 512)
        gx, gy = np.arange(x0, x1), np.arange(y0, y1)
        stamp = peak * np.exp(-((gy[:, None] - y) ** 2
                                + (gx[None, :] - x) ** 2) / (2 * sigma ** 2))
        img[y0:y1, x0:x1] += stamp * response[y0:y1, x0:x1]


def add_cosmic_rays(img, rng):
    """Peristiwa sinar kosmik: 1-3 piksel dengan ekor pendek (elektron)."""
    for _ in range(15):
        y, x = int(rng.integers(2, 510)), int(rng.integers(2, 510))
        length, angle = int(rng.integers(1, 4)), rng.uniform(0, 2 * np.pi)
        val = rng.uniform(500.0, 20000.0)
        for step in range(length):
            yy, xx = int(round(y + step * np.sin(angle))), \
                int(round(x + step * np.cos(angle)))
            if 0 <= yy < 512 and 0 <= xx < 512:
                img[yy, xx] += val
            val *= 0.5


def make_overscan_frame(rng, maps, stars, kind, exptime):
    """Frame (512, 528): citra 512x512 + strip overscan 16 kolom.

    Elektronik bias: level dasar + jitter antar frame + geseran halus
    sepanjang baris -- strip mengikutinya persis (dibaca bersamaan).
    """
    ny, nx = 512, 512
    jitter = rng.normal(0.0, BIAS_JITTER)
    drift = np.cumsum(rng.normal(0.0, 0.3, ny))
    drift = BIAS_ROW_DRIFT * (drift - drift.mean()) / max(np.abs(drift).max(), 1e-9)

    img = (BIAS_LEVEL + jitter) + drift[:, None] + maps['col_pattern'][None, :]

    rate = np.zeros((ny, nx))
    if kind in ('DARK', 'SCIENCE'):     # arus gelap + piksel panas
        rate += DARK_CURRENT + maps['hot_rate']
    if kind == 'FLAT':                  # iluminasi merata ~25 000 e-
        rate += (FLAT_ILLUM / exptime) * maps['response']
    elif kind == 'SCIENCE':             # langit + lapangan bintang
        rate += SKY_RATE * maps['response']
    if exptime > 0:
        img += rng.poisson(np.clip(rate * exptime, 0.0, None)).astype(float)
    if kind == 'SCIENCE':
        stamp_stars(img, stars, maps['response'])
        add_cosmic_rays(img, rng)

    img += rng.normal(0.0, READ_NOISE, img.shape)

    # strip overscan: elektronik saja -- TANPA cahaya, TANPA arus gelap
    os_pattern = rng.normal(0.0, OVERSCAN_COLSIGMA, N_OVERSCAN)
    strip = np.tile((BIAS_LEVEL + OVERSCAN_OFFSET + jitter + os_pattern)[None, :],
                    (ny, 1))
    strip += drift[:, None]
    strip += rng.normal(0.0, READ_NOISE, strip.shape)

    return np.clip(np.round(np.hstack([img, strip])), 0, SATURATION).astype(np.uint16)


def main(seed: int = SEED, out_dir: str | None = None):
    global OUT_DIR
    if out_dir:
        OUT_DIR = out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    print('--- writing data/mock_basics ---')

    # same star positions for the 3 channel frames -> real colors in composite
    channel_rng = np.random.default_rng(seed + 99)
    n_ch_stars = 25
    ch_pos = [(channel_rng.uniform(10, 246), channel_rng.uniform(10, 246),
               channel_rng.uniform(1.0, 2.2), channel_rng.uniform(300, 3000))
              for _ in range(n_ch_stars)]

    channels = {
        'r': {'level': 30.0, 'color': np.array([1.0, 0.25, 0.10])},
        'g': {'level': 20.0, 'color': np.array([0.30, 1.0, 0.35])},
        'b': {'level': 15.0, 'color': np.array([0.15, 0.30, 1.0])},
    }

    img = make_color_photo(os.path.join(OUT_DIR, 'color_photo.png'))
    print(f"  wrote {'color_photo.png':20s} shape={img.shape} dtype={img.dtype}")

    arr, idx = make_problem_frame(rng)
    write_fits(arr, header_for('SCIENCE', 'V', 'adu', idx), 'problem_001.fit')

    arr, idx = make_clean_frame(rng)
    write_fits(arr, header_for('SCIENCE', 'V', 'adu', idx), 'clean_001.fit')

    for ch, cfg in channels.items():
        img = np.zeros((256, 256), dtype=float)
        img += np.maximum(cfg['level'] + rng.normal(0, READ_NOISE, (256, 256)), 0)
        for (x, y, sigma, peak) in ch_pos:
            peak_ch = float(peak * cfg['color'][ 'rgb'.index(ch)])
            half = int(np.ceil(4 * sigma))
            x0, x1 = max(int(x) - half, 0), min(int(x) + half + 1, 256)
            y0, y1 = max(int(y) - half, 0), min(int(y) + half + 1, 256)
            gx, gy = np.arange(x0, x1), np.arange(y0, y1)
            img[y0:y1, x0:x1] += peak_ch * np.exp(
                -((gy[:, None] - y) ** 2 + (gx[None, :] - x) ** 2) / (2 * sigma ** 2))
        img = np.clip(np.round(img), 0, SATURATION).astype(np.uint16)
        write_fits(img, header_for('SCIENCE', ch.upper(), 'adu'), f'field-{ch}-001.fit')

    # --- citra uji: grayscale polos TANPA bintang, untuk demo hot-pixel ----
    tp_rng = np.random.default_rng(seed + 555)
    tp_yy, tp_xx = np.mgrid[0:512, 0:512]
    tp = np.full((512, 512), 200.0)
    tp += 100 * np.exp(-((tp_xx - 256) ** 2 + (tp_yy - 256) ** 2)
                       / (2 * 180.0 ** 2))                     # pusat lebih terang
    tp[120:180, 120:180] = 2500.0                              # kotak terang
    tp[(tp_xx - 360) ** 2 + (tp_yy - 300) ** 2 < 45 ** 2] = 60.0  # lingkaran gelap
    for i in range(512):
        tp[i, i:i + 8] = 1800.0                                # pita diagonal
    tp += tp_rng.normal(0, READ_NOISE, tp.shape)
    tp = np.clip(tp, 0, SATURATION)
    for _ in range(60):                                        # hot-pixel terisolasi
        y = int(tp_rng.integers(0, 512))
        x = int(tp_rng.integers(0, 512))
        tp[y, x] = tp_rng.uniform(1500, 4000)
    for _ in range(8):                                         # pencilan ekstrem
        y = int(tp_rng.integers(0, 512))
        x = int(tp_rng.integers(0, 512))
        tp[y, x] = tp_rng.uniform(40000, 50000)
    write_fits(tp.astype(np.uint16), header_for('SCIENCE', 'NONE', 'adu'),
               'testpattern_001.fit')

    # --- set overscan: citra 512x512 + strip 16 kolom ------------------------
    for sub in ('bias', 'dark', 'flat', 'science'):
        os.makedirs(os.path.join(OUT_DIR, 'overscan', sub), exist_ok=True)

    ov_rng = np.random.default_rng(seed + 777)   # aliran terpisah dari rng utama
    ov_maps = make_overscan_maps(ov_rng)
    ov_stars = make_star_catalog(ov_rng)
    ov_specs = [('BIAS', 0.0, 'NONE', 5, 'bias/bias-mock-ov-{:03d}.fit'),
                ('DARK', 120.0, 'NONE', 5, 'dark/dark-mock-ov-0120s-{:03d}.fit'),
                ('FLAT', 2.0, 'V', 5, 'flat/flat-mock-ov-v-2s-{:03d}.fit'),
                ('SCIENCE', 120.0, 'V', 2, 'science/field-mock-ov-v-120s-{:03d}.fit')]

    print('--- overscan (citra 512x512 + 16 kolom overscan) ---')
    date_idx = 0
    for kind, exptime, filt, n, pattern in ov_specs:
        for i in range(1, n + 1):
            arr = make_overscan_frame(ov_rng, ov_maps, ov_stars, kind, exptime)
            write_fits(arr, overscan_header(kind, filt, exptime, date_idx),
                       'overscan/' + pattern.format(i))
            date_idx += 1

    total = sum(os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(OUT_DIR) for f in files)
    print(f'\nDone. {total / 1e6:.1f} MB in {OUT_DIR}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--outdir', default=None)
    args = parser.parse_args()
    main(seed=args.seed, out_dir=args.outdir)
