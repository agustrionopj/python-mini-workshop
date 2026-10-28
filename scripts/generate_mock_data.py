"""Generate synthetic CCD frames for the mini-workshop.

Creates two datasets under data/:

  mock_nooverscan/  - 512x512 image only (plain camera)
  mock_overscan/    - 512x512 image + 16 overscan columns (528 wide),
                      headers carry DATASEC / BIASSEC / TRIMSEC

Each dataset contains:
  10 bias, 10 dark (5 x 2s + 5 x 120s), 10 flat (V, 2s), 5 science (V, 120s)

Usage:
    python scripts/generate_mock_data.py

Deterministic: fixed RNG seed, so re-running reproduces identical files.
Run this locally as the maintainer, then commit the output data/ to the repo.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits

# --------------------------------------------------------------------------
# Detector + observing parameters (single source of truth for the simulation)
# --------------------------------------------------------------------------
SEED = 20261031

IMG_SHAPE = (512, 512)      # (rows, cols) of the light-sensitive area
OVERSCAN_COLS = 16          # extra readout columns (overscan dataset only)

GAIN = 1.0                  # e-/ADU (kept at 1 so ADU == e- for teaching)
READ_NOISE = 5.0            # e- rms
BIAS_LEVEL = 500.0          # ADU nominal bias level
BIAS_JITTER = 2.0           # ADU frame-to-frame level jitter
BIAS_ROW_DRIFT = 3.0        # ADU smooth drift across rows
COL_PATTERN_SIG = 1.5       # ADU sigma of per-column fixed pattern
OVERSCAN_OFFSET = 3.0       # ADU: overscan strip level vs image-area level

DARK_CURRENT = 0.3          # e-/s/pix
HOT_FRACTION = 0.001        # fraction of hot pixels
HOT_RATE_RANGE = (10.0, 100.0)  # e-/s in hot pixels

FLAT_ILLUM = 25000.0        # e- total signal at frame centre for flats
VIGNETTING = 0.15           # fractional light loss at far corners
PRNU_SIG = 0.02             # pixel-to-pixel sensitivity sigma (2%)
N_DUST = 3                  # dust donuts on the window

SKY_RATE = 2.0              # e-/s/pix sky background
N_STARS = 80
STAR_PEAK_RANGE = (2000.0, 20000.0)   # e- at PSF centre
PSF_SIGMA_RANGE = (1.2, 2.0)          # px
N_CR = 15                              # cosmic-ray events per science frame
CR_E_RANGE = (500.0, 20000.0)          # e- in the brightest CR pixel

SET_TEMP = -10.0             # C
FILTER_FLAT_SCI = 'V'
OBS_START = datetime(2026, 10, 30, 21, 0, 0)

N_BIAS, N_FLAT, N_SCI = 10, 10, 5
N_DARK_PER_EXP = 5           # dark frames per exposure time (2s and 120s)


# --------------------------------------------------------------------------
# Detector maps (shared by both datasets: same simulated camera)
# --------------------------------------------------------------------------
def make_detector_maps(rng):
    """Fixed detector properties: column pattern, PRNU, vignetting, dust, hot px."""
    ny, nx = IMG_SHAPE

    col_pattern = rng.normal(0.0, COL_PATTERN_SIG, nx)

    prnu = rng.normal(1.0, PRNU_SIG, (ny, nx))

    # radial vignetting
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    r = np.hypot(yy - cy, xx - cx)
    r = r / r.max()
    vign = 1.0 - VIGNETTING * r ** 2

    # dust donuts: smooth transmission dips
    dust = np.ones((ny, nx))
    for _ in range(N_DUST):
        dy = rng.integers(40, ny - 40)
        dx = rng.integers(40, nx - 40)
        rad = rng.uniform(10, 25)
        depth = rng.uniform(0.02, 0.08)
        rr = np.hypot(yy - dy, xx - dx)
        dust *= 1.0 - depth * np.exp(-0.5 * (rr / (rad / 2)) ** 2)

    # hot pixels (extra dark current)
    hotmask = rng.random((ny, nx)) < HOT_FRACTION
    hot_rate = np.where(hotmask,
                        rng.uniform(*HOT_RATE_RANGE, (ny, nx)),
                        0.0)

    return {
        'col_pattern': col_pattern,
        'response': vign * prnu * dust,   # multiplies incoming light
        'hot_rate': hot_rate,
    }


def make_star_catalog(rng):
    """Fixed star field shared by every science frame."""
    ny, nx = IMG_SHAPE
    stars = []
    for _ in range(N_STARS):
        x = rng.uniform(15, nx - 15)
        y = rng.uniform(15, ny - 15)
        sigma = rng.uniform(*PSF_SIGMA_RANGE)
        peak = rng.uniform(*STAR_PEAK_RANGE)
        stars.append((x, y, sigma, peak))
    return stars


# --------------------------------------------------------------------------
# Frame simulation
# --------------------------------------------------------------------------
def add_stars(img, stars, response):
    """Add Gaussian PSF stars (electrons), attenuated by the pixel response."""
    ny, nx = img.shape
    for x, y, sigma, peak in stars:
        half = int(np.ceil(4 * sigma))
        x0, x1 = int(x) - half, int(x) + half + 1
        y0, y1 = int(y) - half, int(y) + half + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, nx), min(y1, ny)
        gx = np.arange(x0, x1)
        gy = np.arange(y0, y1)
        stamp = peak * np.exp(-((gy[:, None] - y) ** 2 + (gx[None, :] - x) ** 2)
                              / (2 * sigma ** 2))
        img[y0:y1, x0:x1] += stamp * response[y0:y1, x0:x1]


def add_cosmic_rays(img, rng):
    """Sharp 1-3 pixel events with a short decay tail (electrons)."""
    ny, nx = img.shape
    for _ in range(N_CR):
        y = rng.integers(2, ny - 2)
        x = rng.integers(2, nx - 2)
        length = rng.integers(1, 4)
        angle = rng.uniform(0, 2 * np.pi)
        peak = rng.uniform(*CR_E_RANGE)
        val = peak
        for step in range(length):
            yy = int(round(y + step * np.sin(angle)))
            xx = int(round(x + step * np.cos(angle)))
            if 0 <= yy < ny and 0 <= xx < nx:
                img[yy, xx] += val
            val *= 0.5


def build_frame(kind, exptime, frame_idx, rng, maps, stars, overscan):
    """Assemble full 2D data + header for one frame, overscan optional."""
    ny, nx = IMG_SHAPE
    level_jitter = rng.normal(0.0, BIAS_JITTER)

    # --- image area --------------------------------------------------
    img = np.zeros((ny, nx), dtype=float)
    drift = np.cumsum(rng.normal(0.0, 0.3, ny))
    drift = drift - drift.mean()
    drift = BIAS_ROW_DRIFT * drift / max(np.abs(drift).max(), 1e-9)
    img += (BIAS_LEVEL + level_jitter) + drift[:, None]
    img += maps['col_pattern'][None, :]

    rate = np.zeros((ny, nx))
    if kind in ('dark', 'science'):
        rate += DARK_CURRENT + maps['hot_rate']   # dark current accumulates in any timed frame
    if kind == 'flat':
        rate += (FLAT_ILLUM / exptime) * maps['response']
    elif kind == 'science':
        rate += SKY_RATE * maps['response']
    if exptime > 0:
        img += rng.poisson(np.clip(rate * exptime, 0.0, None)).astype(float)

    if kind == 'science':
        add_stars(img, stars, maps['response'])
        add_cosmic_rays(img, rng)

    img += rng.normal(0.0, READ_NOISE, img.shape)

    # --- overscan strip (bias-only, same rows/jitter/level) -----------
    if overscan:
        os_col_pattern = rng.normal(0.0, COL_PATTERN_SIG, OVERSCAN_COLS)
        strip = np.tile(
            (BIAS_LEVEL + level_jitter + OVERSCAN_OFFSET + os_col_pattern)[None, :],
            (ny, 1),
        )
        strip += drift[:, None]
        strip += rng.normal(0.0, READ_NOISE, strip.shape)
        full = np.hstack([img, strip])
    else:
        full = img

    header = make_header(kind, exptime, frame_idx, overscan)
    return full, header


def make_header(kind, exptime, frame_idx, overscan):
    h = fits.Header()
    h['IMAGETYP'] = {'bias': 'BIAS', 'dark': 'DARK',
                     'flat': 'FLAT', 'science': 'SCIENCE'}[kind]
    h['EXPTIME'] = exptime
    h['FILTER'] = FILTER_FLAT_SCI if kind in ('flat', 'science') else 'NONE'
    h['OBJECT'] = ('STARFIELD' if kind == 'science'
                   else 'SKYFLAT' if kind == 'flat' else kind.upper())
    h['GAIN'] = GAIN
    h['RDNOISE'] = READ_NOISE
    h['SET-TEMP'] = SET_TEMP
    h['DATE-OBS'] = (OBS_START + timedelta(seconds=frame_idx * 7)).isoformat()
    h['TELESCOP'] = 'MOCK-0.5m'
    h['INSTRUME'] = 'MOCK-CCD'
    if overscan:
        h['DATASEC'] = f'[1:{IMG_SHAPE[1]},1:{IMG_SHAPE[0]}]'
        h['BIASSEC'] = (f'[{IMG_SHAPE[1] + 1}:{IMG_SHAPE[1] + OVERSCAN_COLS},'
                        f'1:{IMG_SHAPE[0]}]')
        h['TRIMSEC'] = h['DATASEC']
    h['HISTORY'] = 'Synthetic data for python-mini-workshop (generate_mock_data.py)'
    return h


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(seed: int = SEED, data_root: str | None = None):
    rng = np.random.default_rng(seed)
    maps = make_detector_maps(rng)
    stars = make_star_catalog(rng)

    if data_root is None:
        data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    written = []

    def emit(kind, exptime, i, set_name, subdir, fname, overscan):
        data, header = build_frame(kind, exptime, i, rng, maps, stars, overscan)
        rel = os.path.join(set_name, subdir, fname).replace(os.sep, '/')
        path = os.path.join(data_root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fits.PrimaryHDU(data=np.clip(np.round(data), 0, 65535).astype(np.uint16),
                        header=header).writeto(path, overwrite=True)
        written.append(rel)

    for set_name, overscan in (('mock_nooverscan', False), ('mock_overscan', True)):
        print(f'--- writing {set_name} (overscan={overscan}) ---')

        # bias: 10 x 0s
        for i in range(1, N_BIAS + 1):
            emit('bias', 0.0, i, set_name, 'bias', f'bias-mock-{i:03d}.fit', overscan)

        # dark: 5 x 2s + 5 x 120s
        for i in range(1, N_DARK_PER_EXP + 1):
            emit('dark', 2.0, i, set_name, 'dark', f'dark-mock-0002s-{i:03d}.fit', overscan)
        for i in range(1, N_DARK_PER_EXP + 1):
            emit('dark', 120.0, i, set_name, 'dark', f'dark-mock-0120s-{i:03d}.fit', overscan)

        # flat: 10 x 2s, V
        for i in range(1, N_FLAT + 1):
            emit('flat', 2.0, i, set_name, 'flat', f'flat-mock-v-2s-{i:03d}.fit', overscan)

        # science: 5 x 120s, V
        for i in range(1, N_SCI + 1):
            emit('science', 120.0, i, set_name, 'science',
                 f'field-mock-v-120s-{i:03d}.fit', overscan)

    # ---- summary ----
    total = sum(os.path.getsize(os.path.join(data_root, r)) for r in written)
    print(f'\nWrote {len(written)} files, total {total / 1e6:.1f} MB under {data_root}')

    print('\nPaste-ready lists for the notebook (main set, no overscan):')
    for list_name, sub in (('BIAS_FILES', 'bias'), ('DARK_FILES', 'dark'),
                           ('FLAT_FILES', 'flat'), ('SCIENCE_FILES', 'science')):
        files = [f for f in written if f.startswith(f'mock_nooverscan/{sub}')]
        print(f'{list_name} = [' + ', '.join(f'"{f}"' for f in files) + ']')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--outdir', default=None, help='data root (default: ../data)')
    args = parser.parse_args()
    main(seed=args.seed, data_root=args.outdir)
