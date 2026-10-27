# CCD Preprocessing Mini-Workshop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOURUSER/YOURREPO/blob/main/ccd_preprocessing_astropy.ipynb?forcePlay=true)

Hands-on notebook for the **Pelatihan Python — Imahnoong** mini-workshop:
CCD image preprocessing (bias / dark / flat-field correction) with
[Astropy](https://www.astropy.org/) and [ccdproc](https://ccdproc.readthedocs.io/).

Click the badge above to open the notebook in **playground mode** — you can run
everything, but changes are only saved to your own Google Drive, never to this
repository.

## Repository structure

```
ccd-workshop/
├── README.md
├── ccd_preprocessing_astropy.ipynb   ← the workshop notebook (open with badge)
└── data/
    ├── bias/       # TODO: bias frames (e.g. bias_001.fits …)
    ├── dark/       # TODO: dark frames
    ├── flat/       # TODO: flat frames
    └── science/    # TODO: raw science frames
```

The notebook downloads the FITS files over HTTP — participants don't need
to clone the repo or install anything besides a browser.

## The dataset

| Frame type | File pattern | Count | Exposure | Notes |
|---|---|---|---|---|
| Bias | `bias/bias_*.fits` | TODO | 0 s | |
| Dark | `dark/dark_*.fits` | TODO | TODO s | same temperature as science |
| Flat | `flat/flat_*.fits` | TODO | TODO s | TODO: filter used |
| Science | `science/science_*.fits` | TODO | TODO s | TODO: object/target |

TODO: one line about the camera/detector (gain, read noise, temperature),
and any credit for the data if it comes from someone else's telescope.

## Before the workshop (maintainer checklist)

- [ ] Replace `YOURUSER/YOURREPO` in the badge link (README **and** first notebook cell)
- [ ] Replace `YOURUSER/YOURREPO` in the `DATA_BASE_URL` cell (Section 1 of the notebook)
- [ ] Commit the FITS files under `data/` (keep every file **under 100 MB** so a
      normal `git push` works without Git LFS)
- [ ] Adjust the file lists in the notebook (Section 1) to the real filenames
- [ ] Run the notebook top-to-bottom in Colab once, as a fresh participant
      would (via an incognito window / a friend's account)
- [ ] Fill in the TODOs in the author, dataset, and exercise cells

## How to run (for participants)

1. Click the **Open In Colab** badge above.
2. Run the cells in order (Runtime → Run before, section by section is better
   for following along).
3. If you want to keep your edits / plots: **File → Save a copy in Drive**.
