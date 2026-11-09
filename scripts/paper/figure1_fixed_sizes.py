#!/usr/bin/env python3
"""Figure 1: effect of a fixed stamp size and FFT size on one galaxy.

One isolated, elliptical exponential disk convolved with a round Moffat PSF.
Reference: GalSim with default settings (GalSim picks the stamp size and FFT size).
JAX-GalSim panels: stamp size or FFT size smaller / larger than GalSim's choice.
Every stamp is drawn at the same position and pasted into the center of a larger
canvas, so residuals are compared on a common grid. Residuals are relative to the
peak of the GalSim image, with a shared color scale.

Constraint: in JAX-GalSim, with `minimum_fft_size == maximum_fft_size`, the stamp
cannot be larger than the FFT image. So "stamp larger" raises the FFT size to
`good_fft_size(stamp)` if needed (GalSim's rule), and "FFT smaller" reduces the
stamp to the FFT size if needed.

Draft notebook: notebooks/2026-11-09-paper-figure1-fixed-sizes.ipynb
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"

from pathlib import Path

import galsim
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import AsinhNorm, SymLogNorm
from paper_style import IMAGE_CMAP, RES_CMAP, TEXT_WIDTH, clean_axes, set_style

import jax_galsim as jgs

# ---------------------------------------------------------------- parameters
PIXEL_SCALE = 0.2  # arcsec / pixel

# galaxy: exponential disk only
HLR = 0.8  # arcsec
Q = 0.6  # axis ratio
BETA = 30.0  # degrees
FLUX = 1e5

# PSF: round Moffat (same as `--psf-type moffat` in timing_tests_general.py)
PSF_FWHM = 0.8  # arcsec
PSF_BETA = 2.5

# canvas the stamps are pasted into, and the central region shown in the figure
CANVAS = 192
CROP = 100  # must be larger than the largest stamp (STAMP_LARGE)

# smaller / larger than GalSim's choice
STAMP_SMALL_FACTOR = 0.5  # times GalSim's stamp size
STAMP_LARGE = 86  # pixels; good_fft_size(86) = 96 <= GalSim's FFT size, so the FFT size is unchanged
FFT_SMALL = 32  # pixels; below the galaxy's own good size, so flux folds back in
FFT_LARGE_FACTOR = 2.0  # times GalSim's FFT size

# display
IMAGE_LINEAR_WIDTH = 1e-2  # asinh linear width, in units of the peak
IMAGE_TICKS = (0, 0.01, 0.1, 1)
RES_LINTHRESH = 1e-5  # symlog linear threshold, in units of the peak
# ----------------------------------------------------------------------------

canvas_center = galsim.PositionD(CANVAS / 2 + 0.5, CANVAS / 2 + 0.5)


def make_obj(lib):
    """Same profile in GalSim (lib=galsim) or JAX-GalSim (lib=jgs)."""
    gal = lib.Exponential(half_light_radius=HLR, flux=FLUX).shear(
        q=Q, beta=BETA * lib.degrees
    )
    psf = lib.Moffat(beta=PSF_BETA, fwhm=PSF_FWHM)
    return lib.Convolve([gal, psf])


def galsim_sizes(obj):
    """Stamp size and FFT size GalSim uses for drawImage with default settings."""
    stamp = obj.drawImage(scale=PIXEL_SCALE)
    nstamp = stamp.array.shape[0]
    nfft = galsim.Image.good_fft_size(max(obj.getGoodImageSize(PIXEL_SCALE), nstamp))
    nfft = max(nfft, obj.gsparams.minimum_fft_size)
    return nstamp, nfft


def paste(arr, xmin, ymin):
    out = np.zeros((CANVAS, CANVAS))
    out[ymin - 1 : ymin - 1 + arr.shape[0], xmin - 1 : xmin - 1 + arr.shape[1]] += arr
    return out


def draw_galsim():
    im = make_obj(galsim).drawImage(scale=PIXEL_SCALE, center=canvas_center)
    return paste(im.array, im.bounds.xmin, im.bounds.ymin)


def draw_jax(stamp_size, fft_size):
    gsparams = jgs.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)
    obj = make_obj(jgs).withGSParams(gsparams)
    im = obj.drawImage(
        nx=stamp_size,
        ny=stamp_size,
        scale=PIXEL_SCALE,
        center=jgs.PositionD(canvas_center.x, canvas_center.y),
    )
    return paste(np.asarray(im.array), int(im.bounds.xmin), int(im.bounds.ymin))


def crop(x):
    lo = (CANVAS - CROP) // 2
    return x[lo : lo + CROP, lo : lo + CROP]


def main(out: Path = typer.Option(..., help="Output PNG path.")):
    set_style()
    assert CROP > STAMP_LARGE

    obj_gs = make_obj(galsim)
    gs_stamp, gs_fft = galsim_sizes(obj_gs)
    print(f"GalSim stamp size: {gs_stamp}")
    print(
        f"GalSim FFT size: {gs_fft} (minimum_fft_size = {obj_gs.gsparams.minimum_fft_size})"
    )
    print(f"getGoodImageSize: {obj_gs.getGoodImageSize(PIXEL_SCALE)}")
    assert galsim.Image.good_fft_size(STAMP_LARGE) <= gs_fft
    assert FFT_SMALL < obj_gs.getGoodImageSize(PIXEL_SCALE)

    ref = draw_galsim()
    peak = ref.max()

    good = galsim.Image.good_fft_size
    s_small = int(gs_stamp * STAMP_SMALL_FACTOR)
    f_large = int(gs_fft * FFT_LARGE_FACTOR)
    cases = {
        "same as GalSim": (gs_stamp, gs_fft),
        "stamp smaller": (s_small, gs_fft),
        "stamp larger": (STAMP_LARGE, max(gs_fft, good(STAMP_LARGE))),
        "FFT smaller": (min(gs_stamp, FFT_SMALL), FFT_SMALL),
        "FFT larger": (gs_stamp, f_large),
    }

    residuals = {}
    for name, (nstamp, nfft) in cases.items():
        r = draw_jax(nstamp, nfft) - ref
        residuals[name] = r
        print(
            f"{name:15s} stamp={nstamp:4d} fft={nfft:4d}  "
            f"max|res|/peak={np.abs(r).max() / peak:.2e}  dflux/flux={r.sum() / ref.sum():+.2e}"
        )

    panels = ["stamp smaller", "stamp larger", "FFT smaller", "FFT larger"]
    res_rel = {k: crop(residuals[k]) / peak for k in panels}
    vmax = max(np.abs(r).max() for r in res_rel.values())
    res_norm = SymLogNorm(linthresh=RES_LINTHRESH, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(
        1, 5, figsize=(TEXT_WIDTH, 1.75), constrained_layout=True
    )
    im0 = axes[0].imshow(
        crop(ref) / peak,
        origin="lower",
        cmap=IMAGE_CMAP,
        norm=AsinhNorm(linear_width=IMAGE_LINEAR_WIDTH, vmin=0, vmax=1),
    )
    axes[0].set_title(f"GalSim\n(stamp {gs_stamp}, FFT {gs_fft})")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02, ticks=IMAGE_TICKS)
    cb0.ax.set_yticklabels([f"{t:g}" for t in IMAGE_TICKS])
    cb0.set_label("flux / peak")

    for ax, name in zip(axes[1:], panels):
        nstamp, nfft = cases[name]
        imr = ax.imshow(res_rel[name], origin="lower", cmap=RES_CMAP, norm=res_norm)
        ax.set_title(f"{name}\n(stamp {nstamp}, FFT {nfft})")
    cbr = fig.colorbar(imr, ax=axes[1:], shrink=0.85, pad=0.01)
    cbr.set_label("(JAX-GalSim $-$ GalSim) / peak")

    clean_axes(axes)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
