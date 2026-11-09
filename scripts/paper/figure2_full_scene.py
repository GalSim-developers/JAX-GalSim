#!/usr/bin/env python3
"""Figure 2: one full 250 x 250 pixel scene, GalSim versus JAX-GalSim.

The scene is drawn the same way as in the timing runs (`scripts/timing_tests_general.py`).

* GalSim "default": GalSim loops over galaxies and picks the stamp size and FFT size
  for each one.
* JAX-GalSim: galaxies are grouped into size bins; each bin has a fixed stamp size,
  FFT size and batch size (stratified approach, scatter-add into the scene).
* GalSim "matched": GalSim draws each galaxy with the stamp size and FFT size of its
  JAX-GalSim bin (`draw_all_galsim_stratified`).

Panels: GalSim scene; JAX-GalSim scene; residual (JAX-GalSim - GalSim default);
residual (JAX-GalSim - GalSim matched). Residuals are relative to the peak of the
GalSim scene, as in Figure 1. No noise is added.

The galaxies are those of sample `SAMPLE_INDEX` of a timing run with the same seed and
number of samples (the PSF here is fixed instead of varying per scene).

Draft notebook: notebooks/2026-11-09-paper-figure2-full-scene.ipynb
"""

import os
import sys

os.environ["JAX_ENABLE_X64"] = "True"

from functools import partial
from pathlib import Path

import galsim
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax import jit, random
from matplotlib.colors import AsinhNorm, SymLogNorm
from paper_style import IMAGE_CMAP, RES_CMAP, TEXT_WIDTH, clean_axes, set_style

import jax_galsim as jgs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draw_scene_functions import (  # noqa: E402
    draw_all_bins_jgs,
    draw_all_galsim_stratified,
    draw_galsim,
    draw_jgs_vmap_scatter_stamps,
    get_good_sizes_galsim,
    get_one_full_sample,
    prepare_catalog,
    prepare_per_bin_samples,
)

# ---------------------------------------------------------------- parameters
IMAGE_SLEN = 250  # pixels (0.2 arcsec / pixel)
MAX_MAG = 27.0  # catalog cut on i_ab, as in the timing runs

# PSF: round Moffat (same as `--psf-type moffat` in timing_tests_general.py)
PSF_FWHM = 0.8  # arcsec
PSF_BETA = 2.5

# which galaxies: same keys as a timing run with this seed and number of samples
SEED = 45
N_SAMPLES_RUN = 1000
SAMPLE_INDEX = 0

# JAX-GalSim bins (250-pixel run with outliers, scripts/gpu-tests-scatter-vmap.sbatch)
STAMP_SLEN_BINS = (63, 83, 103, 250)
FFT_SIZE_BINS = (128, 128, 128, 256)
MAX_N_GALS_BINS = (65, 20, 5, 3)
MAX_N_ITERS = 5
MAX_N_GALS_GLOBAL = 150
BUFFER = 4  # pixels added to GalSim's good size before picking a bin

# display
SCENE_LINEAR_WIDTH = 0.02  # asinh linear width, in units of the peak
SCENE_TICKS = (0, 0.01, 0.1, 1)
# symlog linear threshold for each residual panel, in units of the peak
RES_LINTHRESH = {"default": 1e-6, "matched": 1e-11}
# ----------------------------------------------------------------------------


def main(
    out: Path = typer.Option(..., help="Output PNG path."),
    catsim_fpath: Path = typer.Option(..., help="CatSim OneDegSq.fits file."),
    cache_dir: Path = typer.Option(..., help="Folder to cache GalSim good sizes."),
):
    set_style()
    cache_dir.mkdir(parents=True, exist_ok=True)

    psf_gs = galsim.Moffat(fwhm=PSF_FWHM, beta=PSF_BETA, flux=1.0)
    psf_jgs = jgs.Moffat(fwhm=PSF_FWHM, beta=PSF_BETA, flux=1.0)

    # catalog, good sizes and cuts, as in timing_tests_general.py (with outliers)
    cat = prepare_catalog(str(catsim_fpath), min_hlr=0.0, max_mag=MAX_MAG)
    good_sizes, good_fft_sizes = get_good_sizes_galsim(
        cat=cat,
        psf=psf_gs,
        out_path=cache_dir,
        suffix=f"moffat-fwhm{PSF_FWHM}-imag{MAX_MAG:g}",
    )
    cat["good_size"] = good_sizes
    cat["good_fft_size"] = good_fft_sizes
    cat = cat[cat["good_size"] + BUFFER <= IMAGE_SLEN]
    assert max(STAMP_SLEN_BINS) >= max(cat["good_size"]) + BUFFER
    for s, f in zip(STAMP_SLEN_BINS, FFT_SIZE_BINS):
        assert np.all(cat[cat["good_size"] <= s]["good_fft_size"] <= f)
    print(f"galaxies in catalog after cuts: {len(cat)}")

    # one scene
    rkeys = random.split(random.key(SEED), N_SAMPLES_RUN)
    _, k2 = random.split(rkeys[SAMPLE_INDEX])
    sample, n_gals, gsizes = get_one_full_sample(
        k2, cat=cat, ilen=IMAGE_SLEN, max_n_gals=MAX_N_GALS_GLOBAL
    )
    bin_idx = np.searchsorted(STAMP_SLEN_BINS, gsizes + BUFFER)
    print(f"number of galaxies: {n_gals}")
    print(f"galaxies per bin: {np.bincount(bin_idx, minlength=len(STAMP_SLEN_BINS))}")

    # GalSim default; also checks each JAX-GalSim bin is at least as large as GalSim's stamp
    gs_default = draw_galsim(
        sample,
        n_gals,
        ilen=IMAGE_SLEN,
        psf=psf_gs,
        check_stamp_sizes=True,
        max_slen=STAMP_SLEN_BINS[-1],
        good_sizes=gsizes,
        buffer=BUFFER,
        size_bins=STAMP_SLEN_BINS,
    )

    # GalSim with the same stamp and FFT sizes as JAX-GalSim
    gs_matched = draw_all_galsim_stratified(
        sample,
        ilen=IMAGE_SLEN,
        psf=psf_gs,
        gsizes=gsizes,
        stamp_slen_bins=STAMP_SLEN_BINS,
        fft_size_bins=FFT_SIZE_BINS,
        buffer=BUFFER,
    )

    # JAX-GalSim, as in the timing runs (on whatever device JAX picks)
    device = jax.devices()[0]
    draw_fncs = [
        jit(
            partial(
                draw_jgs_vmap_scatter_stamps,
                ilen=IMAGE_SLEN,
                slen=s,
                fft_size=f,
                max_n_gals=m,
            )
        )
        for s, f, m in zip(STAMP_SLEN_BINS, FFT_SIZE_BINS, MAX_N_GALS_BINS)
    ]
    all_draw_fnc = jit(
        partial(
            draw_all_bins_jgs,
            ilen=IMAGE_SLEN,
            draw_fncs=draw_fncs,
            device=device,
            n_bins=len(STAMP_SLEN_BINS),
        )
    )
    samples_per_bin, n_iters_per_bin = prepare_per_bin_samples(
        sample,
        gsizes,
        stamp_slen_bins=STAMP_SLEN_BINS,
        max_n_gals_bins=MAX_N_GALS_BINS,
        max_n_iters=MAX_N_ITERS,
        buffer=BUFFER,
    )
    jgs_scene = np.asarray(
        all_draw_fnc(samples_per_bin, jnp.asarray(n_iters_per_bin), psf_jgs)
    )

    # residuals relative to the GalSim peak
    peak = gs_default.max()
    res = {"default": jgs_scene - gs_default, "matched": jgs_scene - gs_matched}
    print(f"scene peak: {peak:.4g} electrons")
    for name, r in res.items():
        print(
            f"{name:8s} max|res|/peak={np.abs(r).max() / peak:.2e}  "
            f"dflux/flux={r.sum() / gs_default.sum():+.2e}"
        )

    # each residual panel gets its own symmetric color scale
    res_rel = {k: v / peak for k, v in res.items()}
    res_norms = {}
    for name, linthresh in RES_LINTHRESH.items():
        vmax = np.abs(res_rel[name]).max()
        res_norms[name] = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
    scene_norm = AsinhNorm(linear_width=SCENE_LINEAR_WIDTH, vmin=0, vmax=1)

    fig, axes = plt.subplots(
        1, 4, figsize=(TEXT_WIDTH, 2.0), constrained_layout=True
    )
    for ax, arr, title in zip(
        axes[:2], [gs_default, jgs_scene], ["GalSim", "JAX-GalSim"]
    ):
        ims = ax.imshow(arr / peak, origin="lower", cmap=IMAGE_CMAP, norm=scene_norm)
        ax.set_title(title)
    cbs = fig.colorbar(ims, ax=axes[:2], shrink=0.85, pad=0.01, ticks=SCENE_TICKS)
    cbs.ax.set_yticklabels([f"{t:g}" for t in SCENE_TICKS])
    cbs.set_label("flux / peak")

    for ax, name, title in zip(
        axes[2:],
        ["default", "matched"],
        ["JAX-GalSim $-$ GalSim\n(default sizes)", "JAX-GalSim $-$ GalSim\n(matched sizes)"],
    ):
        imr = ax.imshow(
            res_rel[name], origin="lower", cmap=RES_CMAP, norm=res_norms[name]
        )
        ax.set_title(title)
        cbr = fig.colorbar(imr, ax=ax, shrink=0.85, pad=0.02)
    cbr.set_label("residual / peak")

    clean_axes(axes)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
