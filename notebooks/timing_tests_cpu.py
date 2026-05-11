#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"
import time

import galsim
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from draw_scripts import (
    draw_galsim,
    draw_jgs_scan_stamps,
    draw_jgs_vmap_stamps,
    get_default_lsst_background,
    get_one_full_sample,
)
from jax import device_put, random
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import jax_galsim as jgs

IMAGE_SLEN = 200
MAX_N_GALS = 120
STAMP_SLEN = 80
BACKGROUND = get_default_lsst_background()  # for residual assessment
FFT_SIZE = 128


def _prepare_catalog(
    catsim_file: str, max_hlr: float = 2.0, min_mag: float = 20.0, max_mag: float = 27.0
):
    cat = Table.read(catsim_file, format="fits")

    # avoid objects that are too bright, too dim, or too big
    hlr_b = np.sqrt(cat["a_b"] * cat["b_b"])
    hlr_d = np.sqrt(cat["a_d"] * cat["b_d"])
    _mask1 = hlr_b < max_hlr
    _mask2 = hlr_d < max_hlr
    _mask3 = (hlr_b > 0) | (hlr_d > 0)
    _mask4 = (cat["r_ab"] < max_mag) & (cat["r_ab"] > min_mag)
    mask = _mask1 & _mask2 & _mask3 & _mask4
    fcat = cat[mask]
    return fcat


def _add_results_to_psf(gs_arr, jgs_np_arr, t_galsim, t_jgalsim, ii, pdf):

    vmin = min(gs_arr.min(), jgs_np_arr.min())
    vmax = max(gs_arr.max(), jgs_np_arr.max())

    residual = gs_arr - jgs_np_arr

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Sample {ii}  |  GalSim: {t_galsim:.3f}s  |  JAX-GalSim: {t_jgalsim:.3f}s",
        fontsize=13,
    )

    im0 = axes[0].imshow(gs_arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("GalSim")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        jgs_np_arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[1].set_title("JAX-GalSim")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(residual, origin="lower", cmap="RdBu_r")
    axes[2].set_title("Residual (GalSim - JAX-GalSim)")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    psf = galsim.Gaussian(half_light_radius=0.7, flux=1.0)
    xpsf = jgs.Gaussian(half_light_radius=0.7, flux=1.0)

    catsim_file = "../../../Downloads/catsim/OneDegSq.fits"

    cat = _prepare_catalog(catsim_file)

    times_galsim = []
    times_jgalsim = []

    # timing results average over 100 samples
    rkeys = random.split(random.PRNGKey(42), 100)
    with PdfPages("residuals.pdf") as pdf:
        for ii, rkey in tqdm(enumerate(rkeys)):
            # sample in numpy
            sample, n = get_one_full_sample(
                rkey, cat=cat, ilen=IMAGE_SLEN, max_n_gals=MAX_N_GALS
            )

            if ii == 0:  # trigger jit compilation
                _ = draw_jgs_scan_stamps(
                    sample,
                    psf=xpsf,
                    ilen=IMAGE_SLEN,
                    slen=STAMP_SLEN,
                    fft_size=FFT_SIZE,
                    max_n_gals=MAX_N_GALS,
                )

            t1 = time.time()
            gs_arr = draw_galsim(
                sample, n, psf=psf, ilen=IMAGE_SLEN, max_n_gals=MAX_N_GALS
            )
            t2 = time.time()
            t_galsim = t2 - t1
            times_galsim.append(t_galsim)

            # could time the device put separately
            sample_jax = device_put(sample)

            # scan is the same
            t1 = time.time()
            jgs_arr = draw_jgs_scan_stamps(
                sample_jax,
                psf=xpsf,
                ilen=IMAGE_SLEN,
                slen=STAMP_SLEN,
                fft_size=FFT_SIZE,
                max_n_gals=MAX_N_GALS,
            ).block_until_ready()
            t2 = time.time()
            t_jgalsim = t2 - t1
            times_jgalsim.append(t_jgalsim)

            # save residual images to a multipage pdf for inspection
            _add_results_to_psf(gs_arr, np.array(jgs_arr), t_galsim, t_jgalsim, ii, pdf)

    print(np.median(times_galsim), np.median(times_jgalsim))


if __name__ == "__main__":
    main()
