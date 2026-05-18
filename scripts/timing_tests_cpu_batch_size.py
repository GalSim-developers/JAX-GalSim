#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"
import time
from functools import partial
from pathlib import Path

import galsim
import jax
import jax.numpy as jnp
import numpy as np
from draw_scene_functions import (
    add_results_to_pdf,
    draw_galsim,
    draw_jgs_scan_stamps,
    get_default_lsst_background,
    get_good_sizes_galsim,
    get_one_full_sample,
    prepare_catalog,
)
from jax import device_put, random
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import jax_galsim as jgs

IMAGE_SLEN = 250
MAX_N_GALS = 130
STAMP_SLEN = 61
# good image size (isolated galaxy) and stamp size used in scene by galsim differ
BUFFER = 4
BACKGROUND = get_default_lsst_background()
FFT_SIZE = 128
BATCH_BINS = (61, 81, 101)
SUFFIX = f"{IMAGE_SLEN}-{BATCH_BINS[-1]}-scan-cpu-batch-sizes"


def main():
    psf = galsim.Gaussian(half_light_radius=0.7, flux=1.0)
    xpsf = jgs.Gaussian(half_light_radius=0.7, flux=1.0)

    catsim_file = "../../../Downloads/catsim/OneDegSq.fits"

    cat = prepare_catalog(catsim_file)
    n1 = len(cat)
    good_sizes = get_good_sizes_galsim(cat=cat, psf=psf, overwrite=False)
    cat["good_size"] = good_sizes
    mask_good_size = cat["good_size"] < BATCH_BINS[-1] - BUFFER
    cat = cat[mask_good_size]
    print(
        "INFO: Catalog prepared with {} galaxies after cuts (before cut {}).".format(
            len(cat), n1
        )
    )

    times_galsim = []
    times_jgalsim = []

    # prepare three draw functions
    _draw1 = partial(
        draw_jgs_scan_stamps,
        psf=xpsf,
        ilen=IMAGE_SLEN,
        fft_size=FFT_SIZE,
        max_n_gals=MAX_N_GALS,
        slen=BATCH_BINS[0],
    )
    _draw2 = partial(
        draw_jgs_scan_stamps,
        psf=xpsf,
        ilen=IMAGE_SLEN,
        fft_size=FFT_SIZE,
        max_n_gals=MAX_N_GALS,
        slen=BATCH_BINS[1],
    )
    _draw3 = partial(
        draw_jgs_scan_stamps,
        psf=xpsf,
        ilen=IMAGE_SLEN,
        fft_size=FFT_SIZE,
        max_n_gals=MAX_N_GALS,
        slen=BATCH_BINS[2],
    )

    # timing results average over 100 samples
    pdf_name = Path("out") / f"residuals_{SUFFIX}.pdf"
    rkeys = random.split(random.PRNGKey(42), 10)
    with PdfPages(pdf_name) as pdf:
        for ii, rkey in tqdm(
            enumerate(rkeys), total=len(rkeys), desc="Timing galsim vs jax-galsim..."
        ):
            # sample in numpy
            sample, n, gsizes = get_one_full_sample(
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
                sample,
                n,
                psf=psf,
                ilen=IMAGE_SLEN,
                max_n_gals=MAX_N_GALS,  # sanity
                max_slen=BATCH_BINS[-1],  # sanity
                # good_sizes=gsizes,
            )
            t2 = time.time()
            t_galsim = t2 - t1
            times_galsim.append(t_galsim)

            # NOTE: could time the device put separately
            sample_jax = device_put(sample)
            gsizes_jax = device_put(gsizes)

            # jax draw
            t1 = time.time()
            with jax.transfer_guard("disallow"):
                # split into batches using good sizes estimated from galsim
                jgs_arr = jnp.zeros((IMAGE_SLEN, IMAGE_SLEN))
                _drawn = jnp.zeros_like(gsizes).astype(bool)
                for b in BATCH_BINS:
                    _mask = (~_drawn) & (gsizes <= b)
                    _sample_jax = {}
                    for p in sample_jax:
                        _sample_jax[p] = sample_jax[p] * _mask.astype(float)
                    _jgs_arr = draw_jgs_scan_stamps(
                        _sample_jax,
                        psf=xpsf,
                        ilen=IMAGE_SLEN,
                        slen=b,
                        fft_size=FFT_SIZE,
                        max_n_gals=MAX_N_GALS,
                    ).block_until_ready()
                    assert _jgs_arr.shape[0] == IMAGE_SLEN
                    jgs_arr += _jgs_arr
                    _drawn = _drawn.at[_mask].set(True)
            t2 = time.time()
            t_jgalsim = t2 - t1
            assert jnp.all(_drawn)
            times_jgalsim.append(t_jgalsim)

            # save residual images to a multipage pdf for inspection
            add_results_to_pdf(gs_arr, np.array(jgs_arr), t_galsim, t_jgalsim, ii, pdf)

    # print summary timing results
    summary_fname = Path("out") / f"summary_{SUFFIX}.txt"
    with open(summary_fname, "w") as fp:
        print(
            f"Median time (per image) for GalSim: {np.median(times_galsim):.3f} seconds",
            file=fp,
        )
        print(
            f"Average time (per image) for GalSim: {np.mean(times_galsim):.3f} seconds",
            file=fp,
        )

        print(
            f"Median time (per image) for JAX-GalSim: {np.median(times_jgalsim):.3f} seconds",
            file=fp,
        )
        print(
            f"Average time (per image) for JAX-GalSim: {np.mean(times_jgalsim):.3f} seconds",
            file=fp,
        )


if __name__ == "__main__":
    main()
