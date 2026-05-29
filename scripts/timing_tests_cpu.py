#!/usr/bin/env python3

import os
from functools import partial

os.environ["JAX_ENABLE_X64"] = "True"
import time
from pathlib import Path

import galsim
import jax
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
from jax import device_put, jit, random
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import jax_galsim as jgs

IMAGE_SLEN = 1000
MAX_N_GALS = 1500
STAMP_SLEN = 61
BUFFER = 3  # good image size (of isolated galaxy) and stamp size used in scene by galsim differ
BACKGROUND = get_default_lsst_background()  # for residual assessment
FFT_SIZE = 128
SUFFIX = f"{IMAGE_SLEN}-{STAMP_SLEN}-scan-cpu-smallsubset-v1"
N_SAMPLES = 250


def main():
    psf = galsim.Gaussian(half_light_radius=0.7, flux=1.0)
    xpsf = jgs.Gaussian(half_light_radius=0.7, flux=1.0)

    catsim_file = "../../../Downloads/catsim/OneDegSq.fits"

    cat = prepare_catalog(catsim_file)
    good_sizes = get_good_sizes_galsim(
        cat=cat, psf=psf, overwrite=False, suffix="gaussian-07"
    )
    cat["good_size"] = good_sizes
    _mask = cat["good_size"] < STAMP_SLEN - BUFFER
    cat = cat[_mask]

    print("INFO: Catalog prepared with {} galaxies after cuts.".format(len(cat)))

    times_galsim = []
    times_jgalsim = []

    # jax draw function prepare and jit
    draw_jax_fnc = jit(
        partial(
            draw_jgs_scan_stamps,
            psf=xpsf,
            ilen=IMAGE_SLEN,
            slen=STAMP_SLEN,
            fft_size=FFT_SIZE,
            max_n_gals=MAX_N_GALS,
        )
    )

    # timing results average over 100 samples
    out_folder = Path("scripts") / "output" / SUFFIX
    out_folder.mkdir(exist_ok=True, parents=False)
    pdf_name = out_folder / "residuals.pdf"
    rkeys = random.split(random.PRNGKey(42), N_SAMPLES)
    with PdfPages(pdf_name) as pdf:
        for ii, rkey in tqdm(
            enumerate(rkeys), total=len(rkeys), desc="Timing galsim vs jax-galsim..."
        ):
            # sample in numpy
            sample, n, _ = get_one_full_sample(
                rkey, cat=cat, ilen=IMAGE_SLEN, max_n_gals=MAX_N_GALS
            )

            if ii == 0:  # trigger jit compilation
                _ = draw_jax_fnc(sample).block_until_ready()  # do we need block here?

            # timing galsim
            t1 = time.time()
            gs_arr = draw_galsim(
                sample,
                n,
                psf=psf,
                ilen=IMAGE_SLEN,
                max_n_gals=MAX_N_GALS,  # just sanity checking
                max_slen=STAMP_SLEN,  # just sanity checking
            )
            t2 = time.time()
            t_galsim = t2 - t1
            times_galsim.append(t_galsim)

            # could time the device put separately
            sample_jax = device_put(sample)

            # timing jax-galsim
            t1 = time.time()
            with jax.transfer_guard("disallow"):
                jgs_arr = draw_jax_fnc(sample_jax).block_until_ready()
            t2 = time.time()
            t_jgalsim = t2 - t1
            times_jgalsim.append(t_jgalsim)

            # save residual images to a multipage pdf for inspection
            add_results_to_pdf(gs_arr, np.array(jgs_arr), t_galsim, t_jgalsim, ii, pdf)

    # print summary timing results
    summary_fname = out_folder / "summary.txt"
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
