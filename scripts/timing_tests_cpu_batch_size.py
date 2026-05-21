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
    DUMMY_PARAMS,
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
# good image size (isolated galaxy) and stamp size used in scene by galsim differ
BUFFER = 3  # sometimes good size below and final stamp size differ by a small amount, IDK why
BACKGROUND = get_default_lsst_background()
FFT_SIZE = 128
SLEN_BINS = (61, 81, 101)
MAX_N_GALS = 150
MAX_N_GAL_BINS = (130, 21, 7)
SUFFIX = f"{IMAGE_SLEN}-{SLEN_BINS[-1]}-scan-cpu-batch-sizes"
DEVICE = jax.devices()[0]
assert sorted(SLEN_BINS) == list(SLEN_BINS), "Needs to be sorted"


def main():
    psf = galsim.Gaussian(half_light_radius=0.7, flux=1.0)
    xpsf = jgs.Gaussian(half_light_radius=0.7, flux=1.0)

    catsim_file = "../../../Downloads/catsim/OneDegSq.fits"

    cat = prepare_catalog(catsim_file)
    n1 = len(cat)
    good_sizes = get_good_sizes_galsim(
        cat=cat, psf=psf, overwrite=False, suffix="gaussian-07"
    )
    cat["good_size"] = good_sizes
    mask_good_size = cat["good_size"] < SLEN_BINS[-1] - BUFFER
    cat = cat[mask_good_size]
    print(
        "INFO: Catalog prepared with {} galaxies after cuts (before cut {}).".format(
            len(cat), n1
        )
    )

    times_galsim = []
    times_jgalsim = []

    # prepare draw_function
    draw_fncs = []
    for _max_n_gals, _batch_slen in zip(MAX_N_GAL_BINS, SLEN_BINS):
        draw_fncs.append(
            partial(
                draw_jgs_scan_stamps,
                psf=xpsf,
                ilen=IMAGE_SLEN,
                fft_size=FFT_SIZE,
                max_n_gals=_max_n_gals,
                slen=_batch_slen,
            )
        )

    # timing results average over 100 samples
    pdf_name = Path("out") / f"residuals_{SUFFIX}.pdf"
    rkeys = random.split(random.PRNGKey(43), 100)
    with PdfPages(pdf_name) as pdf:
        for ii, rkey in tqdm(
            enumerate(rkeys), total=len(rkeys), desc="Timing galsim vs jax-galsim..."
        ):
            # sample in numpy
            sample, n, gsizes = get_one_full_sample(
                rkey, cat=cat, ilen=IMAGE_SLEN, max_n_gals=MAX_N_GALS
            )

            t1 = time.time()
            gs_arr = draw_galsim(
                sample,
                n,
                psf=psf,
                ilen=IMAGE_SLEN,
                max_n_gals=MAX_N_GALS,  # sanity
                max_slen=SLEN_BINS[-1],  # sanity
                # good_sizes=gsizes,
            )
            t2 = time.time()
            t_galsim = t2 - t1
            times_galsim.append(t_galsim)

            if ii == 0:  # trigger jit compilation for all draw function
                for _max_n_gals, draw_fnc in zip(MAX_N_GAL_BINS, draw_fncs):
                    _new_dict = {p: sample[p][:_max_n_gals] for p in sample}
                    draw_fnc(_new_dict).block_until_ready()

            sample_jax = device_put(sample, device=DEVICE)
            gsizes_jax = device_put(gsizes, device=DEVICE)
            assert gsizes_jax.shape == sample_jax["flux_b"].shape

            # jax draw
            t1 = time.time()
            # the following two lines will trigger transfer guard, but that is unavoidable when
            # creating arrays, care about other implicit ones that can happen below.
            # TODO: explain why unavoidable
            jgs_arr = jnp.zeros((IMAGE_SLEN, IMAGE_SLEN), device=DEVICE)
            _drawn = jnp.zeros_like(gsizes_jax, device=DEVICE).astype(bool)
            _drawn = _drawn | jnp.less_equal(gsizes_jax, 1)  # simplicity later
            with jax.transfer_guard("allow"):
                # split into batches using good sizes estimated from galsim
                for _draw_fnc, _max_n_gals, _sslen in zip(
                    draw_fncs, MAX_N_GAL_BINS, SLEN_BINS
                ):
                    _mask1 = ~_drawn
                    _mask2 = jnp.less_equal(
                        gsizes_jax, device_put(_sslen, device=DEVICE) - BUFFER
                    )
                    _mask = _mask1 & _mask2
                    if _mask.sum() == 0:
                        continue

                    # need to make new array that includes only galaxies in mask
                    # padded by 0 to match `_max_n_gals`
                    # throw error if array ends up being larger than this
                    _sample_jax = {}
                    for p in sample_jax:
                        if p in ("flux_b", "flux_d"):
                            _sample_jax[p] = sample_jax[p][_mask]
                        else:
                            _sample_jax[p] = sample_jax[p][_mask]

                    n_gals = len(_sample_jax["flux_b"])
                    assert n_gals <= _max_n_gals, f"{n_gals} larger than {_max_n_gals}"
                    # padding if required
                    for _ in range(n_gals, _max_n_gals, 1):
                        for p in _sample_jax:
                            _sample_jax[p] = np.append(_sample_jax[p], DUMMY_PARAMS[p])
                    assert len(_sample_jax["flux_b"]) == _max_n_gals

                    # assert not jnp.any(jnp.isnan(_jgs_arr))
                    # assert _jgs_arr.shape[0] == IMAGE_SLEN
                    _jgs_arr = _draw_fnc(_sample_jax).block_until_ready()
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
            f"Average time (per image) for GalSim: {np.mean(times_galsim):.3f} seconds",
            file=fp,
        )
        print(
            f"Median time (per image) for GalSim: {np.median(times_galsim):.3f} seconds",
            file=fp,
        )

        print(
            f"Average time (per image) for JAX-GalSim: {np.mean(times_jgalsim):.3f} seconds",
            file=fp,
        )
        print(
            f"Median time (per image) for JAX-GalSim: {np.median(times_jgalsim):.3f} seconds",
            file=fp,
        )


if __name__ == "__main__":
    main()
