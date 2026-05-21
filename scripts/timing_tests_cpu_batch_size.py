#!/usr/bin/env python3

import math
import os

os.environ["JAX_ENABLE_X64"] = "True"
import pickle
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
from jax import device_put, jit, random
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import jax_galsim as jgs

# TODO: add functionality to handle small edge cases larger than _max_n_gal
# (for loop within scan?)


IMAGE_SLEN = 1000
N_SAMPLES = 152

# good image size (isolated galaxy) and stamp size used in scene by galsim differ
BUFFER = 3  # sometimes good size below and final stamp size differ by a small amount, IDK why
BACKGROUND = get_default_lsst_background()
FFT_SIZE = 128
SLEN_BINS = (61, 81, 101)
# MAX_N_GALS = 150 * FACTOR
# MAX_N_GAL_BINS = (100 * FACTOR, 10 * FACTOR, 5 * FACTOR)
MAX_N_GALS = 1600
MAX_N_GAL_BINS = (1000, 100, 20)
DEVICE = jax.devices()[0]
assert sorted(SLEN_BINS) == list(SLEN_BINS), "Needs to be sorted"
MAX_N_ITERS = 2

SUFFIX = f"{IMAGE_SLEN}-{SLEN_BINS[-1]}-{N_SAMPLES}-scan-cpu-batch-sizes"


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
            jit(
                partial(
                    draw_jgs_scan_stamps,
                    psf=xpsf,
                    ilen=IMAGE_SLEN,
                    fft_size=FFT_SIZE,
                    max_n_gals=_max_n_gals,
                    slen=_batch_slen,
                )
            )
        )

    # timing results average over 100 samples
    pdf_name = Path("out") / f"residuals_{SUFFIX}.pdf"
    n_gals_record = []
    rkeys = random.split(random.PRNGKey(43), N_SAMPLES)
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

            _n_gals_record = []
            with jax.transfer_guard("allow"):
                # split into batches using good sizes estimated from galsim
                for _draw_fnc, _max_n_gals, _sslen in zip(
                    draw_fncs, MAX_N_GAL_BINS, SLEN_BINS
                ):
                    _mask1 = ~_drawn
                    _sslen_jax = device_put(_sslen, device=DEVICE)
                    _mask2 = jnp.less_equal(gsizes_jax, _sslen_jax - BUFFER)
                    _mask = _mask1 & _mask2
                    if _mask.sum() == 0:
                        continue

                    n_gals = int(_mask.sum().item())
                    _n_iters_needed = math.ceil(n_gals / _max_n_gals)
                    assert _n_iters_needed <= MAX_N_ITERS, (
                        f"Consider increasing max_n_gals (n_gals: {n_gals}, max_n_gals:{_max_n_gals}) in bin {_sslen}, niters: {_n_iters_needed}"
                    )
                    # assert n_gals <= _max_n_gals, f"{n_gals} larger than {_max_n_gals}"
                    # print("ngals:", n_gals)
                    # print("niters:", _n_iters_needed)
                    for kk in range(_n_iters_needed):
                        idx1 = kk * _max_n_gals
                        idx2 = (kk + 1) * _max_n_gals
                        _sample_kk = {
                            k: v[_mask][idx1:idx2] for k, v in sample_jax.items()
                        }
                        _n_gals_kk = len(_sample_kk["flux_b"])

                        # padding if required
                        for _ in range(_n_gals_kk, _max_n_gals, 1):
                            for p in _sample_kk:
                                _sample_kk[p] = np.append(
                                    _sample_kk[p], DUMMY_PARAMS[p]
                                )
                        assert len(_sample_kk["flux_b"]) == _max_n_gals

                        # assert not jnp.any(jnp.isnan(_jgs_arr))
                        # assert _jgs_arr.shape[0] == IMAGE_SLEN
                        _jgs_arr = _draw_fnc(_sample_kk).block_until_ready()
                        jgs_arr += _jgs_arr

                    _drawn = _drawn.at[_mask].set(True)
                    _n_gals_record.append(n_gals)
            n_gals_record.append(_n_gals_record)
            t2 = time.time()
            t_jgalsim = t2 - t1
            assert jnp.all(_drawn)
            times_jgalsim.append(t_jgalsim)

            # save residual images to a multipage pdf for inspection
            add_results_to_pdf(gs_arr, np.array(jgs_arr), t_galsim, t_jgalsim, ii, pdf)

    with open(f"out/record-{N_SAMPLES}-{IMAGE_SLEN}.pickle", "wb") as handle:
        pickle.dump(n_gals_record, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
