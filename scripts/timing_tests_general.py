#!/usr/bin/env python3

import math
import os

os.environ["JAX_ENABLE_X64"] = "True"
import json
import pickle
import time
from functools import partial
from pathlib import Path

import galsim
import jax
import jax.numpy as jnp
import numpy as np
import typer
from draw_scene_functions import (
    DUMMY_PARAMS,
    add_results_to_pdf,
    draw_galsim,
    draw_jgs_scan_stamps,
    draw_jgs_vmap_stamps,
    get_good_sizes_galsim,
    get_one_full_sample,
    prepare_catalog,
)
from jax import Array, device_put, jit, random
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import jax_galsim as jgs

DEVICE = jax.devices()[0]


def main(
    stamp_bins_str: str = typer.Option(),
    max_n_gals_bins_str=typer.Option(),
    image_slen: int = typer.Option(),
    max_n_gals_global: int = typer.Option(),
    n_samples: int = typer.Option(help="How many big images do you want?"),
    catsim_fpath: str = "../../../Downloads/catsim/OneDegSq.fits",
    outdir: str = typer.Option(),
    scan_or_vmap: str = typer.Option(default="scan"),
    cpu_or_gpu: str = typer.Option(default="cpu"),
    psf_type: str = typer.Option(default="gaussian"),
    buffer: int = typer.Option(default=3),
    fft_size: int = typer.Option(default=128),
    seed: int = typer.Option(default=42),
    min_mag: float = typer.Option(default=20.0),
    max_mag: float = typer.Option(default=27.0),
    min_hlr: float = typer.Option(default=0.0),
    max_hlr: float = typer.Option(default=2.0),  # arcsecs
    extra_suffix: str = typer.Option(default=""),
    fix_galsim_stamp_size: bool = False,  # fixed to largest in stamp_slen_bins
):
    stamp_bins = _parse_bins_str_input(stamp_bins_str)
    max_n_gals_bins = _parse_bins_str_input(max_n_gals_bins_str)
    # does not support multi-threading or multiprocessing
    assert tuple(sorted(stamp_slen_bins)) == stamp_slen_bins
    assert scan_or_vmap in ("scan", "vmap")
    assert cpu_or_gpu in ("cpu", "gpu")

    if fix_galsim_stamp_size:
        # only supported for case when only 1 bin is being used
        # otherwise galsim is used too inefficiently to be useful
        assert len(max_n_gals_bins) == len(stamp_slen_bins) == 1

    if cpu_or_gpu == "cpu":
        device = jax.devices("cpu")[0]
    else:
        device = jax.devices("gpu")[0]

    max_stamp_size = max(stamp_slen_bins)
    if psf_type == "gaussian":
        psf = galsim.Gaussian(half_light_radius=0.7, flux=1.0)
        xpsf = jgs.Gaussian(half_light_radius=0.7, flux=1.0)

    elif psf_type == "moffat":
        # beta from galsim tutorial 2
        psf = galsim.Moffat(half_light_radius=0.7, beta=5.0, flux=1.0)
        xpsf = jgs.Moffat(half_light_radius=0.7, beta=5.0, flux=1.0)

    else:
        raise NotImplementedError("Only 'gaussian' or 'moffat' are supported.")

    out_path = Path(outdir)
    assert out_path.exists()

    # get hash for bins specified
    bin_hash = _get_bins_hash(out_path, stamp_slen_bins, max_n_gals_bins)

    # hash for unique folder name
    fix_galsim_str = "fix-galsim-" if fix_galsim_stamp_size else ""
    hash_name = (
        f"{n_samples}-{image_slen}-{psf_type}-{fft_size}-{seed}-"
        f"hb{bin_hash}-{cpu_or_gpu}-{fix_galsim_str}{extra_suffix}"
    )

    out_folder = out_path / hash_name
    out_folder.mkdir(parents=False, exist_ok=True)

    # catalog preparation and masking
    cat = prepare_catalog(
        catsim_fpath, min_hlr=min_hlr, max_hlr=max_hlr, min_mag=min_mag, max_mag=max_mag
    )
    n1 = len(cat)
    good_sizes = get_good_sizes_galsim(
        cat=cat, psf=psf, overwrite=False, out_path=out_path, suffix=f"{psf_type}-07"
    )
    cat["good_size"] = good_sizes
    mask_good_size = cat["good_size"] < max_stamp_size - buffer
    cat = cat[mask_good_size]

    print(
        f"INFO: Catalog prepared with {len(cat)} galaxies after (good size) cut "
        f"(before this cut {n1}). Percentage included out: {len(cat) / n1 * 100:2f}%"
    )

    times_galsim = []
    times_jgalsim = []

    # prepare draw_function
    draw_fncs = []
    _draw_fnc_raw = (
        draw_jgs_scan_stamps if scan_or_vmap == "scan" else draw_jgs_vmap_stamps
    )
    for _max_n_gals, _batch_slen in zip(max_n_gals_bins, stamp_slen_bins):
        draw_fncs.append(
            jit(
                partial(
                    _draw_fnc_raw,
                    psf=xpsf,
                    ilen=image_slen,
                    fft_size=fft_size,
                    max_n_gals=_max_n_gals,
                    slen=_batch_slen,
                )
            )
        )

    # timing results average over 100 samples
    pdf_name = Path("out") / f"residuals_{hash_name}.pdf"
    n_gals_record = []
    rkeys = random.split(random.PRNGKey(seed), n_samples)
    with PdfPages(pdf_name) as pdf:
        for ii, rkey in tqdm(
            enumerate(rkeys), total=len(rkeys), desc="Timing galsim vs jax-galsim..."
        ):
            # sample in numpy
            sample, n, gsizes = get_one_full_sample(
                rkey, cat=cat, ilen=image_slen, max_n_gals=max_n_gals_global
            )

            # trigger jit compilation for all draw function
            if ii == 0:
                for _max_n_gals, _draw_fnc in zip(max_n_gals_bins, draw_fncs):
                    _new_dict = {p: sample[p][:_max_n_gals] for p in sample}
                    _draw_fnc(_new_dict).block_until_ready()

            # galsim timing
            t1 = time.time()
            gs_arr = draw_galsim(sample, n, psf=psf, ilen=image_slen)
            t2 = time.time()
            t_galsim = t2 - t1
            times_galsim.append(t_galsim)

            # do not measure this size
            sample_jax = device_put(sample, device=DEVICE)
            gsizes_jax = device_put(gsizes, device=DEVICE)
            assert gsizes_jax.shape == sample_jax["flux_b"].shape

            # jax galsim timing
            t1 = time.time()
            jgs_arr = draw_jax_galsim_size_bins(
                sample_jax,
                stamp_slen_bins=stamp_slen_bins,
                max_n_gals_bins=max_n_gals_bins,
                draw_fncs=draw_fncs,
                ilen=image_slen,
                gsizes_jax=gsizes_jax,
                device=device,
                max_n_iters=None,
            )
            t2 = time.time()
            t_jgalsim = t2 - t1
            times_jgalsim.append(t_jgalsim)

            # write down record for potential refinement
            n_gals_record.append((gsizes_jax))

            # save residual images to a multipage pdf for inspection
            add_results_to_pdf(gs_arr, np.array(jgs_arr), t_galsim, t_jgalsim, ii, pdf)

    with open(out_folder / "record.pickle", "wb") as handle:
        pickle.dump(n_gals_record, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _save_timing_results(
        suffix=hash_name, times_galsim=times_galsim, times_jgalsim=times_jgalsim
    )


def _save_timing_results(*, times_galsim: list, times_jgalsim: list, out_folder: Path):
    # print summary timing results
    summary_fname = out_folder / "summary.txt"
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


def draw_jax_galsim_size_bins(
    samples_jax: dict,
    *,
    stamp_slen_bins: int,  # should be sorted
    max_n_gals_bins: int,  # maximum number of galaxies in each bin
    draw_fncs: int,
    ilen: int,
    gsizes_jax: Array,
    device,
    max_n_iters: int | None = None,
    buffer: int,
):
    # cannot put inside tranfer guard
    jgs_arr = jnp.zeros((ilen, ilen), device=device)
    _drawn = jnp.zeros_like(gsizes_jax, device=device).astype(bool)
    buffer_jax = device_put(buffer)
    DUMMY_PARAMS_JAX = device_put(DUMMY_PARAMS, device=device)

    # we mark objects zeroed out as drawn already
    _drawn = _drawn | jnp.less_equal(gsizes_jax, 1)

    with jax.transfer_guard("disable"):
        # split into batches using good sizes estimated from galsim
        for _draw_fnc, _max_n_gals, _sslen in zip(
            draw_fncs, max_n_gals_bins, stamp_slen_bins
        ):
            _mask1 = ~_drawn
            _sslen_jax = device_put(_sslen, device=device)
            _mask2 = jnp.less_equal(gsizes_jax, _sslen_jax - buffer_jax)
            _mask = _mask1 & _mask2
            if _mask.sum() == 0:  # no objects to draw in this bin
                continue

            n_gals = int(_mask.sum().item())
            _n_iters_needed = math.ceil(n_gals / _max_n_gals)

            if max_n_iters:
                assert _n_iters_needed <= max_n_iters, (
                    f"Consider increasing max_n_gals (n_gals: {n_gals}, max_n_gals:{_max_n_gals}) in bin {_sslen}, niters: {_n_iters_needed}"
                )

            for kk in range(_n_iters_needed):
                idx1 = kk * _max_n_gals
                idx2 = (kk + 1) * _max_n_gals
                _sample_kk = {k: v[_mask][idx1:idx2] for k, v in samples_jax.items()}
                _n_gals_kk = len(_sample_kk["flux_b"])

                # add zeroed out sources if necessary
                for _ in range(_n_gals_kk, _max_n_gals, 1):
                    for p in _sample_kk:
                        _sample_kk[p] = np.append(_sample_kk[p], DUMMY_PARAMS_JAX[p])
                assert len(_sample_kk["flux_b"]) == _max_n_gals

                _jgs_arr = _draw_fnc(_sample_kk).block_until_ready()
                jgs_arr += _jgs_arr
            _drawn = _drawn.at[_mask].set(True)

    assert _drawn.all(), "Not all real galaxies were drawn."
    return jgs_arr


def _get_bins_hash(out_path: Path, *, stamp_slen_bins, max_n_gals_bins) -> int:
    # very simple, just save these bins with an "id" if never has been used before
    # otherwise assign next one available
    hash_json = out_path / "bin_hash.json"
    _key = (*stamp_slen_bins, *max_n_gals_bins)  # just unpack tuple

    h = {}  # will be overwritten if exists
    if hash_json.exists():
        with open(hash_json, "r") as handle:
            h = json.load(handle)
            if _key in h:
                the_hash = h[_key]
            else:
                the_hash = max(list(h.values())) + 1
                h[_key] = the_hash
    else:
        h = {}
        h[_key] = 0
        the_hash = 0
    with open(hash_json, "w") as handle:
        json.dump(h, handle)
    return the_hash


def _parse_bins_str_input(bins_str: str):
    the_bins = [float(x) for x in bins_str.split(",")]
    # they need to bwe integers
    for x in the_bins:
        assert int(x) == x
    the_bins = [int(x) for x in the_bins]
    return the_bins


if __name__ == "__main__":
    typer.run(main)
