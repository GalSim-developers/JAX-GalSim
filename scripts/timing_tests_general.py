#!/usr/bin/env python3

import math
import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
import json
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
from jax import block_until_ready, device_put, jit, random
from jax.lax import fori_loop
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import jax_galsim as jgs

# TODO: consider splitting script into galsim/jax-galsim as we probably want to avoid
# wasting time running galsim in the GPU, and/or running the same images over and over


def main(
    stamp_slen_bins_str: str = typer.Option(),
    max_n_gals_bins_str=typer.Option(),
    image_slen: int = typer.Option(),
    max_n_gals_global: int = typer.Option(),
    n_samples: int = typer.Option(help="How many big images do you want?"),
    catsim_fpath: str = "../../Downloads/catsim/OneDegSq.fits",
    outdir: str = typer.Option(),
    scan_or_vmap: str = typer.Option(default="scan"),
    cpu_or_gpu: str = typer.Option(default="cpu"),
    psf_type: str = typer.Option(default="gaussian"),
    buffer: int = typer.Option(default=3),
    fft_size: int = typer.Option(default=128),
    seed: int = typer.Option(default=42),
    min_mag: float = typer.Option(default=20.0),
    max_n_iters: int = typer.Option(default=2),
    max_mag: float = typer.Option(default=27.0),
    min_hlr: float = typer.Option(default=0.0),
    max_hlr: float = typer.Option(default=2.0),  # arcsecs
    extra_suffix: str = typer.Option(default=""),
    fix_galsim_stamp_size: bool = False,  # fixed to largest in stamp_slen_bins
):
    # does not support multi-threading or multiprocessing
    # need to parse as str as typer does not support lists
    stamp_slen_bins = _parse_bins_str_input(stamp_slen_bins_str)
    max_n_gals_bins = _parse_bins_str_input(max_n_gals_bins_str)

    n_bins = len(stamp_slen_bins)
    assert n_bins == len(max_n_gals_bins)
    assert tuple(sorted(stamp_slen_bins)) == stamp_slen_bins
    assert scan_or_vmap in ("scan", "vmap")
    assert cpu_or_gpu in ("cpu", "gpu")

    stamp_size_galsim = None
    if fix_galsim_stamp_size:
        # only supported for case when only 1 bin is being used
        # otherwise galsim would be used too inefficiently for this
        # to be a useful comparison
        assert len(max_n_gals_bins) == len(stamp_slen_bins) == 1
        stamp_size_galsim = stamp_slen_bins[0]

    if cpu_or_gpu == "cpu":
        device = jax.devices("cpu")[0]
    elif cpu_or_gpu == "gpu":
        device = jax.devices("gpu")[0]
    else:
        raise ValueError()

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

    out_root_path = Path(outdir)
    assert out_root_path.exists()

    # get hash for specified bin argument
    bin_hash = _get_bins_hash(
        out_root_path, stamp_slen_bins=stamp_slen_bins, max_n_gals_bins=max_n_gals_bins
    )

    # create unique folder name
    fix_galsim_str = "-fix-galsim" if fix_galsim_stamp_size else ""
    extra_suffix_str = f"-{extra_suffix}" if extra_suffix else ""
    hash_name = (
        f"{image_slen}-{n_samples}-{psf_type}-{fft_size}-{seed}-"
        f"hb{bin_hash}-{cpu_or_gpu}-{scan_or_vmap}{fix_galsim_str}{extra_suffix_str}"
    )

    out_folder = out_root_path / hash_name
    out_folder.mkdir(parents=False, exist_ok=True)

    # catalog preparation and masking
    cat = prepare_catalog(
        catsim_fpath, min_hlr=min_hlr, max_hlr=max_hlr, min_mag=min_mag, max_mag=max_mag
    )
    n1 = len(cat)
    good_sizes = get_good_sizes_galsim(
        cat=cat,
        psf=psf,
        overwrite=False,
        out_path=out_root_path,
        suffix=f"{psf_type}-07",  # TODO: need to hash more psf properties just in case (size,beta)
    )
    cat["good_size"] = good_sizes

    # buffer is needed for a few pixel difference that sometimes occurs between estimating
    # good size on isolated images like in the function above and the stamp size galsim
    # ultimately uses when drawing stamp onto the big image
    mask_good_size = cat["good_size"] < max_stamp_size - buffer
    cat = cat[mask_good_size]

    print(
        f"INFO: Catalog prepared with {len(cat)} galaxies after (good size) cut "
        f"(before this cut {n1}). Percentage included is: {len(cat) / n1 * 100:2f}%"
    )

    times_galsim = []
    times_jgalsim = []
    times_transfer = []
    n_gals_record = []

    # prepare draw function for jax_galsim for each size bin
    draw_fncs = []
    draw_fnc_raw = (
        draw_jgs_scan_stamps if scan_or_vmap == "scan" else draw_jgs_vmap_stamps
    )
    for ii in range(n_bins):
        _max_n_gals = max_n_gals_bins[ii]
        _stamp_slen = stamp_slen_bins[ii]
        draw_fncs.append(
            jit(
                partial(
                    draw_fnc_raw,
                    psf=xpsf,
                    ilen=image_slen,
                    fft_size=fft_size,
                    max_n_gals=_max_n_gals,
                    slen=_stamp_slen,
                )
            )
        )

    # now jit the all bin draw function, will compile below
    all_draw_fnc = jit(
        partial(
            draw_all_bins_jgs,
            ilen=image_slen,
            draw_fncs=draw_fncs,
            device=device,
            n_bins=n_bins,
        )
    )

    # timing start
    pdf_name = out_folder / "residuals.pdf"
    rkeys = random.split(random.PRNGKey(seed), n_samples)
    with PdfPages(pdf_name) as pdf:
        for ii, rkey in tqdm(
            enumerate(rkeys), total=n_samples, desc="Timing galsim vs jax-galsim..."
        ):
            # sample in numpy
            sample, n, gsizes = get_one_full_sample(
                rkey, cat=cat, ilen=image_slen, max_n_gals=max_n_gals_global
            )
            assert sample["flux_b"].shape == (n,)
            assert gsizes.shape == (n,)
            assert np.all(gsizes > 1)

            # galsim timing
            t1 = time.time()
            gs_arr = draw_galsim(
                sample,
                n,
                psf=psf,
                ilen=image_slen,
                slen=stamp_size_galsim,
                max_slen=stamp_slen_bins[-1],  # sanity
            )
            t2 = time.time()
            t_galsim = t2 - t1
            times_galsim.append(t_galsim)

            # transfer to device, separate sampled parameters into bins, and time separately
            t1 = time.time()
            samples_per_bin, n_iters_per_bin = _prepare_per_bin_samples(
                sample,
                gsizes,
                stamp_slen_bins=stamp_slen_bins,
                max_n_gals_bins=max_n_gals_bins,
                max_n_iters=max_n_iters,
                buffer=buffer,
            )

            # need block until ready here to ensure computation+transfer happens!
            samples_per_bin_jax = block_until_ready(
                device_put(samples_per_bin, device=device)
            )
            n_iters_per_bin_jax = block_until_ready(
                device_put(n_iters_per_bin, device=device)
            )
            t2 = time.time()
            times_transfer.append(t2 - t1)
            assert n_bins == len(samples_per_bin) == len(n_iters_per_bin)
            del samples_per_bin, n_iters_per_bin  # numpy versions no longer needed

            # compilation (not timed)
            if ii == 0:
                _ = block_until_ready(
                    all_draw_fnc(samples_per_bin_jax, n_iters_per_bin_jax)
                )

            # jax galsim timing
            t1 = time.time()
            with jax.transfer_guard("disallow"):
                jgs_arr = block_until_ready(
                    all_draw_fnc(samples_per_bin_jax, n_iters_per_bin_jax)
                )
            t2 = time.time()
            t_jgalsim = t2 - t1
            times_jgalsim.append(t_jgalsim)

            # save residual images to a multipage pdf for inspection
            add_results_to_pdf(
                ii,
                pdf,
                gs_arr=gs_arr,
                jgs_np_arr=np.array(jgs_arr),
                t_galsim=t_galsim,
                t_jgalsim=t_jgalsim,
            )

            # write down record for potential refinement of bins (only on CPU)
            if cpu_or_gpu == "cpu":
                n_gals_record.append(_create_record(gsizes, stamp_slen_bins, buffer))
                if ii == n_samples - 1:
                    with open(out_folder / "record.json", "w") as fp:
                        json.dump(n_gals_record, fp, indent="\t")

            # free memory as appropriate
            del gs_arr, jgs_arr, samples_per_bin_jax, n_iters_per_bin_jax

    _save_timing_results(
        out_folder=out_folder,
        times_galsim=times_galsim,
        times_jgalsim=times_jgalsim,
        times_transfer=times_transfer,
    )


def _get_bins_hash(out_path: Path, *, stamp_slen_bins, max_n_gals_bins) -> int:
    # very simple, just save these bins with an "id" if never has been used before
    # otherwise assign next one available
    hash_json = out_path / "bin_hash.json"
    _key = (*stamp_slen_bins, *max_n_gals_bins)  # just unpack tuple
    _key = "-".join([str(x) for x in _key])

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
    if "," not in bins_str:
        # try to cast as a single number
        return (int(bins_str),)
    else:
        the_bins = [float(x) for x in bins_str.split(",")]  # fail on "60," and "60,80,"
        # they need to be integers
        for x in the_bins:
            assert int(x) == x
        the_bins = [int(x) for x in the_bins]
        return tuple(the_bins)


def _create_record(gsizes, stamp_slen_bins, buffer):
    _record = []
    for ii in range(len(stamp_slen_bins)):
        if ii == 0:
            s1 = stamp_slen_bins[0]
            dummy_mask = gsizes > 1
            _record.append(int(np.sum(dummy_mask & (gsizes <= s1 - buffer))))

        else:
            s1 = stamp_slen_bins[ii - 1] - buffer
            s2 = stamp_slen_bins[ii] - buffer
            _record.append(int(np.sum((gsizes > s1) & (gsizes <= s2))))
    return _record


def _save_timing_results(
    *, out_folder: Path, times_galsim: list, times_jgalsim: list, times_transfer: list
):
    # print summary timing results
    summary_fname = out_folder / "summary.txt"
    with open(summary_fname, "w") as fp:
        print(
            f"Average time (per image) for GalSim: {np.mean(times_galsim):.4f} seconds",
            file=fp,
        )
        print(
            f"Median time (per image) for GalSim: {np.median(times_galsim):.4f} seconds",
            file=fp,
        )

        print(file=fp)
        print(
            f"Average time (per image) for JAX-GalSim: {np.mean(times_jgalsim):.4f} seconds",
            file=fp,
        )
        print(
            f"Median time (per image) for JAX-GalSim: {np.median(times_jgalsim):.4f} seconds",
            file=fp,
        )

        print(file=fp)
        print(
            f"Average JAX transfer time (per image): {np.mean(times_transfer):.4f} seconds",
            file=fp,
        )
        print(
            f"Median JAX transfer time (per image): {np.median(times_transfer):.4f} seconds",
            file=fp,
        )


def _prepare_per_bin_samples(
    sample: np.ndarray,
    gsizes: np.ndarray,
    *,
    stamp_slen_bins: tuple,
    max_n_gals_bins: tuple,
    max_n_iters: int,
    buffer: int,
):

    samples_per_bin = []
    n_iters_per_bin = []
    n_bins = len(stamp_slen_bins)

    # keep track of already assigned galaxies
    # ignore ones that are dummies in sample
    _already_assigned = np.zeros_like(gsizes).astype(bool)
    for jj in range(n_bins):
        stamp_slen_jj = stamp_slen_bins[jj]
        max_n_gals_jj = max_n_gals_bins[jj]

        _mask1 = ~_already_assigned
        _mask2 = np.less_equal(gsizes, stamp_slen_jj - buffer)
        _mask = _mask1 & _mask2
        sample_jj = {k: v[_mask] for k, v in sample.items()}

        n_gals = _mask.sum().item()
        n_iters_jj = math.ceil(n_gals / max_n_gals_jj)
        assert n_iters_jj <= max_n_iters, (
            f"Number of iterations in size bin {jj} is {n_iters_jj} which is larger than max_n_iters:{max_n_iters}"
        )

        # here we want static shapes (small memory overheard with parameters)
        # but in the drawing function will explicitly skip in while loop these extra ones
        # based on n_iters_per_bin ==> especially useful for vmap
        n_pad = max_n_iters * max_n_gals_jj - n_gals
        for p in sample_jj:
            _padding = np.full(n_pad, fill_value=DUMMY_PARAMS[p])
            sample_jj[p] = np.concatenate([sample_jj[p], _padding])
            sample_jj[p] = sample_jj[p].reshape(max_n_iters, max_n_gals_jj)

        samples_per_bin.append(sample_jj)
        n_iters_per_bin.append(n_iters_jj)
        _already_assigned[_mask] = True

    return samples_per_bin, np.array(n_iters_per_bin)


def draw_all_bins_jgs(
    samples_per_bin_jax,
    n_iters_per_bin,
    *,
    ilen: int,
    draw_fncs: tuple,
    n_bins: int,
    device,
):
    param_names = tuple(samples_per_bin_jax[0].keys())
    jgs_arr = jnp.zeros((ilen, ilen), device=device, dtype=jnp.float64)

    for jj in range(n_bins):  # unrolled at traced time, could vmap?
        draw_fnc_jj = draw_fncs[jj]
        n_iters_jj = n_iters_per_bin[jj]
        samples_per_bin_jj = samples_per_bin_jax[jj]

        def _body_fnc(kk, arr):
            _batch = {p: samples_per_bin_jj[p][kk] for p in param_names}
            return arr + draw_fnc_jj(_batch)

        jgs_arr = fori_loop(0, n_iters_jj, _body_fnc, jgs_arr)

    return jgs_arr


if __name__ == "__main__":
    typer.run(main)
