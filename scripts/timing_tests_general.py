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
from jax import device_put, jit, random
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import jax_galsim as jgs

DEVICE = jax.devices()[0]

# TODO: consider splitting script into galsim/jax-galsim as we probably want to avoid
# wasting time running galsim in the GPU.


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
    max_mag: float = typer.Option(default=27.0),
    min_hlr: float = typer.Option(default=0.0),
    max_hlr: float = typer.Option(default=2.0),  # arcsecs
    extra_suffix: str = typer.Option(default=""),
    fix_galsim_stamp_size: bool = False,  # fixed to largest in stamp_slen_bins
    max_n_iters: int | None = typer.Option(None),
):
    # does not support multi-threading or multiprocessing
    # need to parse as str as typer does not support lists
    stamp_slen_bins = _parse_bins_str_input(stamp_slen_bins_str)
    max_n_gals_bins = _parse_bins_str_input(max_n_gals_bins_str)

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
        suffix=f"{psf_type}-07",
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
    times_jgalsim_transfer = []
    n_gals_record = []

    # prepare draw function for jax_galsim for each size bin
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

    # timing test
    pdf_name = out_folder / "residuals.pdf"
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
                    _new_dict_jax = device_put(
                        {p: sample[p][:_max_n_gals] for p in sample}, device=device
                    )
                    _draw_fnc(_new_dict_jax).block_until_ready()

            # galsim timing
            # TODO: do we guarantee no stamps are larger? or should we add check flag?
            t1 = time.time()
            gs_arr = draw_galsim(
                sample,
                n,
                psf=psf,
                ilen=image_slen,
                slen=stamp_size_galsim,
                max_slen=stamp_slen_bins[-1],  # sanity only
            )
            t2 = time.time()
            t_galsim = t2 - t1
            times_galsim.append(t_galsim)

            # timing device transfer for jax-galsim
            t1 = time.time()
            jgs_arr = jnp.zeros((image_slen, image_slen), device=device)
            sample_jax = device_put(sample, device=DEVICE)
            gsizes_jax = device_put(gsizes, device=DEVICE)
            assert gsizes_jax.shape == sample_jax["flux_b"].shape

            # this function allocates some extra memory probably,
            # not sure if it's a concern but could not figure out how to
            # avoid transfer guard error otherwise
            samples_split = setup_draw_jgalsim_size_bins(
                sample_jax=sample_jax,
                gsizes_jax=gsizes_jax,
                max_n_gals_bins=max_n_gals_bins,
                stamp_slen_bins=stamp_slen_bins,
                max_n_iters=max_n_iters,
                buffer=buffer,
                device=device,
            )
            t2 = time.time()
            times_jgalsim_transfer.append(t2 - t1)

            # jax galsim timing
            t1 = time.time()
            with jax.transfer_guard("disallow"):
                # split into batches using good sizes estimated from galsim
                for jj, draw_fnc_jj in enumerate(draw_fncs):
                    n_iters = len(samples_split[jj])
                    for kk in range(n_iters):
                        _jgs_arr = draw_fnc_jj(
                            samples_split[jj][kk]
                        ).block_until_ready()
                        jgs_arr += _jgs_arr

            t2 = time.time()
            t_jgalsim = t2 - t1
            times_jgalsim.append(t_jgalsim)

            # write down record for potential refinement
            n_gals_record.append(_create_record(gsizes, stamp_slen_bins, buffer))

            # save residual images to a multipage pdf for inspection
            add_results_to_pdf(gs_arr, np.array(jgs_arr), t_galsim, t_jgalsim, ii, pdf)

    with open(out_folder / "record.json", "w") as fp:
        json.dump(n_gals_record, fp, indent="\t")

    _save_timing_results(
        out_folder=out_folder,
        times_galsim=times_galsim,
        times_jgalsim=times_jgalsim,
        times_transfer=times_jgalsim_transfer,
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
    elif bins_str.count(",") == 1:
        # something like '60,' was passed in, remove comma and try to cast
        return (int(bins_str[:-1]),)
    else:
        the_bins = [float(x) for x in bins_str.split(",")]
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
            f"Average time (per image) for GalSim: {np.mean(times_galsim):.3f} seconds",
            file=fp,
        )
        print(
            f"Median time (per image) for GalSim: {np.median(times_galsim):.3f} seconds",
            file=fp,
        )

        print(file=fp)
        print(
            f"Average time (per image) for JAX-GalSim: {np.mean(times_jgalsim):.3f} seconds",
            file=fp,
        )
        print(
            f"Median time (per image) for JAX-GalSim: {np.median(times_jgalsim):.3f} seconds",
            file=fp,
        )

        print(file=fp)
        print(
            f"Average JAX transfer time (per image): {np.mean(times_transfer):.3f} seconds",
            file=fp,
        )
        print(
            f"Median JAX transfer time (per image): {np.median(times_transfer):.3f} seconds",
            file=fp,
        )


# TODO: perhaps we can jit this or replace the first for loop with a scan?
# could this be made faster? the problem is `_n_iters_needed` is dynamic
def setup_draw_jgalsim_size_bins(
    *,
    sample_jax,
    gsizes_jax,
    max_n_gals_bins,
    stamp_slen_bins,
    buffer: int,
    max_n_iters: int,
    device,
):
    """Prepare samples that will be used by jax_galsim to avoid transfer guard."""
    # we mark objects zeroed out as drawn already

    _drawn = jnp.zeros_like(gsizes_jax, device=device).astype(bool)
    _drawn |= jnp.less_equal(gsizes_jax, 1)
    _dummy_jax = device_put(DUMMY_PARAMS, device=device)

    samples_out = []

    for jj, (_max_n_gals, _sslen) in enumerate(zip(max_n_gals_bins, stamp_slen_bins)):
        samples_out.append([])
        _mask1 = ~_drawn
        _sslen_jax = device_put(_sslen, device=device)
        _mask2 = jnp.less_equal(gsizes_jax, _sslen_jax - buffer)
        _mask = _mask1 & _mask2

        # no objects to draw in this bin
        if _mask.sum() == device_put(0, device=device):
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
            _sample_kk = {k: v[_mask][idx1:idx2] for k, v in sample_jax.items()}
            _n_gals_kk = len(_sample_kk["flux_b"])

            # add zeroed out sources if necessary
            for _ in range(_n_gals_kk, _max_n_gals, 1):
                for p in _sample_kk:
                    _sample_kk[p] = np.append(_sample_kk[p], _dummy_jax[p])
            assert len(_sample_kk["flux_b"]) == _max_n_gals
            samples_out[jj].append(_sample_kk)
        _drawn = _drawn.at[_mask].set(True)
    assert jnp.all(_drawn)
    return device_put(samples_out, device=device)


if __name__ == "__main__":
    typer.run(main)
