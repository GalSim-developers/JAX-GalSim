#!/usr/bin/env python3

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
import matplotlib.pyplot as plt
import numpy as np
import typer
from draw_scene_functions import (
    add_results_to_pdf,
    draw_all_bins_jgs,
    draw_galsim,
    draw_jgs_scan_stamps,
    draw_jgs_vmap_stamps,
    get_good_sizes_galsim,
    get_one_full_sample,
    prepare_catalog,
    prepare_per_bin_samples,
)
from jax import block_until_ready, device_put, jit, random
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import jax_galsim as jgs

# metadetect lsst paper
BETA_PSF = 2.5
FWHM_PSF = 0.8  # probably was the requirement (Axel)
MIN_FWHM_PSF = 0.7
MAX_FWHM_PSF = 1.0

# tolerance for how positive residual (gs-jgs) can be for a given image (lower => more strict)
# some elements of the residual will be positive due to numerics and can be above threshold esp.
# for very bright objects so this does not necessarily indicate an issue.
POS_RESIDUAL_THRESHOLD = 1


def main(
    stamp_slen_bins_str: str = typer.Option(),
    max_n_gals_bins_str: str = typer.Option(),
    fft_size_bins_str: str = typer.Option(),
    image_slen: int = typer.Option(),
    max_n_gals_global: int = typer.Option(),
    n_samples: int = typer.Option(help="How many big images do you want?"),
    catsim_fpath: str = "../../Downloads/catsim/OneDegSq.fits",
    outdir: str = typer.Option(),
    scan_or_vmap: str = typer.Option(default="scan"),
    cpu_or_gpu: str = typer.Option(default="cpu"),
    psf_type: str = typer.Option(default="gaussian"),
    buffer: int = typer.Option(default=4),
    seed: int = typer.Option(default=42),
    max_n_iters: int = typer.Option(default=5),
    max_mag: float = typer.Option(default=27.0),  # minimal cuts
    min_hlr: float = typer.Option(default=0.0),  # minimal cuts
    fix_galsim_stamp_size: bool = False,  # fixed to largest in stamp_slen_bins
    fix_galsim_fft_size: bool = False,
    check_stamp_sizes: bool = False,
    include_outliers: bool = False,
    outlier_fraction: float = 1e-3,
    progress_bar: bool = True,
    verbose: bool = False,
):
    # does not support multi-threading or multiprocessing
    # need to parse as str as typer does not support lists
    stamp_slen_bins = _parse_bins_str_input(stamp_slen_bins_str)
    max_n_gals_bins = _parse_bins_str_input(max_n_gals_bins_str)
    fft_size_bins = _parse_bins_str_input(fft_size_bins_str)

    n_bins = len(stamp_slen_bins)
    assert n_bins == len(max_n_gals_bins) == len(fft_size_bins)
    assert tuple(sorted(stamp_slen_bins)) == stamp_slen_bins
    assert tuple(sorted(fft_size_bins)) == fft_size_bins
    assert tuple(sorted(max_n_gals_bins, reverse=True)) == max_n_gals_bins
    assert scan_or_vmap in ("scan", "vmap")
    assert cpu_or_gpu in ("cpu", "gpu")

    stamp_size_galsim = None
    if fix_galsim_stamp_size:
        # only supported for case when only 1 bin is being used
        # otherwise galsim would be used too inefficiently for this
        # to be a useful comparison
        print("INFO: Fixing Stamp Size for GalSim (not a production run)")
        assert len(max_n_gals_bins) == len(stamp_slen_bins) == 1
        stamp_size_galsim = stamp_slen_bins[0]

    fft_size_galsim = None
    if fix_galsim_fft_size:
        print("INFO: Fixing FFT Size for GalSim (not a production run)")
        assert len(fft_size_bins) == 1
        fft_size_galsim = fft_size_bins[-1]

    if check_stamp_sizes:
        print("INFO: Stamp sizes are being checked by GalSim (not a production run).")

    if cpu_or_gpu == "cpu":
        device = jax.devices("cpu")[0]
    elif cpu_or_gpu == "gpu":
        device = jax.devices("gpu")[0]
    else:
        raise ValueError()

    out_root_path = Path(outdir)
    assert out_root_path.exists(), "Need to create root output directory."

    # get hash for specified bin argument
    bin_hash = _get_bins_hash(
        out_root_path,
        stamp_slen_bins=stamp_slen_bins,
        max_n_gals_bins=max_n_gals_bins,
        fft_size_bins=fft_size_bins,
    )

    # create unique folder name
    fix_str = ""
    if fix_galsim_stamp_size:
        fix_str += "-fix-stamp"
    if fix_galsim_fft_size:
        fix_str += "-fix-fft-size"
    if check_stamp_sizes:
        fix_str += "-check-sizes"
    outliers_str = ""
    if include_outliers:
        outliers_str = "-outliers"
    hash_name = (
        f"{image_slen}-{n_samples}-{psf_type}-{seed}-"
        f"hb{bin_hash}-{cpu_or_gpu}-{scan_or_vmap}{fix_str}{outliers_str}"
    )
    print(f"INFO: Running timing results and saving to folder '{hash_name}'")

    out_folder = out_root_path / hash_name
    out_folder.mkdir(parents=False, exist_ok=True)

    # prepare psf
    get_galsim_psf, get_jgs_psf, ref_galsim_psf = _prepare_psf_functions(psf_type)

    # catalog preparation and masking
    cat = prepare_catalog(catsim_fpath, min_hlr=min_hlr, max_mag=max_mag)
    good_sizes, good_fft_sizes = get_good_sizes_galsim(
        cat=cat,
        psf=ref_galsim_psf,
        overwrite=False,
        out_path=out_root_path,
        suffix=psf_type,
    )
    cat["good_size"] = good_sizes
    cat["good_fft_size"] = good_fft_sizes

    # remove galaxies that require an image size larger than the full image itself (w/ buffer)
    # we should never draw these galaxies
    _mask = cat["good_size"] + buffer <= image_slen
    cat = cat[_mask]
    print(
        "INFO: Number of galaxies with 'good size' larger than image size (with buffer) that will be excluded:",
        sum(~_mask),
    )

    # remove outliers if requested, as defined by `outlier_fraction` of the catalog (with buffer)
    if not include_outliers:
        # buffer is needed for a few pixel difference that sometimes occurs between estimating
        # good size on isolated images like in the function above and the stamp size galsim
        # ultimately uses when drawing stamp onto the big image.
        # this prevents us from EVER drawing a smaller stamp than needed in jax-galsim otherwise we
        # crash when 'check-stamp-sizes' flag is active.
        # this combined with the `max_slen` argument in the `draw_galsim` function
        # SHOULD cover all bases in terms of using a too small stamp size or missing galaxies
        _outlier_good_size = np.quantile(cat["good_size"], 1 - outlier_fraction)
        _mask = cat["good_size"] < _outlier_good_size
        cat = cat[_mask]
        print(
            f"INFO: Removing outliers of maximum good size ({outlier_fraction * 100:.2g}%): {_outlier_good_size:.0f} (pixels)"
        )
        print(f"INFO: Number of galaxies that are outliers: {sum(~_mask)}")
    else:
        print("INFO: Good size outliers are being kept.")

    # one more check if we can draw all of the galaxies in the catalog given our size bins
    assert max(stamp_slen_bins) >= max(cat["good_size"]) + buffer, (
        "Some very large galaxies will not be assigned to any bin."
    )

    # sanity check fft size bins chosen
    print(
        "INFO: Sanity checking the fft size bins that were chosen for each stamp size bin"
    )
    for ii in range(n_bins):
        _stamp_slen = stamp_slen_bins[ii]
        _fft_size = fft_size_bins[ii]
        _mask = cat["good_size"] <= _stamp_slen
        assert np.all(cat[_mask]["good_fft_size"] <= _fft_size), (
            "FFT that will be used for some galaxy in JAX-GalSim is smaller than the Galsim chosen FFT Size."
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
        _fft_size = fft_size_bins[ii]
        draw_fncs.append(
            jit(
                partial(
                    draw_fnc_raw,
                    ilen=image_slen,
                    fft_size=_fft_size,
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
            enumerate(rkeys),
            total=n_samples,
            desc="Timing GalSim vs JAX-GalSim",
            disable=not progress_bar,
        ):
            k1, k2 = random.split(rkey)

            # get psf for this image
            _psf = get_galsim_psf(k1)
            _xpsf = get_jgs_psf(k1)

            # sample in numpy
            sample, n, gsizes = get_one_full_sample(
                k2, cat=cat, ilen=image_slen, max_n_gals=max_n_gals_global
            )
            assert sample["flux_b"].shape == (n,)
            assert gsizes.shape == (n,)
            assert np.all(gsizes > 1), "There should be no dummies in this array."

            # galsim timing
            t1 = time.time()
            gs_arr = draw_galsim(
                sample,
                n,
                psf=_psf,
                ilen=image_slen,
                slen=stamp_size_galsim,
                fft_size=fft_size_galsim,
                check_stamp_sizes=check_stamp_sizes,
                max_slen=stamp_slen_bins[-1] if check_stamp_sizes else None,
                good_sizes=gsizes if check_stamp_sizes else None,
                buffer=buffer if check_stamp_sizes else None,
                size_bins=stamp_slen_bins if check_stamp_sizes else None,
            )
            t2 = time.time()
            t_galsim = t2 - t1
            times_galsim.append(t_galsim)

            # transfer to device, separate sampled parameters into bins, and time separately
            t1 = time.time()
            samples_per_bin, n_iters_per_bin = prepare_per_bin_samples(
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
            xpsf_gpu = block_until_ready(device_put(_xpsf, device=device))
            t2 = time.time()
            times_transfer.append(t2 - t1)
            assert n_bins == len(samples_per_bin) == len(n_iters_per_bin)
            del (samples_per_bin, n_iters_per_bin, _xpsf)
            # cpu versions no longer needed

            # compilation (not timed)
            if ii == 0:
                _ = block_until_ready(
                    all_draw_fnc(samples_per_bin_jax, n_iters_per_bin_jax, xpsf_gpu)
                )

            # jax galsim timing
            t1 = time.time()
            with jax.transfer_guard("disallow"):
                jgs_arr = block_until_ready(
                    all_draw_fnc(samples_per_bin_jax, n_iters_per_bin_jax, xpsf_gpu)
                )
            t2 = time.time()
            t_jgalsim = t2 - t1
            times_jgalsim.append(t_jgalsim)

            if check_stamp_sizes and verbose:
                _res = gs_arr - np.array(jgs_arr)
                if np.any(_res > POS_RESIDUAL_THRESHOLD):
                    mask = _res > POS_RESIDUAL_THRESHOLD
                    print(
                        f"WARNING: Positive residual above threshold found for image index '{ii}'. Consider taking a look at the PDF. Likely caused by a very bright galaxy if there was no assertion error. Values above threshold printed below."
                    )
                    print(_res[mask].ravel())

            if cpu_or_gpu == "cpu":
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

    print("INFO: Done running! Now saving timing results...")
    _save_timing_results(
        out_folder=out_folder,
        times_galsim=times_galsim,
        times_jgalsim=times_jgalsim,
        times_transfer=times_transfer,
    )


def _get_bins_hash(
    out_path: Path, *, stamp_slen_bins, max_n_gals_bins, fft_size_bins
) -> int:
    # very simple, just save these bins with an "id" if never has been used before
    # otherwise assign next one available
    hash_json = out_path / "bin_hash.json"
    _key = (*stamp_slen_bins, *max_n_gals_bins, *fft_size_bins)
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

    # save histogram of each in png format
    hist_times_file = out_folder / "time_histograms.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(times_galsim, bins=25, histtype="step", label="GalSim")
    ax.hist(times_jgalsim, bins=25, histtype="step", label="JAX-GalSim")
    ax.hist(times_transfer, bins=25, histtype="step", label="Transfer")
    ax.legend()
    fig.savefig(hist_times_file, dpi=500, format="png", bbox_inches="tight")

    # save timing arrays to numpy files npz format
    time_array_file = out_folder / "time_array_results.npz"
    np.savez(
        time_array_file,
        t_galsim=np.array(times_galsim),
        t_jgalsim=np.array(times_jgalsim),
        t_transfer=np.array(times_transfer),
    )


def _prepare_psf_functions(psf_type: str):
    if psf_type == "gaussian":

        def _get_galsim_psf(key):
            return galsim.Gaussian(fwhm=FWHM_PSF, flux=1.0)

        def _get_jgs_psf(key):
            return jgs.Gaussian(fwhm=FWHM_PSF, flux=1.0)

        _ref_galsim_psf = galsim.Gaussian(fwhm=FWHM_PSF, flux=1.0)

    elif psf_type == "moffat":
        # beta value from galsim tutorial 2
        def _get_galsim_psf(key):
            return galsim.Moffat(fwhm=FWHM_PSF, beta=BETA_PSF, flux=1.0)

        def _get_jgs_psf(key):
            return jgs.Moffat(fwhm=FWHM_PSF, beta=BETA_PSF, flux=1.0)

        _ref_galsim_psf = galsim.Moffat(fwhm=FWHM_PSF, beta=BETA_PSF, flux=1.0)

    elif psf_type == "vary-moffat":

        def _get_galsim_psf(key):
            _fwhm = random.uniform(
                key, minval=MIN_FWHM_PSF, maxval=MAX_FWHM_PSF, shape=()
            )
            _fwhm = _fwhm.item()
            return galsim.Moffat(fwhm=_fwhm, beta=BETA_PSF, flux=1.0)

        def _get_jgs_psf(key):
            _fwhm = random.uniform(
                key, minval=MIN_FWHM_PSF, maxval=MAX_FWHM_PSF, shape=()
            )
            _fwhm = _fwhm.item()
            return jgs.Moffat(fwhm=_fwhm, beta=BETA_PSF, flux=1.0)

        # biggest size PSF for computing good sizes
        _ref_galsim_psf = galsim.Moffat(fwhm=MAX_FWHM_PSF, beta=BETA_PSF, flux=1.0)

    else:
        raise NotImplementedError(
            "Only ('gaussian', 'moffat', or 'vary-moffat') options are supported for `psf-type`."
        )

    return _get_galsim_psf, _get_jgs_psf, _ref_galsim_psf


if __name__ == "__main__":
    typer.run(main)
