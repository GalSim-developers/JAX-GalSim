#!/usr/bin/env python3
"""GPU-profile the full scene drawing pipeline (stratified stamp size bins) via
`jax.profiler.trace`, in the same spirit as `test_roofline_xprof.py` but for the more
realistic multi-bin pipeline exercised in `timing_tests_general.py`.

Host-side sampling, `prepare_per_bin_samples`, and host->device transfer happen inside the same
`jax.profiler.trace` context as the drawing itself (one trace covers all `n_samples`), but each
phase is wrapped in a `jax.profiler.TraceAnnotation` so "transfer" and "draw" show up as
clearly separate, non-overlapping events in the trace viewer. Per-iteration device arrays are
freed right after use, so GPU memory stays bounded to ~1 sample instead of scaling with
`n_samples`.
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"

from functools import partial
from pathlib import Path

import galsim
import jax
import jax.profiler
import typer
from draw_scene_functions import (
    draw_all_bins_jgs,
    draw_jgs_scan_stamps,
    draw_jgs_vmap_scatter_stamps,
    draw_jgs_vmap_stamps,
    get_good_sizes_galsim,
    get_one_full_sample,
    prepare_catalog,
    prepare_per_bin_samples,
)
from jax import block_until_ready, device_put, jit, random

import jax_galsim as jgs

BETA_PSF = 2.5
MIN_FWHM_PSF = 0.7
MAX_FWHM_PSF = 1.0


def main(
    stamp_slen_bins_str: str = typer.Option(),
    max_n_gals_bins_str: str = typer.Option(),
    fft_size_bins_str: str = typer.Option(),
    image_slen: int = typer.Option(),
    max_n_gals_global: int = typer.Option(),
    n_samples: int = typer.Option(help="How many big images to trace."),
    catsim_fpath: str = typer.Option(),
    out_dir: str = typer.Option(),
    scan_or_vmap: str = typer.Option(),
    buffer: int = typer.Option(default=4),
    seed: int = typer.Option(default=42),
    max_n_iters: int = typer.Option(default=5),
    max_mag: float = typer.Option(default=27.0),
    min_hlr: float = typer.Option(default=0.0),
):
    stamp_slen_bins = _parse_bins_str_input(stamp_slen_bins_str)
    max_n_gals_bins = _parse_bins_str_input(max_n_gals_bins_str)
    fft_size_bins = _parse_bins_str_input(fft_size_bins_str)
    n_bins = len(stamp_slen_bins)
    assert n_bins == len(max_n_gals_bins) == len(fft_size_bins)
    assert scan_or_vmap in ("scan", "vmap", "scatter-vmap")

    out_root_path = Path(out_dir)
    out_root_path.mkdir(parents=False, exist_ok=True)
    device = jax.devices("gpu")[0]  # only works on GPU

    # vary-moffat: PSF FWHM is drawn per-sample; use the largest FWHM to size stamps/FFTs
    ref_galsim_psf = galsim.Moffat(fwhm=MAX_FWHM_PSF, beta=BETA_PSF, flux=1.0)

    def _get_jgs_psf(key):
        _fwhm = random.uniform(key, minval=MIN_FWHM_PSF, maxval=MAX_FWHM_PSF, shape=())
        return jgs.Moffat(fwhm=_fwhm.item(), beta=BETA_PSF, flux=1.0)

    assert Path(catsim_fpath).exists(), "CATSIM catalog does not exist..."
    cat = prepare_catalog(catsim_fpath, min_hlr=min_hlr, max_mag=max_mag)
    good_sizes, good_fft_sizes = get_good_sizes_galsim(
        cat=cat,
        psf=ref_galsim_psf,
        overwrite=False,
        out_path=out_root_path,
        suffix="vary-moffat",
    )
    cat["good_size"] = good_sizes
    cat["good_fft_size"] = good_fft_sizes

    _mask = cat["good_size"] + buffer <= image_slen
    cat = cat[_mask]
    assert max(stamp_slen_bins) >= max(cat["good_size"]) + buffer, (
        "Some very large galaxies will not be assigned to any bin."
    )

    draw_fnc_raw = {
        "scan": draw_jgs_scan_stamps,
        "vmap": draw_jgs_vmap_stamps,
        "scatter-vmap": draw_jgs_vmap_scatter_stamps,
    }[scan_or_vmap]

    draw_fncs = tuple(
        jit(
            partial(
                draw_fnc_raw,
                ilen=image_slen,
                fft_size=fft_size_bins[jj],
                max_n_gals=max_n_gals_bins[jj],
                slen=stamp_slen_bins[jj],
            )
        )
        for jj in range(n_bins)
    )
    all_draw_fnc = jit(
        partial(
            draw_all_bins_jgs,
            ilen=image_slen,
            draw_fncs=draw_fncs,
            device=device,
            n_bins=n_bins,
        )
    )

    rkeys = random.split(random.key(seed), n_samples)

    def _transfer_one(rkey):
        k1, k2 = random.split(rkey)
        sample, n, gsizes = get_one_full_sample(
            k2, cat=cat, ilen=image_slen, max_n_gals=max_n_gals_global
        )
        samples_per_bin, n_iters_per_bin = prepare_per_bin_samples(
            sample,
            gsizes,
            stamp_slen_bins=stamp_slen_bins,
            max_n_gals_bins=max_n_gals_bins,
            max_n_iters=max_n_iters,
            buffer=buffer,
        )
        samples_per_bin_jax = block_until_ready(
            device_put(samples_per_bin, device=device)
        )
        n_iters_per_bin_jax = block_until_ready(
            device_put(n_iters_per_bin, device=device)
        )
        xpsf_gpu = block_until_ready(device_put(_get_jgs_psf(k1), device=device))
        return samples_per_bin_jax, n_iters_per_bin_jax, xpsf_gpu

    # untraced warm-up so JIT compilation doesn't pollute the trace
    print("INFO: Running compilation...")
    _warmup = _transfer_one(rkeys[0])
    _ = block_until_ready(all_draw_fnc(*_warmup))
    del _warmup

    trace_name = f"jax-trace-scenes-{scan_or_vmap}-{seed}-{image_slen}-{n_samples}"
    trace_dir = out_root_path / trace_name
    print(f"INFO: Tracing {n_samples} sample(s) to '{trace_dir}'...")
    with jax.profiler.trace(trace_dir):
        for rkey in rkeys:
            with jax.profiler.TraceAnnotation("transfer"):
                samples_per_bin_jax, n_iters_per_bin_jax, xpsf_gpu = _transfer_one(rkey)

            with jax.profiler.TraceAnnotation("draw"):
                with jax.transfer_guard("disallow"):
                    _ = block_until_ready(
                        all_draw_fnc(samples_per_bin_jax, n_iters_per_bin_jax, xpsf_gpu)
                    )

            # free this iteration's device buffers before the next iteration transfers new ones
            del samples_per_bin_jax, n_iters_per_bin_jax, xpsf_gpu
    print("INFO: Done tracing.")


def _parse_bins_str_input(bins_str: str):
    if "," not in bins_str:
        return (int(bins_str),)
    the_bins = [float(x) for x in bins_str.split(",")]
    for x in the_bins:
        assert int(x) == x
    return tuple(int(x) for x in the_bins)


if __name__ == "__main__":
    typer.run(main)
