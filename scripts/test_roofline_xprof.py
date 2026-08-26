import os

os.environ["JAX_ENABLE_X64"] = "True"

from functools import partial
from pathlib import Path

import galsim
import jax
import typer
from draw_scene_functions import (
    draw_jgs_vmap_stamps,
    get_good_sizes_galsim,
    get_one_full_sample,
    prepare_catalog,
)
from jax import block_until_ready, device_put, jit

import jax_galsim as jgs


def main(
    seed: int = typer.Option(default=42),
    image_slen: int = typer.Option(),
    max_n_gals_global: int = typer.Option(),
    stamp_slen: int = typer.Option(default=61),
    fft_size: int = typer.Option(default=128),
    out_dir: str = ".",
):
    # let's just measure utilization on simplest case with one small size stamp bin??
    k = jax.random.key(seed)
    device = jax.devices("gpu")[0]

    psf = galsim.Moffat(beta=2.5, fwhm=0.8, flux=1.0)
    xpsf = jgs.Moffat(beta=2.5, fwhm=0.8, flux=1.0)

    # prepare catalog
    cat = prepare_catalog("../OneDegSq.fits", min_hlr=0, max_mag=27)
    good_sizes, good_fft_sizes = get_good_sizes_galsim(
        cat=cat,
        psf=psf,
        overwrite=False,
        out_path=Path("scripts/output_roofline"),
        suffix="moffat",
    )
    cat["good_size"] = good_sizes

    # cut all galaxies below SLEN
    mask = (good_sizes <= stamp_slen) & (good_fft_sizes <= fft_size)
    cat = cat[mask]
    print(f"INFO: Number of galaxies in catalog is {len(cat)}")

    sample, n, _ = get_one_full_sample(
        k, cat=cat, ilen=image_slen, max_n_gals=max_n_gals_global
    )

    sample_jax = block_until_ready(device_put(sample, device=device))
    xpsf_gpu = block_until_ready(device_put(xpsf, device=device))

    draw_func = jit(
        partial(
            draw_jgs_vmap_stamps,
            ilen=image_slen,
            slen=stamp_slen,
            fft_size=fft_size,
            max_n_gals=n,
        )
    )

    # compilation
    _ = block_until_ready(draw_func(sample_jax, xpsf_gpu))

    # roofline plot
    with jax.profiler.trace(Path(out_dir) / "jax-trace-{seed}-{image_slen}"):
        with jax.transfer_guard("disallow"):
            _ = block_until_ready(draw_func(sample_jax, xpsf_gpu))


if __name__ == "__main__":
    main()
