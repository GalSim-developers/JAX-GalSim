#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import time
from functools import partial

import galsim as _galsim
import jax.random as jrng
import numpy as np
from jax import block_until_ready, jit, transfer_guard, vmap

import jax_galsim as jgs
from jax_galsim.photon_array import fixed_photon_array_size


def main():
    rng_key = jrng.key(42)
    n_obj = 1000

    psf = _galsim.Gaussian(half_light_radius=0.7, flux=1.0)
    xpsf = jgs.Gaussian(half_light_radius=0.7, flux=1.0)

    # prepare jax_galsim draw function
    _draw_fnc = jit(partial(_generate_image_phot_jgs, psf=xpsf, n_obj=n_obj))

    # time galsim
    t1 = time.time()
    _ = _generate_image_phot_galsim(rng_key, psf=psf, n_obj=n_obj)
    t2 = time.time()
    t_galsim = t2 - t1

    # time jax-galsim
    # compilation
    _ = block_until_ready(_draw_fnc(rng_key))

    # timing
    t1 = time.time()
    with transfer_guard("disallow"):
        _ = block_until_ready(_draw_fnc(rng_key))
    t2 = time.time()
    t_jgs = t2 - t1

    print(f"Time GalSim: {t_galsim:.4f}")
    print(f"Time JAX-GalSim: {t_jgs:.4f}")


def _generate_random_params(rng_key):
    rng_key, use_key = jrng.split(rng_key)
    flux = jrng.uniform(use_key, minval=10000.5, maxval=20000.5)
    rng_key, use_key = jrng.split(rng_key)
    hlr = jrng.uniform(use_key, minval=0.5, maxval=2.5)
    rng_key, use_key = jrng.split(rng_key)
    g1 = jrng.uniform(use_key, minval=-0.1, maxval=0.1)
    rng_key, use_key = jrng.split(rng_key)
    g2 = jrng.uniform(use_key, minval=-0.1, maxval=0.1)

    rng_key, use_key = jrng.split(rng_key)
    dx = jrng.uniform(use_key, minval=-10, maxval=10)
    rng_key, use_key = jrng.split(rng_key)
    dy = jrng.uniform(use_key, minval=-10, maxval=10)
    rng_key, use_key = jrng.split(rng_key)

    return (
        flux,
        hlr,
        g1,
        g2,
        dx,
        dy,
    ), use_key


def _generate_image_one_phot_jgs(rng_key, psf):
    (flux, hlr, g1, g2, dx, dy), use_key = _generate_random_params(rng_key)

    return (
        jgs.Convolve(
            [
                jgs.Exponential(half_light_radius=hlr)
                .shear(g1=g1, g2=g2)
                .shift(dx, dy)
                .withFlux(flux),
                psf,
            ]
        )
        .withGSParams(minimum_fft_size=1024, maximum_fft_size=1024)
        .drawImage(
            nx=200, ny=200, scale=0.2, method="phot", rng=jgs.BaseDeviate(use_key)
        )
    )


def _generate_image_phot_jgs(rng_key, psf, n_obj):
    use_keys = jrng.split(rng_key, num=n_obj + 1)
    rng_key = use_keys[0]
    use_keys = use_keys[1:]

    with fixed_photon_array_size(1000):
        images = vmap(_generate_image_one_phot_jgs, in_axes=(0, None))(use_keys, psf)
    assert images.array.shape[0] == n_obj
    return images


def _generate_image_phot_galsim(rng_key, psf, n_obj):
    use_keys = jrng.split(rng_key, num=n_obj + 1)
    rng_key = use_keys[0]
    use_keys = use_keys[1:]

    all_out = []
    for _ in range(n_obj):
        (flux, hlr, g1, g2, dx, dy), _ = _generate_random_params(rng_key)
        flux = flux.item()
        hlr = hlr.item()
        g1 = g1.item()
        g2 = g2.item()
        dx = dx.item()
        dy = dy.item()
        img = (
            _galsim.Convolve(
                [
                    _galsim.Exponential(half_light_radius=hlr)
                    .shear(g1=g1, g2=g2)
                    .shift(dx, dy)
                    .withFlux(flux),
                    psf,
                ]
            )
            .drawImage(nx=200, ny=200, scale=0.2, method="phot")
            .array
        )
        all_out.append(img)
    images = np.stack(all_out, axis=0)
    assert images.shape[0] == n_obj
    return images


if __name__ == "__main__":
    main()
