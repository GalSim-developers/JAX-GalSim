#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import time
from functools import partial

import galsim as _galsim
import jax
import jax.random as jrng
import numpy as np
import typer
from jax import block_until_ready, device_put, jit, transfer_guard, vmap

import jax_galsim as jgs
from jax_galsim.photon_array import fixed_photon_array_size


def main(
    cpu_or_gpu: str = typer.Option(default="cpu"),
):

    if cpu_or_gpu == "cpu":
        device = jax.devices("cpu")[0]
    elif cpu_or_gpu == "gpu":
        device = jax.devices("gpu")[0]
    else:
        raise ValueError()

    rng_key = jrng.key(42)
    n_obj = 1000
    k1, k2 = jrng.split(rng_key)
    pkeys = jrng.split(k1, n_obj)
    skeys = jrng.split(k2, n_obj)

    psf = _galsim.Gaussian(half_light_radius=0.7, flux=1.0)
    xpsf = jgs.Gaussian(half_light_radius=0.7, flux=1.0)

    # prepare jax_galsim draw function
    _draw_fnc = jit(partial(_generate_image_phot_jgs, psf=xpsf, n_obj=n_obj))

    # get all parameters for both galsim and jgs
    params = vmap(_generate_random_params)(pkeys)
    params = [np.array(p) for p in params]

    # time galsim
    t1 = time.time()
    _ = _generate_image_phot_galsim(params, psf=psf, n_obj=n_obj)
    t2 = time.time()
    t_galsim = t2 - t1

    # time jax-galsim
    # compilation and transfer
    params_jax = device_put(params, device=device)
    _ = block_until_ready(_draw_fnc(skeys, params_jax))

    # timing
    t1 = time.time()
    with transfer_guard("disallow"):
        _ = block_until_ready(_draw_fnc(skeys, params_jax))
    t2 = time.time()
    t_jgs = t2 - t1

    print(f"Time GalSim (seconds): {t_galsim:.4f}")
    print(f"Time JAX-GalSim (seconds): {t_jgs:.4f}")


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
    return flux, hlr, g1, g2, dx, dy


def _generate_image_phot_jgs(rng_keys, params, psf, n_obj):
    with fixed_photon_array_size(1000):
        images = vmap(_generate_image_one_phot_jgs, in_axes=(0, 0, None))(
            rng_keys, params, psf
        )
    assert images.array.shape[0] == n_obj
    return images


def _generate_image_one_phot_jgs(rng_key, params, psf):
    flux, hlr, g1, g2, dx, dy = params

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
            nx=200, ny=200, scale=0.2, method="phot", rng=jgs.BaseDeviate(rng_key)
        )
    )


def _generate_image_phot_galsim(params, psf, n_obj):

    all_out = []
    for ii in range(n_obj):
        (flux, hlr, g1, g2, dx, dy) = [p[ii] for p in params]
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
    typer.run(main)
