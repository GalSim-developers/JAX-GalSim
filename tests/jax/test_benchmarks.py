import time
from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim as jgs


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmarks_interpolated_image(benchmark, kind):
    gal = jgs.Gaussian(fwhm=1.2)
    im_gal = gal.drawImage(nx=32, ny=32, scale=0.2)
    igal = jgs.InterpolatedImage(
        im_gal, gsparams=jgs.GSParams(minimum_fft_size=128, maximum_fft_size=128)
    )

    def f():
        return igal.drawImage(nx=32, ny=32, scale=0.2)

    jitf = jax.jit(f)

    if kind == "compile":
        jax.clear_caches()
        t0 = time.time()
        benchmark(lambda: jitf().array.block_until_ready())
        t0 = time.time() - t0
    elif kind == "run":
        jitf().array.block_until_ready()
        t0 = time.time()
        benchmark(lambda: jitf().array.block_until_ready())
        t0 = time.time() - t0
    else:
        raise ValueError(f"kind={kind} not recognized")

    print(f"time: {t0 / 1e-3:0.4g} ms", end=" ")


@partial(jax.jit, static_argnames=("nk",))
def _metacal_jax_galsim_render(im, psf, g1, target_psf, scale, nk):
    prepsf_im = jgs.Convolve(im, jgs.Deconvolve(psf))
    prepsf_im = prepsf_im.shear(g1=g1, g2=0.0)

    prof = jgs.Convolve(
        prepsf_im,
        target_psf,
        gsparams=jgs.GSParams(minimum_fft_size=nk, maximum_fft_size=nk),
    )

    return prof.drawImage(
        nx=33,
        ny=33,
        scale=scale,
        method="no_pixel",
    ).array.astype(np.float64)


@partial(jax.jit, static_argnames=("nk",))
def _metacal_jax_galsim(im, psf, nse_im, scale, target_fwhm, g1, nk):
    iim = jgs.InterpolatedImage(jgs.ImageD(im), scale=scale, x_interpolant="lanczos15")
    ipsf = jgs.InterpolatedImage(
        jgs.ImageD(psf), scale=scale, x_interpolant="lanczos15"
    )
    inse = jgs.InterpolatedImage(
        jgs.ImageD(jnp.rot90(nse_im, 1)), scale=scale, x_interpolant="lanczos15"
    )

    target_psf = jgs.Gaussian(fwhm=target_fwhm)

    sim = _metacal_jax_galsim_render(iim, ipsf, g1, target_psf, scale, nk)

    snse = _metacal_jax_galsim_render(inse, ipsf, g1, target_psf, scale, nk)

    return sim + jnp.rot90(snse, 3)


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmarks_metacal(benchmark, kind):
    seed = 42
    hlr = 0.5
    fwhm = 0.9
    scale = 0.2
    g1 = 0.01
    target_fwhm = 1.0
    nse = 0.001

    rng = np.random.RandomState(seed)

    im = (
        _galsim.Convolve(
            _galsim.Exponential(half_light_radius=hlr),
            _galsim.Gaussian(fwhm=fwhm),
        )
        .drawImage(
            nx=33,
            ny=33,
            scale=scale,
        )
        .array.astype(np.float64)
    )

    psf = (
        _galsim.Gaussian(fwhm=fwhm)
        .drawImage(
            nx=33,
            ny=33,
            scale=scale,
        )
        .array.astype(np.float64)
    )

    nse_im = rng.normal(size=im.shape, scale=nse)
    im += rng.normal(size=im.shape, scale=nse)

    def _run():
        return _metacal_jax_galsim(
            im.copy(),
            psf.copy(),
            nse_im.copy(),
            scale,
            target_fwhm,
            g1,
            128,
        ).block_until_ready()

    if kind == "compile":
        jax.clear_caches()
        t0 = time.time()
        benchmark(lambda: _run())
        t0 = time.time() - t0
    elif kind == "run":
        t0 = time.time()
        _run()
        benchmark(lambda: _run())
        t0 = time.time() - t0
    else:
        raise ValueError(f"kind={kind} not recognized")

    print(f"time: {t0 / 1e-3:0.4g} ms", end=" ")
