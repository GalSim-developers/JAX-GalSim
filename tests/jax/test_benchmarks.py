from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim as jgs
from jax_galsim.core.testing import time_code_block


def _run_benchmarks(benchmark, kind, func):
    if kind == "compile":

        def _run():
            jax.clear_caches()
            func()

    elif kind == "run":
        # run once to compile
        func()

        def _run():
            func()

    else:
        raise ValueError(f"kind={kind} not recognized")

    with time_code_block(quiet=True) as tr:
        benchmark(_run)

    return tr.dt


@pytest.mark.parametrize("kind", ["compile", "run"])
@pytest.mark.parametrize(
    "conserve_dc", [True, False], ids=["conserve_dc", "no_conserve_dc"]
)
@pytest.mark.parametrize("method", ["xval", "kval"])
def test_benchmarks_lanczos_interp(benchmark, kind, conserve_dc, method):
    interp = jgs.Lanczos(5, conserve_dc=conserve_dc)
    if method == "xval":
        f = jax.jit(interp.xval)
    elif method == "kval":
        f = jax.jit(interp.kval)
    else:
        raise ValueError(f"method={method} not recognized")

    k = jnp.linspace(0.3, 0.5, 10000)

    dt = _run_benchmarks(benchmark, kind, lambda: f(k).block_until_ready())
    print(f"time: {dt:0.4g} ms", end=" ")


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

    dt = _run_benchmarks(benchmark, kind, lambda: jitf().array.block_until_ready())
    print(f"time: {dt:0.4g} ms", end=" ")


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


@pytest.mark.parametrize("kind", ["run"])
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

    dt = _run_benchmarks(benchmark, kind, _run)
    print(f"time: {dt:0.4g} ms", end=" ")


def _run_spergel_bench_conv(gsmod):
    obj = gsmod.Spergel(nu=-0.6, scale_radius=5)
    psf = gsmod.Gaussian(fwhm=0.9)
    obj = gsmod.Convolve(
        [obj, psf],
        gsparams=gsmod.GSParams(minimum_fft_size=2048, maximum_fft_size=2048),
    )
    return obj.drawImage(nx=50, ny=50, scale=0.2).array


_run_spergel_bench_conv_jit = jax.jit(partial(_run_spergel_bench_conv, jgs))


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_spergel_conv(benchmark, kind):
    dt = _run_benchmarks(
        benchmark, kind, lambda: _run_spergel_bench_conv_jit().block_until_ready()
    )
    print(f"time: {dt:0.4g} ms", end=" ")


def _run_spergel_bench_xvalue(gsmod):
    obj = gsmod.Spergel(nu=-0.6, scale_radius=5)
    return obj.drawImage(nx=1024, ny=1204, scale=0.05, method="no_pixel").array


_run_spergel_bench_xvalue_jit = jax.jit(partial(_run_spergel_bench_xvalue, jgs))


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_spergel_xvalue(benchmark, kind):
    dt = _run_benchmarks(
        benchmark, kind, lambda: _run_spergel_bench_xvalue_jit().block_until_ready()
    )
    print(f"time: {dt:0.4g} ms", end=" ")


def _run_spergel_bench_kvalue(gsmod):
    obj = gsmod.Spergel(nu=-0.6, scale_radius=5)
    return obj.drawKImage(nx=1024, ny=1204, scale=0.05).array


_run_spergel_bench_kvalue_jit = jax.jit(partial(_run_spergel_bench_kvalue, jgs))


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_spergel_kvalue(benchmark, kind):
    dt = _run_benchmarks(
        benchmark, kind, lambda: _run_spergel_bench_kvalue_jit().block_until_ready()
    )
    print(f"time: {dt:0.4g} ms", end=" ")


@jax.jit
def _run_spergel_bench_init():
    return jgs.Spergel(nu=-0.6, half_light_radius=3.4).scale_radius


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_spergel_init(benchmark, kind):
    dt = _run_benchmarks(
        benchmark, kind, lambda: _run_spergel_bench_init().block_until_ready()
    )
    print(f"time: {dt:0.4g} ms", end=" ")


@jax.jit
def _run_gaussian_bench_init():
    return jgs.Gaussian(half_light_radius=3.4).sigma


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_gaussian_init(benchmark, kind):
    dt = _run_benchmarks(
        benchmark, kind, lambda: _run_gaussian_bench_init().block_until_ready()
    )
    print(f"time: {dt:0.4g} ms", end=" ")


def _run_benchmark_interpimage_flux_frac(img):
    x, y = img.get_pixel_centers()
    cenx = img.center.x
    ceny = img.center.y
    return jgs.interpolatedimage._flux_frac(img.array, x, y, cenx, ceny)


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_interpimage_flux_frac(benchmark, kind):
    obj = jgs.Gaussian(half_light_radius=0.9).shear(g1=0.1, g2=0.2)
    img = obj.drawImage(nx=55, ny=55, scale=0.2, method="no_pixel")
    dt = _run_benchmarks(
        benchmark,
        kind,
        lambda: _run_benchmark_interpimage_flux_frac(img).block_until_ready(),
    )
    print(f"time: {dt:0.4g} ms", end=" ")


@jax.jit
def _run_benchmark_rng_discard(rng):
    rng.discard(1000)
    return rng._state.key


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_rng_discard(benchmark, kind):
    rng = jgs.BaseDeviate(seed=42)
    dt = _run_benchmarks(
        benchmark, kind, lambda: _run_benchmark_rng_discard(rng).block_until_ready()
    )
    print(f"time: {dt:0.4g} ms", end=" ")


def _run_benchmark_invert_ab_noraise(u, v, ab):
    return jgs.fitswcs._invert_ab_noraise(u, v, ab)[0]


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_invert_ab_noraise(benchmark, kind):
    u = jnp.arange(1000).astype(jnp.float64)
    v = jnp.arange(1000).astype(jnp.float64)
    ab = jnp.array([[[-0.5, 0.3], [-0.1, 2.0]], [[-1.0, 0.3], [-0.1, 4.0]]])
    dt = _run_benchmarks(
        benchmark,
        kind,
        lambda: _run_benchmark_invert_ab_noraise(u, v, ab).block_until_ready(),
    )
    print(f"time: {dt:0.4g} ms", end=" ")


def _run_benchmark_moffat_init():
    return jgs.Moffat(beta=2.5, half_light_radius=0.6, trunc=1.2).scale_radius


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_moffat_init(benchmark, kind):
    dt = _run_benchmarks(
        benchmark, kind, lambda: _run_benchmark_moffat_init().block_until_ready()
    )
    print(f"time: {dt:0.4g} ms", end=" ")


def _run_benchmark_spergel_calcfluxrad():
    return jgs.spergel.calculateFluxRadius(1e-10, 2.0)


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_benchmark_spergel_calcfluxrad(benchmark, kind):
    dt = _run_benchmarks(
        benchmark,
        kind,
        lambda: _run_benchmark_spergel_calcfluxrad().block_until_ready(),
    )
    print(f"time: {dt:0.4g} ms", end=" ")
