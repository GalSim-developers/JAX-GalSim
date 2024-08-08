import time
from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim


def _metacal_galsim(
    im,
    psf,
    nse_im,
    scale,
    target_fwhm,
    g1,
    iim_kwargs,
    ipsf_kwargs,
    inse_kwargs,
    nk,
):
    iim = _galsim.InterpolatedImage(
        _galsim.ImageD(im),
        scale=scale,
        x_interpolant="lanczos15",
        **iim_kwargs,
    )
    ipsf = _galsim.InterpolatedImage(
        _galsim.ImageD(psf),
        scale=scale,
        x_interpolant="lanczos15",
        **ipsf_kwargs,
    )
    inse = _galsim.InterpolatedImage(
        _galsim.ImageD(np.rot90(nse_im, 1)),
        scale=scale,
        x_interpolant="lanczos15",
        **inse_kwargs,
    )

    ppsf_iim = _galsim.Convolve(iim, _galsim.Deconvolve(ipsf))
    ppsf_iim = ppsf_iim.shear(g1=g1, g2=0.0)

    prof = _galsim.Convolve(
        ppsf_iim,
        _galsim.Gaussian(fwhm=target_fwhm),
        gsparams=_galsim.GSParams(minimum_fft_size=nk, maximum_fft_size=nk),
    )

    sim = prof.drawImage(
        nx=33,
        ny=33,
        scale=scale,
        method="no_pixel",
    ).array.astype(np.float64)

    ppsf_inse = _galsim.Convolve(inse, _galsim.Deconvolve(ipsf))
    ppsf_inse = ppsf_inse.shear(g1=g1, g2=0.0)
    snse = (
        _galsim.Convolve(
            ppsf_inse,
            _galsim.Gaussian(fwhm=target_fwhm),
            gsparams=_galsim.GSParams(minimum_fft_size=nk, maximum_fft_size=nk),
        )
        .drawImage(
            nx=33,
            ny=33,
            scale=scale,
            method="no_pixel",
        )
        .array.astype(np.float64)
    )
    return sim + np.rot90(snse, 3)


@partial(jax.jit, static_argnames=("nk",))
def _metacal_jax_galsim_render(im, psf, g1, target_psf, scale, nk):
    prepsf_im = jax_galsim.Convolve(im, jax_galsim.Deconvolve(psf))
    prepsf_im = prepsf_im.shear(g1=g1, g2=0.0)

    prof = jax_galsim.Convolve(
        prepsf_im,
        target_psf,
        gsparams=jax_galsim.GSParams(minimum_fft_size=nk, maximum_fft_size=nk),
    )

    return prof.drawImage(
        nx=33,
        ny=33,
        scale=scale,
        method="no_pixel",
    ).array.astype(np.float64)


@partial(jax.jit, static_argnames=("nk",))
def _metacal_jax_galsim(im, psf, nse_im, scale, target_fwhm, g1, nk):
    iim = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(im), scale=scale, x_interpolant="lanczos15"
    )
    ipsf = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(psf), scale=scale, x_interpolant="lanczos15"
    )
    inse = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(jnp.rot90(nse_im, 1)), scale=scale, x_interpolant="lanczos15"
    )

    target_psf = jax_galsim.Gaussian(fwhm=target_fwhm)

    sim = _metacal_jax_galsim_render(iim, ipsf, g1, target_psf, scale, nk)

    snse = _metacal_jax_galsim_render(inse, ipsf, g1, target_psf, scale, nk)

    return sim + jnp.rot90(snse, 3)


@pytest.mark.parametrize("nse", [1e-3, 1e-10])
def test_metacal_comp_to_galsim(nse):
    seed = 42
    hlr = 0.5
    fwhm = 0.9
    scale = 0.2
    g1 = 0.01
    target_fwhm = 1.0

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

    # jax galsim and galsim set stepk and maxk differently due to slight
    # algorithmic differences.  We force them to be the same here for this
    # test so it passes.
    iim = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(im),
        scale=scale,
        x_interpolant="lanczos15",
        gsparams=jax_galsim.GSParams(minimum_fft_size=128),
    )
    iim_kwargs = {
        "_force_stepk": iim.stepk.item(),
        "_force_maxk": iim.maxk.item(),
    }
    inse = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(jnp.rot90(nse_im, 1)),
        scale=scale,
        x_interpolant="lanczos15",
        gsparams=jax_galsim.GSParams(minimum_fft_size=128),
    )
    inse_kwargs = {
        "_force_stepk": inse.stepk.item(),
        "_force_maxk": inse.maxk.item(),
    }
    ipsf = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(psf), scale=scale, x_interpolant="lanczos15"
    )
    ipsf_kwargs = {
        "_force_stepk": ipsf.stepk.item(),
        "_force_maxk": ipsf.maxk.item(),
    }

    gt0 = time.time()
    gres = _metacal_galsim(
        im.copy(),
        psf.copy(),
        nse_im.copy(),
        scale,
        target_fwhm,
        g1,
        iim_kwargs,
        ipsf_kwargs,
        inse_kwargs,
        128,
    )
    gt0 = time.time() - gt0

    print("galsim time: ", gt0 * 1e3, " [ms]")

    for i in range(2):
        if i == 0:
            msg = "jit warmup"
        elif i == 1:
            msg = "jit"
        jgt0 = time.time()
        jgres = _metacal_jax_galsim(
            im.copy(),
            psf.copy(),
            nse_im.copy(),
            scale,
            target_fwhm,
            g1,
            128,
        )
        jgres = jax.block_until_ready(jgres)
        jgt0 = time.time() - jgt0
        print("jax-galsim time (%s): " % msg, jgt0 * 1e3, " [ms]")

    if gres is None or jgres is None:
        return

    gim = gres
    jgim = jgres

    atol = 1e-8
    if not np.allclose(gim, jgim, rtol=0, atol=atol):
        import proplot as pplt

        fig, axs = pplt.subplots(ncols=3, nrows=1)

        axs[0].imshow(np.arcsinh(gres / nse))
        axs[1].imshow(np.arcsinh(jgres / nse))
        m = axs[2].imshow(jgres - gres)
        axs[2].colorbar(m, loc="r")

        fig.show()

    np.testing.assert_allclose(gim, jgim, rtol=0, atol=atol)


@pytest.mark.parametrize("ntest", [1, 10, 100])
def test_metacal_vmap(ntest):
    start_seed = 42
    hlr = 0.5
    fwhm = 0.9
    scale = 0.2
    nse = 1e-3
    g1 = 0.01
    target_fwhm = 1.0

    ims = []
    nse_ims = []
    psfs = []
    init_done = False
    for _seed in range(ntest):
        seed = _seed + start_seed
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

        nse_im = rng.normal(size=im.shape) * nse
        im += rng.normal(size=im.shape) * nse

        ims.append(im)
        psfs.append(psf)
        nse_ims.append(nse_im)

        if not init_done:
            init_done = True

            # jax galsim and galsim set stepk and maxk differently due to slight
            # algorithmic differences.  We force them to be the same here for this
            # test so it passes.
            iim = jax_galsim.InterpolatedImage(
                jax_galsim.ImageD(im),
                scale=scale,
                x_interpolant="lanczos15",
                gsparams=jax_galsim.GSParams(minimum_fft_size=128),
            )
            iim_kwargs = {
                "_force_stepk": iim.stepk.item(),
                "_force_maxk": iim.maxk.item(),
            }
            inse = jax_galsim.InterpolatedImage(
                jax_galsim.ImageD(jnp.rot90(nse_im, 1)),
                scale=scale,
                x_interpolant="lanczos15",
                gsparams=jax_galsim.GSParams(minimum_fft_size=128),
            )
            inse_kwargs = {
                "_force_stepk": inse.stepk.item(),
                "_force_maxk": inse.maxk.item(),
            }
            ipsf = jax_galsim.InterpolatedImage(
                jax_galsim.ImageD(psf), scale=scale, x_interpolant="lanczos15"
            )
            ipsf_kwargs = {
                "_force_stepk": ipsf.stepk.item(),
                "_force_maxk": ipsf.maxk.item(),
            }

    ims = np.stack(ims)
    psfs = np.stack(psfs)
    nse_ims = np.stack(nse_ims)

    gt0 = time.time()
    for im, psf, nse_im in zip(ims, psfs, nse_ims):
        _metacal_galsim(
            im.copy(),
            psf.copy(),
            nse_im.copy(),
            scale,
            target_fwhm,
            g1,
            iim_kwargs,
            ipsf_kwargs,
            inse_kwargs,
            128,
        )
    gt0 = time.time() - gt0
    print("Galsim time: ", gt0 * 1e3, " [ms]")

    vmap_mcal = jax.jit(
        jax.vmap(
            _metacal_jax_galsim,
            in_axes=(0, 0, 0, None, None, None, None),
        ),
        static_argnames=("nk",),
    )

    for i in range(2):
        if i == 0:
            msg = "jit warmup"
        elif i == 1:
            msg = "jit"

        jgt0 = time.time()
        vmap_mcal(
            ims,
            psfs,
            nse_ims,
            scale,
            target_fwhm,
            g1,
            128,
        )
        jgt0 = time.time() - jgt0
        print("Jax-Galsim time (%s): " % msg, jgt0 * 1e3, " [ms]")


@pytest.mark.parametrize(
    "draw_method",
    [
        "no_pixel",
        "auto",
    ],
)
@pytest.mark.parametrize(
    "nse",
    [
        4e-3,
        1e-3,
        1e-10,
    ],
)
def test_metacal_iimage_with_noise(nse, draw_method):
    hlr = 0.5
    fwhm = 0.9
    scale = 0.2
    nk = 128
    seed = 42

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
    im += rng.normal(size=im.shape) * nse

    jgiim = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(im),
        scale=scale,
        x_interpolant="lanczos15",
        gsparams=jax_galsim.GSParams(minimum_fft_size=nk),
    )

    giim = _galsim.InterpolatedImage(
        _galsim.ImageD(im),
        scale=scale,
        x_interpolant="lanczos15",
        gsparams=_galsim.GSParams(minimum_fft_size=nk),
        _force_stepk=jgiim.stepk.item(),
        _force_maxk=jgiim.maxk.item(),
    )

    def _plot_real(gim, jgim):
        import proplot as pplt

        fig, axs = pplt.subplots(ncols=3, nrows=1)

        axs[0].imshow(gim)
        axs[1].imshow(jgim)
        m = axs[0, 2].imshow((jgim - gim))
        axs[2].colorbar(m, loc="r")

        fig.show()

    atol = 1e-8
    np.testing.assert_allclose(giim.maxk, jgiim.maxk)
    np.testing.assert_allclose(giim.maxk, jgiim.maxk)

    if draw_method == "no_pixel":
        gim = giim.drawImage(nx=33, ny=33, scale=scale, method="no_pixel").array
        jgim = jgiim.drawImage(nx=33, ny=33, scale=scale, method="no_pixel").array

        if not np.allclose(gim, jgim, rtol=0, atol=atol):
            _plot_real(gim, jgim)
        np.testing.assert_allclose(gim, jgim, rtol=0, atol=atol)
    elif draw_method == "auto":
        gim = giim.drawImage(nx=33, ny=33, scale=scale).array
        jgim = jgiim.drawImage(nx=33, ny=33, scale=scale).array

        if not np.allclose(gim, jgim, rtol=0, atol=atol):
            _plot_real(gim, jgim)
        np.testing.assert_allclose(gim, jgim, rtol=0, atol=atol)
