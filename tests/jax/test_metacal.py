import time
from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np

import jax_galsim


def _metacal_galsim(im, psf, nse_im, scale, target_fwhm, g1, nk):
    iim = _galsim.InterpolatedImage(
        _galsim.ImageD(im), scale=scale, x_interpolant="lanczos15"
    )
    ipsf = _galsim.InterpolatedImage(
        _galsim.ImageD(psf), scale=scale, x_interpolant="lanczos15"
    )
    inse = _galsim.InterpolatedImage(
        _galsim.ImageD(np.rot90(nse_im)), scale=scale, x_interpolant="lanczos15"
    )

    ppsf_iim = _galsim.Convolve(iim, _galsim.Deconvolve(ipsf))
    ppsf_iim = ppsf_iim.shear(g1=g1, g2=0.0)

    sim = (
        _galsim.Convolve(
            ppsf_iim,
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

    return (
        jax_galsim.Convolve(
            prepsf_im,
            target_psf,
            gsparams=jax_galsim.GSParams(minimum_fft_size=nk, maximum_fft_size=nk),
        )
        .drawImage(
            nx=33,
            ny=33,
            scale=scale,
            method="no_pixel",
        )
        .array.astype(np.float64)
    )


def _metacal_jax_galsim(im, psf, nse_im, scale, target_fwhm, g1, nk):
    iim = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(im), scale=scale, x_interpolant="lanczos15"
    )
    ipsf = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(psf), scale=scale, x_interpolant="lanczos15"
    )
    inse = jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(jnp.rot90(nse_im)), scale=scale, x_interpolant="lanczos15"
    )
    target_psf = jax_galsim.Gaussian(fwhm=target_fwhm)

    sim = _metacal_jax_galsim_render(iim, ipsf, g1, target_psf, scale, nk)

    snse = _metacal_jax_galsim_render(inse, ipsf, g1, target_psf, scale, nk)

    return sim + jnp.rot90(snse, 3)




def test_metacal_comp_to_galsim():
    seed = 42
    hlr = 0.5
    fwhm = 0.9
    scale = 0.2
    nse = 1e-3
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

    nse_im = rng.normal(size=im.shape) * nse
    im += rng.normal(size=im.shape) * nse

    gt0 = time.time()
    gres = _metacal_galsim(
        im.copy(), psf.copy(), nse_im.copy(), scale, target_fwhm, g1, 128
    )
    gt0 = time.time() - gt0

    print("Galsim time: ", gt0 * 1e3, " [ms]")

    _func = jax.jit(_metacal_jax_galsim, static_argnames=("nk",))

    for i in range(2):
        if i == 0:
            msg = "jit warmup"
        elif i == 1:
            msg = "jit"
        jgt0 = time.time()
        jgres = _func(
            im,
            psf,
            nse_im,
            scale,
            target_fwhm,
            g1,
            128,
        )
        jgres = jax.block_until_ready(jgres)
        jgt0 = time.time() - jgt0
        print("Jax-Galsim time (%s): " % msg, jgt0 * 1e3, " [ms]")

    gim = gres
    jgim = jgres

    atol = 7e-5

    if not np.allclose(gim, jgim, rtol=0, atol=atol):
        import proplot as pplt

        fig, axs = pplt.subplots(ncols=3, nrows=3)
        _gim = gres
        _jgim = jgres

        gpsf = (
            _galsim.InterpolatedImage(
                _galsim.Image(psf, scale=scale), x_interpolant="lanczos15"
            )
            .drawImage(
                nx=33,
                ny=33,
                scale=scale,
            )
            .array
        )
        jgpsf = (
            jax_galsim.InterpolatedImage(
                jax_galsim.Image(psf, scale=scale), x_interpolant="lanczos15"
            )
            .drawImage(
                nx=33,
                ny=33,
                scale=scale,
            )
            .array
        )

        axs[0, 0].imshow(gpsf)
        axs[0, 1].imshow(jgpsf)
        m = axs[0, 2].imshow((jgpsf - gpsf) / 1e-5)
        axs[0, 2].colorbar(m, loc="r")

        gpsf = (
            _galsim.InterpolatedImage(
                _galsim.Image(psf, scale=scale), x_interpolant="lanczos15"
            )
            .drawImage(
                nx=33,
                ny=33,
                scale=scale,
                method="no_pixel",
            )
            .array
        )
        jgpsf = (
            jax_galsim.InterpolatedImage(
                jax_galsim.Image(psf, scale=scale), x_interpolant="lanczos15"
            )
            .drawImage(
                nx=33,
                ny=33,
                scale=scale,
                method="no_pixel",
            )
            .array
        )

        axs[1, 0].imshow(gpsf)
        axs[1, 1].imshow(jgpsf)
        m = axs[1, 2].imshow((jgpsf - gpsf) / 1e-5)
        axs[1, 2].colorbar(m, loc="r")

        axs[2, 0].imshow(np.arcsinh(_gim / nse))
        axs[2, 1].imshow(np.arcsinh(_jgim / nse))
        m = axs[2, 2].imshow((_jgim - _gim) / 1e-5)
        axs[2, 2].colorbar(m, loc="r")

        fig.show()

    np.testing.assert_allclose(gim, jgim, rtol=0, atol=atol)


def test_metacal_vmap():
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
    for _seed in range(10):
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

    ims = np.stack(ims)
    psfs = np.stack(psfs)
    nse_ims = np.stack(nse_ims)

    gt0 = time.time()
    for im, psf, nse_im in zip(ims, psfs, nse_ims):
        _metacal_galsim(
            im.copy(), psf.copy(), nse_im.copy(), scale, target_fwhm, g1, 128
        )
    gt0 = time.time() - gt0
    print("Galsim time: ", gt0 * 1e3, " [ms]")

    jit_mcal = jax.jit(
        jax.vmap(
            _metacal_jax_galsim,
            in_axes=(0, 0, 0, None, None, None, None),
        ),
        static_argnums=6,
    )

    for i in range(2):
        if i == 0:
            msg = "jit warmup"
        elif i == 1:
            msg = "jit"

        jgt0 = time.time()
        jit_mcal(
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
