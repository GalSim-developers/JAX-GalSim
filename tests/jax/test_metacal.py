import time

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np

import jax_galsim


def _metacal_galsim(gs, im, psf, nse_im, scale, target_fwhm, g1):
    res = {}
    iim = gs.InterpolatedImage(gs.ImageD(im), scale=scale, x_interpolant="lanczos15")
    ipsf = gs.InterpolatedImage(gs.ImageD(psf), scale=scale, x_interpolant="lanczos15")
    inse = gs.InterpolatedImage(
        gs.ImageD(np.rot90(nse_im)), scale=scale, x_interpolant="lanczos15"
    )

    res["maxk"] = (iim.maxk, ipsf.maxk, inse.maxk)
    res["stepk"] = (iim.stepk, ipsf.stepk, inse.stepk)

    res["iim"] = iim.drawImage(
        nx=33, ny=33, scale=scale, method="no_pixel"
    ).array.astype(np.float64)
    res["ipsf"] = ipsf.drawImage(
        nx=33, ny=33, scale=scale, method="no_pixel"
    ).array.astype(np.float64)

    ppsf_iim = gs.Convolve(iim, gs.Deconvolve(ipsf))
    ppsf_iim = ppsf_iim.shear(g1=g1, g2=0.0)

    sim = (
        gs.Convolve(ppsf_iim, gs.Gaussian(fwhm=target_fwhm))
        .drawImage(
            nx=33,
            ny=33,
            scale=scale,
            method="no_pixel",
        )
        .array.astype(np.float64)
    )
    res["im"] = sim

    ppsf_inse = gs.Convolve(inse, gs.Deconvolve(ipsf))
    ppsf_inse = ppsf_iim.shear(g1=g1, g2=0.0)
    snse = (
        gs.Convolve(ppsf_inse, gs.Gaussian(fwhm=target_fwhm))
        .drawImage(
            nx=33,
            ny=33,
            scale=scale,
            method="no_pixel",
        )
        .array.astype(np.float64)
    )
    res["nse"] = snse
    res["tot"] = sim + np.rot90(snse, 3)
    return res


def _metacal_jax_galsim(im, psf, nse_im, scale, target_fwhm, g1, nk):
    gs = jax_galsim
    iim = gs.InterpolatedImage(gs.ImageD(im), scale=scale, x_interpolant="lanczos15")
    ipsf = gs.InterpolatedImage(gs.ImageD(psf), scale=scale, x_interpolant="lanczos15")
    inse = gs.InterpolatedImage(
        gs.ImageD(jnp.rot90(nse_im)), scale=scale, x_interpolant="lanczos15"
    )

    res = {}

    res["maxk"] = (iim.maxk, ipsf.maxk, inse.maxk)
    res["stepk"] = (iim.stepk, ipsf.stepk, inse.stepk)

    res["iim"] = iim.drawImage(
        nx=33, ny=33, scale=scale, method="no_pixel"
    ).array.astype(np.float64)
    res["ipsf"] = ipsf.drawImage(
        nx=33, ny=33, scale=scale, method="no_pixel"
    ).array.astype(np.float64)

    ppsf_iim = gs.Convolve(iim, gs.Deconvolve(ipsf))
    ppsf_iim = ppsf_iim.shear(g1=g1, g2=0.0)

    sim = (
        gs.Convolve(
            ppsf_iim,
            gs.Gaussian(fwhm=target_fwhm),
            gsparams=gs.GSParams(minimum_fft_size=nk, maximum_fft_size=nk),
        )
        .drawImage(
            nx=33,
            ny=33,
            scale=scale,
            method="no_pixel",
        )
        .array.astype(np.float64)
    )
    res["im"] = sim

    ppsf_inse = gs.Convolve(inse, gs.Deconvolve(ipsf))
    ppsf_inse = ppsf_iim.shear(g1=g1, g2=0.0)
    snse = (
        gs.Convolve(
            ppsf_inse,
            gs.Gaussian(fwhm=target_fwhm),
            gsparams=gs.GSParams(minimum_fft_size=nk, maximum_fft_size=nk),
        )
        .drawImage(
            nx=33,
            ny=33,
            scale=scale,
            method="no_pixel",
        )
        .array.astype(np.float64)
    )
    res["nse"] = snse
    res["tot"] = sim + jnp.rot90(snse, 3)
    return res


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
        _galsim, im.copy(), psf.copy(), nse_im.copy(), scale, target_fwhm, g1
    )
    gt0 = time.time() - gt0

    jit_mcal = jax.jit(_metacal_jax_galsim, static_argnums=6)

    for _ in range(2):
        jgt0 = time.time()
        jgres = jit_mcal(
            im,
            psf,
            nse_im,
            scale,
            target_fwhm,
            g1,
            128,
        )
        jgt0 = time.time() - jgt0
        print("Jax-Galsim time: ", jgt0 * 1e3, " [ms]")

    print("Galsim time: ", gt0 * 1e3, " [ms]")

    np.testing.assert_allclose(gres["maxk"], jgres["maxk"], rtol=2e-2)
    np.testing.assert_allclose(gres["stepk"], jgres["stepk"], rtol=2e-2)

    for k in ["iim", "ipsf", "im", "nse", "tot"]:
        gim = gres[k]
        jgim = jgres[k]

        if k in ["iim", "ipsf"]:
            atol = 1e-7
        else:
            atol = 5e-5

        if not np.allclose(gim, jgim, rtol=0, atol=atol):
            import proplot as pplt

            fig, axs = pplt.subplots(ncols=3, nrows=5, figsize=(4.5, 7.5))
            print(axs.shape)
            for row, _k in enumerate(["iim", "ipsf", "im", "nse", "tot"]):
                _gim = gres[_k]
                _jgim = jgres[_k]
                axs[row, 0].imshow(np.arcsinh(_gim / nse))
                axs[row, 1].imshow(np.arcsinh(_jgim / nse))
                axs[row, 2].imshow(_jgim - _gim)
            fig.show()

        np.testing.assert_allclose(
            gim, jgim, err_msg=f"Failed for {k}", rtol=0, atol=atol
        )


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
    for _seed in range(1000):
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
            _galsim, im.copy(), psf.copy(), nse_im.copy(), scale, target_fwhm, g1
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

    for _ in range(2):
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
        print("Jax-Galsim time: ", jgt0 * 1e3, " [ms]")
