import galsim as _galsim
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

    gres = _metacal_galsim(
        _galsim, im.copy(), psf.copy(), nse_im.copy(), scale, target_fwhm, g1
    )
    jgres = _metacal_galsim(
        jax_galsim, im.copy(), psf.copy(), nse_im.copy(), scale, target_fwhm, g1
    )

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
