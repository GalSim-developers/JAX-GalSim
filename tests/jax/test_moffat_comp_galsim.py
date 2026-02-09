import galsim as _galsim
import jax.numpy as jnp
import numpy as np

import jax_galsim as galsim


def test_moffat_comp_galsim_maxk():
    psfs = [
        # Make sure to include all the specialized betas we have in C++ layer.
        # The scale_radius and flux don't matter, but vary themm too.
        # Note: We also specialize beta=1, but that seems to be impossible to realize,
        #       even when it is trunctatd.
        galsim.Moffat(beta=1.5, scale_radius=1, flux=1),
        galsim.Moffat(beta=1.5001, scale_radius=1, flux=1),
        galsim.Moffat(beta=2, scale_radius=0.8, flux=23),
        galsim.Moffat(beta=2.5, scale_radius=1.8e-3, flux=2),
        galsim.Moffat(beta=3, scale_radius=1.8e3, flux=35),
        galsim.Moffat(beta=3.5, scale_radius=1.3, flux=123),
        galsim.Moffat(beta=4, scale_radius=4.9, flux=23),
        galsim.Moffat(beta=1.22, scale_radius=23, flux=23),
        galsim.Moffat(beta=3.6, scale_radius=2, flux=23),
        galsim.Moffat(beta=12.9, scale_radius=5, flux=23),
        galsim.Moffat(beta=1.22, scale_radius=7, flux=23, trunc=30),
        galsim.Moffat(beta=3.6, scale_radius=9, flux=23, trunc=50),
        galsim.Moffat(beta=12.9, scale_radius=11, flux=23, trunc=1000),
    ]
    threshs = [1.0e-3, 1.0e-4, 0.03]
    print("\nbeta \t trunc \t thresh \t kValue(maxk) \t jgs-maxk \t gs-maxk")
    for psf in psfs:
        for thresh in threshs:
            psf = psf.withGSParams(maxk_threshold=thresh)
            gpsf = _galsim.Moffat(
                beta=psf.beta,
                scale_radius=psf.scale_radius,
                flux=psf.flux,
                trunc=psf.trunc,
            )
            gpsf = gpsf.withGSParams(maxk_threshold=thresh)
            fk = psf.kValue(psf.maxk, 0).real / psf.flux

            print(
                f"{psf.beta} \t {int(psf.trunc)} \t {thresh:.1e} \t {fk:.3e} \t {psf.maxk:.3e} \t {gpsf.maxk:.3e}"
            )
            np.testing.assert_allclose(
                psf.kValue(0.0, 0.0), gpsf.kValue(0.0, 0.0), rtol=1e-5
            )
            np.testing.assert_allclose(
                psf.kValue(0.0, 0.1), gpsf.kValue(0.0, 0.1), rtol=1e-5
            )
            np.testing.assert_allclose(
                psf.kValue(-1.0, 0.0), gpsf.kValue(-1.0, 0.0), rtol=1e-5
            )
            np.testing.assert_allclose(
                psf.kValue(1.0, 0.0), gpsf.kValue(1.0, 0.0), rtol=1e-5
            )
            np.testing.assert_allclose(gpsf.maxk, psf.maxk, rtol=0.25, atol=0)


def test_moffat_conv_nan_float32():
    # test case from https://github.com/GalSim-developers/JAX-GalSim/issues/179
    gal_flux = 1.0e5  # counts
    gal_r0 = 2.7  # arcsec
    g1 = 0.1  #
    g2 = 0.2  #
    psf_beta = 5  #
    psf_re = 1.0  # arcsec
    pixel_scale = 0.2  # arcsec / pixel

    # Define the galaxy profile.
    gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

    # Shear the galaxy by some value.
    gal = gal.shear(g1=g1, g2=g2)

    # Define the PSF profile.
    psf = galsim.Moffat(beta=psf_beta, flux=1.0, half_light_radius=psf_re)

    # Final profile is the convolution of these.
    final = galsim.Convolve([gal, psf])

    img_arr = final.drawImage(scale=pixel_scale, dtype=jnp.float32).array

    assert jnp.all(jnp.isfinite(img_arr))
