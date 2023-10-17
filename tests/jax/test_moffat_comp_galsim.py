import galsim as _galsim
import jax_galsim as galsim
import numpy as np


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
    threshs = [1.e-3, 1.e-4, 0.03]
    print('\nbeta \t trunc \t thresh \t kValue(maxk) \t jgs-maxk \t gs-maxk')
    for psf in psfs:
        for thresh in threshs:
            psf = psf.withGSParams(maxk_threshold=thresh)
            gpsf = _galsim.Moffat(beta=psf.beta, scale_radius=psf.scale_radius, flux=psf.flux, trunc=psf.trunc)
            gpsf = gpsf.withGSParams(maxk_threshold=thresh)
            fk = psf.kValue(psf.maxk, 0).real / psf.flux

            print(f'{psf.beta} \t {int(psf.trunc)} \t {thresh:.1e} \t {fk:.3e} \t {psf.maxk:.3e} \t {gpsf.maxk:.3e}')
            np.testing.assert_allclose(psf.kValue(0.0, 0.0), gpsf.kValue(0.0, 0.0), rtol=1e-5)
            np.testing.assert_allclose(psf.kValue(0.0, 0.1), gpsf.kValue(0.0, 0.1), rtol=1e-5)
            np.testing.assert_allclose(psf.kValue(-1.0, 0.0), gpsf.kValue(-1.0, 0.0), rtol=1e-5)
            np.testing.assert_allclose(psf.kValue(1.0, 0.0), gpsf.kValue(1.0, 0.0), rtol=1e-5)
            np.testing.assert_allclose(gpsf.maxk, psf.maxk, rtol=0.5, atol=0)
