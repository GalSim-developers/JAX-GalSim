import numpy as np

import jax_galsim as galsim

from galsim_test_helpers import all_obj_diff


def test_exponential_properties():
    """Test some basic properties of the Exponential profile.
    """
    test_flux = 17.9
    test_scale = 1.8
    expon = galsim.Exponential(flux=test_flux, scale_radius=test_scale)

    # Check Fourier properties
    np.testing.assert_almost_equal(expon.maxk, 10 / test_scale)
    np.testing.assert_almost_equal(expon.stepk, 0.37436747851 / test_scale)
    np.testing.assert_almost_equal(expon.flux, test_flux)
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        expon = galsim.Exponential(flux=inFlux, scale_radius=1.8)
        outFlux = expon.flux
        np.testing.assert_almost_equal(outFlux, inFlux)


def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Exponential.  Params include half_light_radius, scale_radius, flux, gsparams
    # The following should all test unequal:
    gals = [galsim.Exponential(half_light_radius=1.0),
            galsim.Exponential(half_light_radius=1.1),
            galsim.Exponential(scale_radius=1.0),
            galsim.Exponential(half_light_radius=1.0, flux=1.1),
            galsim.Exponential(half_light_radius=1.0, gsparams=gsp)]
    all_obj_diff(gals)
