# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import numpy as np

import jax_galsim as galsim

from galsim_test_helpers import all_obj_diff


def test_gaussian_properties():
    """Test some basic properties of the Gaussian profile."""
    test_flux = 17.9
    test_sigma = 1.8
    gauss = galsim.Gaussian(flux=test_flux, sigma=test_sigma)

    # Check Fourier properties
    np.testing.assert_almost_equal(gauss.maxk, 3.7169221888498383 / test_sigma)
    np.testing.assert_almost_equal(gauss.stepk, 0.533644625664 / test_sigma)
    np.testing.assert_almost_equal(gauss.flux, test_flux)
    import math

    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        gauss = galsim.Gaussian(flux=inFlux, sigma=2.0)
        outFlux = gauss.flux
        np.testing.assert_almost_equal(outFlux, inFlux)


def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Gaussian.  Params include sigma, fwhm, half_light_radius, flux, and gsparams.
    # The following should all test unequal:
    gals = [
        galsim.Gaussian(sigma=1.0),
        galsim.Gaussian(sigma=1.1),
        galsim.Gaussian(fwhm=1.0),
        galsim.Gaussian(half_light_radius=1.0),
        galsim.Gaussian(half_light_radius=1.1),
        galsim.Gaussian(sigma=1.2, flux=1.0),
        galsim.Gaussian(sigma=1.2, flux=1.1),
        galsim.Gaussian(sigma=1.2, gsparams=gsp),
    ]
    # Check that setifying doesn't remove any duplicate items.
    all_obj_diff(gals)
