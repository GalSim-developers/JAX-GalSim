import galsim as ref_galsim
import numpy as np

import jax_galsim


def test_ref_impl_convolve():
    """Validates convolutions against reference GalSim"""
    dx = 0.2
    fwhm_backwards_compatible = 1.0927449310213702

    # First way to do a convolution
    def conv1(galsim):
        psf = galsim.Gaussian(fwhm=fwhm_backwards_compatible, flux=1)
        pixel = galsim.Pixel(scale=dx, flux=1.0)
        conv = galsim.Convolve([psf, pixel], real_space=False)
        return conv.drawImage(nx=32, ny=32, scale=dx, method="sb", use_true_center=False).array

    np.testing.assert_array_almost_equal(
        conv1(ref_galsim),
        conv1(jax_galsim),
        5,
        err_msg="Gaussian convolved with Pixel disagrees with expected result",
    )

    # Second way of doing a convolution
    def conv2(galsim):
        psf = galsim.Gaussian(fwhm=fwhm_backwards_compatible, flux=1)
        pixel = galsim.Pixel(scale=dx, flux=1.0)
        conv = galsim.Convolve(psf, pixel, real_space=False)
        return conv.drawImage(nx=32, ny=32, scale=dx, method="sb", use_true_center=False).array

    np.testing.assert_array_almost_equal(
        conv2(ref_galsim),
        conv2(jax_galsim),
        5,
        err_msg="GSObject Convolve(psf,pixel) disagrees with expected result",
    )


def test_ref_impl_shearconvolve():
    """Verifies that a convolution of a Sheared Gaussian and a Box Profile
    return the expected results.
    """
    e1 = 0.05
    e2 = 0.0
    dx = 0.2

    def func(galsim):
        psf = galsim.Gaussian(flux=1, sigma=1).shear(e1=e1, e2=e2).rotate(33 * galsim.degrees).shift(0.1, 0.4) * 1.1
        pixel = galsim.Pixel(scale=dx, flux=1.0)
        conv = galsim.Convolve([psf, pixel])
        return conv.drawImage(nx=32, ny=32, scale=dx, method="sb", use_true_center=False).array

    np.testing.assert_array_almost_equal(
        func(ref_galsim),
        func(jax_galsim),
        5,
        err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result",
    )
