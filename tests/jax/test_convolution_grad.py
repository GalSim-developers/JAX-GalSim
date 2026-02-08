"""Test that gradients through convolution rendering are correct.

This verifies that the structural optimization (breaking false AD dependency
in Convolution._drawKImage) produces correct gradients by comparing against
finite differences.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jax_galsim as galsim

jax.config.update("jax_enable_x64", True)

gsparams = galsim.GSParams(minimum_fft_size=128, maximum_fft_size=128)


def _draw_and_sum(half_light_radius, flux):
    """Draw a convolution of Exponential * Moffat and return the sum of pixel values."""
    gal = galsim.Exponential(
        half_light_radius=half_light_radius, flux=flux, gsparams=gsparams
    )
    psf = galsim.Moffat(beta=3.5, fwhm=0.7, gsparams=gsparams)
    obj = galsim.Convolve(gal, psf, gsparams=gsparams)
    image = obj.drawImage(nx=64, ny=64, scale=0.2, dtype=float)
    return jnp.sum(image.array)


def test_convolution_grad_vs_finite_diff():
    """Test that jax.grad through Convolve(Exponential, Moffat).drawImage()
    matches finite-difference approximation."""
    hlr = 1.0
    flux = 100.0
    eps = 1e-5

    grad_fn = jax.grad(_draw_and_sum, argnums=(0, 1))
    grad_hlr, grad_flux = grad_fn(jnp.float64(hlr), jnp.float64(flux))

    # Finite-difference for half_light_radius
    f_plus = _draw_and_sum(hlr + eps, flux)
    f_minus = _draw_and_sum(hlr - eps, flux)
    fd_grad_hlr = (f_plus - f_minus) / (2 * eps)

    # Finite-difference for flux
    f_plus = _draw_and_sum(hlr, flux + eps)
    f_minus = _draw_and_sum(hlr, flux - eps)
    fd_grad_flux = (f_plus - f_minus) / (2 * eps)

    np.testing.assert_allclose(
        grad_hlr,
        fd_grad_hlr,
        rtol=1e-3,
        atol=0,
        err_msg="Gradient w.r.t. half_light_radius is incorrect",
    )
    np.testing.assert_allclose(
        grad_flux,
        fd_grad_flux,
        rtol=1e-3,
        atol=0,
        err_msg="Gradient w.r.t. flux is incorrect",
    )


def test_sum_grad_vs_finite_diff():
    """Test that jax.grad through Sum.drawImage() is correct."""

    def _draw_sum(flux1, flux2):
        _gsparams = galsim.GSParams(minimum_fft_size=128, maximum_fft_size=128)
        g1 = galsim.Gaussian(sigma=1.5, flux=flux1, gsparams=_gsparams)
        g2 = galsim.Gaussian(sigma=2.0, flux=flux2, gsparams=_gsparams)
        obj = galsim.Add(g1, g2)
        image = obj.drawImage(nx=64, ny=64, scale=0.2, method="no_pixel", dtype=float)
        return jnp.sum(image.array)

    flux1, flux2 = jnp.float64(50.0), jnp.float64(80.0)
    eps = 1e-5

    grad_fn = jax.grad(_draw_sum, argnums=(0, 1))
    grad_f1, grad_f2 = grad_fn(flux1, flux2)

    fd_grad_f1 = (_draw_sum(flux1 + eps, flux2) - _draw_sum(flux1 - eps, flux2)) / (
        2 * eps
    )
    fd_grad_f2 = (_draw_sum(flux1, flux2 + eps) - _draw_sum(flux1, flux2 - eps)) / (
        2 * eps
    )

    np.testing.assert_allclose(
        grad_f1,
        fd_grad_f1,
        rtol=1e-3,
        atol=0,
        err_msg="Sum gradient w.r.t. flux1 is incorrect",
    )
    np.testing.assert_allclose(
        grad_f2,
        fd_grad_f2,
        rtol=1e-3,
        atol=0,
        err_msg="Sum gradient w.r.t. flux2 is incorrect",
    )
