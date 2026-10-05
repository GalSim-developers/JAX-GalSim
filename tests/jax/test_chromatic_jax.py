"""JAX-specific chromatic tests.

The behavioral GalSim coverage for SED, Bandpass, and chromatic profiles comes
from tests/GalSim/tests through the conftest.py harness, which imports
jax_galsim as galsim.  This file is intentionally limited to JAX behavior and
local regressions that the upstream GalSim tests cannot express directly:

- jit and grad support for traced SED and Bandpass arrays
- pytree round-trips for chromatic objects
- the ChromaticSum non-separability regression
- the local GSObject * SED monkey patch
"""

import jax
import jax.numpy as jnp
import pytest

import jax_galsim as jgs
from jax_galsim.chromatic import (
    ChromaticAtmosphere,
    ChromaticConvolution,
    SimpleChromaticTransformation,
)

WAVE = jnp.linspace(400.0, 900.0, 256)
BP = jgs.Bandpass.tophat(550.0, 750.0)


def flat_sed(scale=1.0):
    return jgs.SED(WAVE, jnp.ones_like(WAVE) * scale)


def test_sed_bandpass_chromatic_pytree_roundtrips():
    sed = flat_sed(2.0)
    bandpass = jgs.Bandpass.tophat(550.0, 750.0)
    gal = jgs.Gaussian(half_light_radius=0.5) * sed
    psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)

    for obj in [sed, bandpass, gal, psf]:
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, obj.__class__)

    sed_leaves, sed_treedef = jax.tree_util.tree_flatten(sed)
    rebuilt_sed = jax.tree_util.tree_unflatten(sed_treedef, sed_leaves)
    assert float(rebuilt_sed(600.0)) == pytest.approx(2.0, rel=1e-5)


def test_bandpass_call_jit_grad_throughput():
    wave = jnp.linspace(500.0, 800.0, 32)
    throughput = jnp.exp(-0.5 * ((wave - 650.0) / 45.0) ** 2)

    @jax.jit
    def sample(tp):
        return jgs.Bandpass(wave, tp)(650.0)

    value = sample(throughput)
    grad = jax.grad(sample)(throughput)

    assert float(value) == pytest.approx(1.0, rel=5e-3)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_bandpass_effective_wavelength_jit_grad_throughput():
    wave = jnp.linspace(500.0, 800.0, 32)
    throughput = jnp.exp(-0.5 * ((wave - 660.0) / 50.0) ** 2)

    @jax.jit
    def eff(tp):
        return jgs.Bandpass(wave, tp).effective_wavelength

    value = eff(throughput)
    grad = jax.grad(eff)(throughput)

    assert float(value) == pytest.approx(660.0, rel=2e-2)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_separable_chromatic_draw_jit_grad_sed_flux():
    @jax.jit
    def render(flux):
        sed = jgs.SED(WAVE, flux)
        gal = jgs.Gaussian(half_light_radius=0.5) * sed
        return gal.drawImage(BP, scale=0.2, nx=32, ny=32).array.sum()

    flux = jnp.ones_like(WAVE)
    result = render(flux)
    grad = jax.grad(render)(flux)
    mask_in = (WAVE >= 550.0) & (WAVE <= 750.0)

    assert float(result) == pytest.approx(200.0, rel=5e-3)
    assert float(grad.sum()) == pytest.approx(200.0, rel=5e-2)
    assert float(grad[10]) == pytest.approx(0.0, abs=1e-8)
    assert float(jnp.asarray(grad)[mask_in].sum()) > 0.0


def test_chromatic_convolution_draw_jit_grad_sed_flux():
    @jax.jit
    def render(flux):
        sed = jgs.SED(WAVE, flux)
        gal = jgs.Gaussian(half_light_radius=0.5) * sed
        psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
        return (
            ChromaticConvolution([gal, psf])
            .drawImage(BP, scale=0.2, nx=64, ny=64, n_waves=32)
            .array.sum()
        )

    flux = jnp.ones_like(WAVE)
    result = render(flux)
    grad = jax.grad(render)(flux)
    mask_in = (WAVE >= 550.0) & (WAVE <= 750.0)

    assert float(result) == pytest.approx(200.0, rel=5e-2)
    assert float(grad.sum()) == pytest.approx(200.0, rel=5e-2)
    assert float(grad.max()) > 0.0
    assert float(grad[10]) == pytest.approx(0.0, abs=1e-8)
    assert float(jnp.asarray(grad)[mask_in].sum()) > 0.0


def test_chromatic_sum_not_generically_separable():
    sed_disk = jgs.SED(WAVE, 1.0 + 0.2 * (WAVE - 650.0) / 250.0)
    sed_bulge = jgs.SED(WAVE, 1.0 - 0.1 * (WAVE - 650.0) / 250.0)
    disk = jgs.Gaussian(half_light_radius=0.7) * sed_disk
    bulge = jgs.Gaussian(half_light_radius=0.3) * sed_bulge

    source = disk + bulge

    assert not source._separable


def test_chromatic_sum_convolution_matches_split_components():
    sed_disk = jgs.SED(WAVE, 1.0 + 0.2 * (WAVE - 650.0) / 250.0)
    sed_bulge = jgs.SED(WAVE, 1.0 - 0.1 * (WAVE - 650.0) / 250.0)
    disk = jgs.Gaussian(half_light_radius=0.7) * sed_disk
    bulge = jgs.Gaussian(half_light_radius=0.3) * sed_bulge
    psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)

    combined = ChromaticConvolution([disk + bulge, psf]).drawImage(
        BP, scale=0.2, nx=32, ny=32, n_waves=16
    )
    split_disk = ChromaticConvolution([disk, psf]).drawImage(
        BP, scale=0.2, nx=32, ny=32, n_waves=16
    )
    split_bulge = ChromaticConvolution([bulge, psf]).drawImage(
        BP, scale=0.2, nx=32, ny=32, n_waves=16
    )
    split = split_disk.array + split_bulge.array

    assert float(jnp.max(jnp.abs(combined.array - split))) == pytest.approx(
        0.0, abs=5e-4
    )


def test_gsobject_mul_sed_returns_chromatic_transformation():
    result = jgs.Gaussian(half_light_radius=0.5) * flat_sed()

    assert isinstance(result, SimpleChromaticTransformation)


def test_gsobject_mul_scalar_still_scales_flux():
    gal = jgs.Gaussian(half_light_radius=0.5, flux=1.0)
    scaled = gal * 5.0

    assert float(scaled.flux) == pytest.approx(5.0)
