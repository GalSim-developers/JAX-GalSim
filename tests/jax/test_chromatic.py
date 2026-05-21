"""Tests for jax_galsim chromatic PSF support.

Verifies:
- SED and Bandpass construction and evaluation
- Chromatic (separable) drawImage correctness
- ChromaticAtmosphere (non-separable) drawImage correctness
- ChromaticConvolution (galaxy × SED ⊗ PSF) correctness
- jax.jit compatibility for all paths
- jax.grad compatibility for SED-flux differentiation
- Pytree round-trip (tree_flatten / tree_unflatten)
- Numerical agreement with analytic expectations
"""

# ruff: noqa: E402,I001

import jax
import jax.numpy as jnp
import pytest

# Enable float64 for accuracy
jax.config.update("jax_enable_x64", True)

import jax_galsim as jgal
from jax_galsim.chromatic import ChromaticAtmosphere, ChromaticConvolution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WAVE = jnp.linspace(400.0, 900.0, 256)  # nm
BP = jgal.Bandpass.tophat(550.0, 750.0)  # 200 nm wide, throughput = 1
BP_NARROW = jgal.Bandpass.tophat(600.0, 700.0)  # 100 nm wide


def flat_sed(scale=1.0):
    return jgal.SED(WAVE, jnp.ones(256) * scale)


# ---------------------------------------------------------------------------
# SED tests
# ---------------------------------------------------------------------------


def test_sed_evaluation():
    sed = flat_sed()
    assert float(sed(600.0)) == pytest.approx(1.0, rel=1e-5)
    assert float(sed(300.0)) == pytest.approx(0.0)   # outside range


def test_sed_redshift():
    sed = jgal.SED(WAVE, jnp.ones(256), redshift=1.0)
    # observed 800 nm → rest-frame 400 nm → flux = 1.0
    assert float(sed(800.0)) == pytest.approx(1.0, rel=1e-4)
    # observed 400 nm → rest-frame 200 nm → outside range → 0
    assert float(sed(400.0)) == pytest.approx(0.0, abs=1e-6)


def test_sed_calculate_flux():
    sed = flat_sed()
    flux = float(sed.calculateFlux(BP))
    # ∫_550^750 1 dλ = 200 nm
    assert flux == pytest.approx(200.0, rel=1e-3)


def test_sed_pytree_roundtrip():
    sed = flat_sed(2.0)
    leaves, treedef = jax.tree_util.tree_flatten(sed)
    sed2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(sed2(600.0)) == pytest.approx(2.0, rel=1e-5)


def test_sed_arithmetic():
    sed1 = flat_sed(2.0)
    sed2 = sed1 * 3.0
    assert float(sed2(600.0)) == pytest.approx(6.0, rel=1e-5)

    sed3 = sed1 + flat_sed(1.0)
    assert float(sed3(600.0)) == pytest.approx(3.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Bandpass tests
# ---------------------------------------------------------------------------


def test_bandpass_evaluation():
    bp = jgal.Bandpass.tophat(550.0, 750.0)
    assert float(bp(625.0)) == pytest.approx(1.0)
    assert float(bp(500.0)) == pytest.approx(0.0)
    assert float(bp(800.0)) == pytest.approx(0.0)


def test_bandpass_effective_wavelength():
    bp = jgal.Bandpass.tophat(550.0, 750.0)
    lam_eff = bp.effective_wavelength
    assert isinstance(lam_eff, float)
    assert lam_eff == pytest.approx(650.0, rel=1e-4)


def test_bandpass_effective_wavelength_concrete():
    """effective_wavelength must be a concrete Python float (safe under JIT)."""
    bp = jgal.Bandpass.tophat(550.0, 750.0)
    lam_eff = bp.effective_wavelength
    # If this were a JAX tracer, float() would raise ConcretizationTypeError
    assert isinstance(lam_eff, float)


def test_bandpass_mul():
    bp1 = jgal.Bandpass.tophat(500.0, 700.0)
    bp2 = jgal.Bandpass.tophat(600.0, 800.0)
    bp = bp1 * bp2
    assert float(bp(650.0)) == pytest.approx(1.0)
    assert float(bp(550.0)) == pytest.approx(0.0)
    assert float(bp(750.0)) == pytest.approx(0.0)


def test_bandpass_pytree_roundtrip():
    bp = jgal.Bandpass.tophat(550.0, 750.0)
    leaves, treedef = jax.tree_util.tree_flatten(bp)
    bp2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(bp2(625.0)) == pytest.approx(1.0)
    assert bp2.effective_wavelength == pytest.approx(650.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Chromatic (separable) tests
# ---------------------------------------------------------------------------


def test_chromatic_construction():
    sed = flat_sed()
    gal = jgal.Gaussian(half_light_radius=0.5) * sed
    assert gal._separable


def test_chromatic_drawImage_flux():
    """Image pixel sum should equal ∫ SED(λ) × BP(λ) dλ."""
    sed = flat_sed()
    gal = jgal.Gaussian(half_light_radius=0.5) * sed
    img = gal.drawImage(BP, scale=0.2, nx=32, ny=32)
    # ∫_550^750 1 dλ = 200
    assert float(img.array.sum()) == pytest.approx(200.0, rel=5e-3)


def test_chromatic_drawImage_narrow_bandpass():
    """Narrower bandpass → smaller total flux."""
    sed = flat_sed()
    gal = jgal.Gaussian(half_light_radius=0.5) * sed
    img = gal.drawImage(BP_NARROW, scale=0.2, nx=32, ny=32)
    assert float(img.array.sum()) == pytest.approx(100.0, rel=5e-3)


def test_chromatic_jit():
    @jax.jit
    def render(flux):
        sed = jgal.SED(WAVE, flux)
        gal = jgal.Gaussian(half_light_radius=0.5) * sed
        return gal.drawImage(BP, scale=0.2, nx=32, ny=32).array.sum()

    result = render(jnp.ones(256))
    assert float(result) == pytest.approx(200.0, rel=5e-3)


def test_chromatic_grad():
    @jax.jit
    def render(flux):
        sed = jgal.SED(WAVE, flux)
        gal = jgal.Gaussian(half_light_radius=0.5) * sed
        return gal.drawImage(BP, scale=0.2, nx=32, ny=32).array.sum()

    grad = jax.grad(render)(jnp.ones(256))
    grad_arr = jnp.asarray(grad)

    # Gradient sums to total bandpass flux ≈ 200
    assert float(grad.sum()) == pytest.approx(200.0, rel=5e-2)
    # Outside bandpass → zero gradient
    idx_out = int((420 - 400) / (900 - 400) * 255)
    assert float(grad[idx_out]) == pytest.approx(0.0, abs=1e-8)
    # Inside bandpass region has positive total contribution
    mask_in = (WAVE >= 550) & (WAVE <= 750)
    assert float(grad_arr[mask_in].sum()) > 0.0


def test_chromatic_jit_recompile():
    """JIT should reuse compiled code when called twice with different flux."""
    @jax.jit
    def render(flux):
        sed = jgal.SED(WAVE, flux)
        gal = jgal.Gaussian(half_light_radius=0.5) * sed
        return gal.drawImage(BP, scale=0.2, nx=32, ny=32).array.sum()

    r1 = float(render(jnp.ones(256)))
    r2 = float(render(jnp.ones(256) * 2.0))
    assert r2 == pytest.approx(r1 * 2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# ChromaticAtmosphere tests
# ---------------------------------------------------------------------------


def test_chromatic_atmosphere_evaluate():
    psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
    prof = psf.evaluateAtWavelength(700.0)
    # At reference wavelength, FWHM should be exactly fwhm_ref
    # jax_galsim Gaussian exposes sigma; FWHM = sigma * fwhm_factor
    assert isinstance(prof, jgal.Gaussian)
    fwhm = float(prof.sigma) * jgal.Gaussian._fwhm_factor
    assert fwhm == pytest.approx(0.7, rel=1e-4)


def test_chromatic_atmosphere_scaling():
    psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
    prof_blue = psf.evaluateAtWavelength(350.0)
    prof_red = psf.evaluateAtWavelength(700.0)
    # FWHM ∝ λ^alpha = λ^(-0.2) → bluer is wider (alpha < 0)
    assert float(prof_blue.sigma) > float(prof_red.sigma)


def test_chromatic_atmosphere_drawImage_flux():
    """Total flux = ∫ BP(λ) × 1 dλ = 200 (flat SED, unit PSF flux)."""
    psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
    img = psf.drawImage(BP, scale=0.2, nx=32, ny=32)
    assert float(img.array.sum()) == pytest.approx(200.0, rel=5e-3)


def test_chromatic_atmosphere_moffat():
    psf = ChromaticAtmosphere(
        fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2,
        profile="moffat", moffat_beta=4.765
    )
    prof = psf.evaluateAtWavelength(700.0)
    assert isinstance(prof, jgal.Moffat)
    img = psf.drawImage(BP, scale=0.2, nx=32, ny=32)
    assert float(img.array.sum()) == pytest.approx(200.0, rel=5e-2)


def test_chromatic_atmosphere_pytree():
    psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
    leaves, treedef = jax.tree_util.tree_flatten(psf)
    psf2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert psf2._fwhm_ref == pytest.approx(0.7)
    assert psf2._alpha == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# ChromaticConvolution tests
# ---------------------------------------------------------------------------


def test_chromatic_convolution_flux():
    """Galaxy × SED ⊗ ChromaticAtmosphere: flux = ∫ SED(λ) × BP(λ) dλ."""
    sed = flat_sed()
    gal = jgal.Gaussian(half_light_radius=0.5) * sed
    psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
    final = ChromaticConvolution([gal, psf])
    img = final.drawImage(BP, scale=0.2, nx=64, ny=64, n_waves=32)
    assert float(img.array.sum()) == pytest.approx(200.0, rel=5e-2)


def test_chromatic_convolution_jit():
    @jax.jit
    def render(flux):
        sed = jgal.SED(WAVE, flux)
        gal = jgal.Gaussian(half_light_radius=0.5) * sed
        psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
        return ChromaticConvolution([gal, psf]).drawImage(
            BP, scale=0.2, nx=64, ny=64, n_waves=32
        ).array.sum()

    result = render(jnp.ones(256))
    assert float(result) == pytest.approx(200.0, rel=5e-2)


def test_chromatic_convolution_grad():
    @jax.jit
    def render(flux):
        sed = jgal.SED(WAVE, flux)
        gal = jgal.Gaussian(half_light_radius=0.5) * sed
        psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
        return ChromaticConvolution([gal, psf]).drawImage(
            BP, scale=0.2, nx=64, ny=64, n_waves=32
        ).array.sum()

    grad = jax.grad(render)(jnp.ones(256))
    grad_arr = jnp.asarray(grad)

    # Gradient is nonzero only at wavelengths that fall inside the bandpass
    # (the 32 quadrature points lie in [550, 750] nm).
    # - sum of all gradients ≈ total bandpass flux = 200
    assert float(grad.sum()) == pytest.approx(200.0, rel=5e-2)
    # - max gradient is positive
    assert float(grad.max()) > 0.0
    # - indices corresponding to wavelengths outside bandpass have zero gradient
    #   ~420 nm → outside [550, 750] bandpass
    idx_out = int((420 - 400) / (900 - 400) * 255)
    assert float(grad[idx_out]) == pytest.approx(0.0, abs=1e-8)
    # - indices well inside bandpass region have nonzero total contribution
    mask_in = (WAVE >= 550) & (WAVE <= 750)
    assert float(grad_arr[mask_in].sum()) > 0.0


def test_chromatic_convolution_linearity():
    """Doubling SED flux doubles image sum."""
    @jax.jit
    def render(flux):
        sed = jgal.SED(WAVE, flux)
        gal = jgal.Gaussian(half_light_radius=0.5) * sed
        psf = ChromaticAtmosphere(fwhm_ref=0.7, lam_ref=700.0, alpha=-0.2)
        return ChromaticConvolution([gal, psf]).drawImage(
            BP, scale=0.2, nx=64, ny=64, n_waves=32
        ).array.sum()

    s1 = float(render(jnp.ones(256)))
    s2 = float(render(jnp.ones(256) * 2.0))
    assert s2 == pytest.approx(s1 * 2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# ChromaticSum tests
# ---------------------------------------------------------------------------


def test_chromatic_sum_flux():
    sed1 = flat_sed(1.0)
    sed2 = flat_sed(2.0)
    gal1 = jgal.Gaussian(half_light_radius=0.3) * sed1
    gal2 = jgal.Gaussian(half_light_radius=0.8) * sed2
    combined = gal1 + gal2
    img = combined.drawImage(BP, scale=0.2, nx=32, ny=32)
    # Total flux = (1 + 2) × 200 = 600
    assert float(img.array.sum()) == pytest.approx(600.0, rel=1e-2)


# ---------------------------------------------------------------------------
# Monkey-patch (GSObject * SED) tests
# ---------------------------------------------------------------------------


def test_gsobject_mul_sed():
    sed = flat_sed()
    gal = jgal.Gaussian(half_light_radius=0.5)
    from jax_galsim.chromatic import Chromatic
    result = gal * sed
    assert isinstance(result, Chromatic)


def test_gsobject_mul_scalar():
    gal = jgal.Gaussian(half_light_radius=0.5, flux=1.0)
    scaled = gal * 5.0
    assert float(scaled.flux) == pytest.approx(5.0)


if __name__ == "__main__":
    # Run all tests inline for quick check
    import sys

    tests = [
        test_sed_evaluation,
        test_sed_redshift,
        test_sed_calculate_flux,
        test_sed_pytree_roundtrip,
        test_sed_arithmetic,
        test_bandpass_evaluation,
        test_bandpass_effective_wavelength,
        test_bandpass_effective_wavelength_concrete,
        test_bandpass_mul,
        test_bandpass_pytree_roundtrip,
        test_chromatic_construction,
        test_chromatic_drawImage_flux,
        test_chromatic_drawImage_narrow_bandpass,
        test_chromatic_jit,
        test_chromatic_grad,
        test_chromatic_jit_recompile,
        test_chromatic_atmosphere_evaluate,
        test_chromatic_atmosphere_scaling,
        test_chromatic_atmosphere_drawImage_flux,
        test_chromatic_atmosphere_moffat,
        test_chromatic_atmosphere_pytree,
        test_chromatic_convolution_flux,
        test_chromatic_convolution_jit,
        test_chromatic_convolution_grad,
        test_chromatic_convolution_linearity,
        test_chromatic_sum_flux,
        test_gsobject_mul_sed,
        test_gsobject_mul_scalar,
    ]

    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)

    print()
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("Failed:", failed)
        sys.exit(1)
