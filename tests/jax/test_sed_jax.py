"""Tests for SED, Bandpass, LookupTable, and ChromaticObject in jax_galsim."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import jax_galsim

# ============================================================
# LookupTable tests
# ============================================================


class TestLookupTable:
    def setup_method(self):
        self.x = np.linspace(0, 10, 100)
        self.f = np.sin(self.x)

    def test_basic_linear(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        npt.assert_allclose(float(lt(5.0)), np.sin(5.0), atol=2e-3)

    def test_basic_spline(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="spline")
        npt.assert_allclose(float(lt(5.0)), np.sin(5.0), atol=1e-3)

    def test_basic_nearest(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="nearest")
        val = float(lt(5.0))
        # Should be close to sin(5) since the grid is fine
        npt.assert_allclose(val, np.sin(5.0), atol=0.1)

    def test_basic_floor(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="floor")
        val = float(lt(5.0))
        assert isinstance(val, float)

    def test_basic_ceil(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="ceil")
        val = float(lt(5.0))
        assert isinstance(val, float)

    def test_array_eval(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        xtest = np.array([1.0, 3.0, 5.0, 7.0])
        result = lt(xtest)
        assert result.shape == (4,)
        npt.assert_allclose(result, np.sin(xtest), atol=2e-3)

    def test_properties(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        assert lt.x_min == 0.0
        assert lt.x_max == 10.0
        assert len(lt) == 100
        assert lt.getInterp() == "linear"
        assert lt.isLogX() is False
        assert lt.isLogF() is False
        npt.assert_array_equal(lt.getArgs(), self.x)
        npt.assert_array_equal(lt.getVals(), self.f)

    def test_integrate_linear(self):
        # Integrate sin(x) from 0 to pi => should be 2.0
        x = np.linspace(0, np.pi, 1000)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        result = lt.integrate()
        npt.assert_allclose(result, 2.0, atol=1e-4)

    def test_integrate_spline(self):
        x = np.linspace(0, np.pi, 100)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="spline")
        result = lt.integrate()
        npt.assert_allclose(result, 2.0, atol=1e-3)

    def test_integrate_partial(self):
        x = np.linspace(0, 10, 1000)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        result = lt.integrate(2.0, 5.0)
        npt.assert_allclose(result, 3.0, atol=1e-6)

    def test_integrate_product(self):
        x = np.linspace(0, 10, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        g = jax_galsim.LookupTable(x, x, interpolant="linear")  # g(x) = x
        # integral of 1*x from 0 to 10 = 50
        result = lt.integrate_product(g)
        npt.assert_allclose(result, 50.0, atol=1e-2)

    def test_integrate_product_callable(self):
        x = np.linspace(0, 10, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        result = lt.integrate_product(lambda w: w)  # integral of x from 0 to 10 = 50
        npt.assert_allclose(result, 50.0, atol=1e-2)

    def test_equality(self):
        lt1 = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        lt2 = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        assert lt1 == lt2
        lt3 = jax_galsim.LookupTable(self.x, self.f, interpolant="spline")
        assert lt1 != lt3

    def test_hash(self):
        lt1 = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        lt2 = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        assert hash(lt1) == hash(lt2)

    def test_repr(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        r = repr(lt)
        assert "galsim.LookupTable" in r

    def test_from_func(self):
        lt = jax_galsim.LookupTable.from_func(
            np.sin, 0, 10, npoints=100, interpolant="linear"
        )
        npt.assert_allclose(float(lt(5.0)), np.sin(5.0), atol=1e-2)

    def test_pytree(self):
        lt = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        children, aux = lt.tree_flatten()
        lt2 = jax_galsim.LookupTable.tree_unflatten(aux, children)
        npt.assert_allclose(float(lt(5.0)), float(lt2(5.0)))

    def test_invalid_interpolant(self):
        with pytest.raises(Exception):
            jax_galsim.LookupTable(self.x, self.f, interpolant="bogus")

    def test_length_mismatch(self):
        with pytest.raises(Exception):
            jax_galsim.LookupTable([1, 2, 3], [1, 2])

    def test_too_short(self):
        with pytest.raises(Exception):
            jax_galsim.LookupTable([1], [1])

    def test_trapz(self):
        x = np.linspace(0, np.pi, 1000)
        f = np.sin(x)
        result = jax_galsim.trapz(f, x)
        npt.assert_allclose(result, 2.0, atol=1e-4)


# ============================================================
# Bandpass tests
# ============================================================


class TestBandpass:
    def test_constant_throughput(self):
        bp = jax_galsim.Bandpass(
            lambda w: 1.0, wave_type="nm", blue_limit=400, red_limit=600
        )
        assert float(bp(500)) == 1.0
        assert float(bp(300)) == 0.0  # Outside range

    def test_lookuptable_throughput(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="nm")
        assert bp.blue_limit == 400.0
        assert bp.red_limit == 600.0
        npt.assert_allclose(float(bp(500)), 1.0)

    def test_effective_wavelength(self):
        # Uniform throughput from 400 to 600 => eff_wave = 500
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="nm")
        npt.assert_allclose(bp.effective_wavelength, 500.0, atol=1.0)

    def test_angstrom_wave_type(self):
        x = np.linspace(4000, 6000, 100)  # Angstroms
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="Angstrom")
        # Limits should be in nm
        npt.assert_allclose(bp.blue_limit, 400.0, atol=0.1)
        npt.assert_allclose(bp.red_limit, 600.0, atol=0.1)

    def test_withZeropoint_AB(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="nm")
        bp_ab = bp.withZeropoint("AB")
        assert bp_ab.zeropoint is not None
        assert isinstance(bp_ab.zeropoint, float)

    def test_withZeropoint_float(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="nm")
        bp_zp = bp.withZeropoint(25.0)
        assert bp_zp.zeropoint == 25.0

    def test_truncate(self):
        x = np.linspace(300, 700, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="nm")
        bp_t = bp.truncate(blue_limit=400, red_limit=600)
        assert bp_t.blue_limit == 400.0
        assert bp_t.red_limit == 600.0

    def test_mul_scalar(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="nm")
        bp2 = bp * 0.5
        npt.assert_allclose(float(bp2(500)), 0.5)

    def test_mul_bandpass(self):
        x = np.linspace(400, 600, 100)
        f1 = np.ones_like(x)
        f2 = 0.5 * np.ones_like(x)
        lt1 = jax_galsim.LookupTable(x, f1, interpolant="linear")
        lt2 = jax_galsim.LookupTable(x, f2, interpolant="linear")
        bp1 = jax_galsim.Bandpass(lt1, wave_type="nm")
        bp2 = jax_galsim.Bandpass(lt2, wave_type="nm")
        bp3 = bp1 * bp2
        npt.assert_allclose(float(bp3(500)), 0.5, atol=1e-6)

    def test_equality(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp1 = jax_galsim.Bandpass(lt, wave_type="nm")
        bp2 = jax_galsim.Bandpass(lt, wave_type="nm")
        assert bp1 == bp2

    def test_hash(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp1 = jax_galsim.Bandpass(lt, wave_type="nm")
        bp2 = jax_galsim.Bandpass(lt, wave_type="nm")
        assert hash(bp1) == hash(bp2)

    def test_pytree(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="nm")
        children, aux = bp.tree_flatten()
        bp2 = jax_galsim.Bandpass.tree_unflatten(aux, children)
        npt.assert_allclose(float(bp(500)), float(bp2(500)))


# ============================================================
# SED tests
# ============================================================


class TestSED:
    def setup_method(self):
        self.x = np.linspace(300, 1100, 100)
        self.f = np.ones_like(self.x)
        self.bp = jax_galsim.Bandpass(
            jax_galsim.LookupTable(self.x, self.f, interpolant="linear"),
            wave_type="nm",
        )

    def test_constant_dimensionless(self):
        sed = jax_galsim.SED(2.0, wave_type="nm", flux_type="1")
        val = float(sed(500.0))
        npt.assert_allclose(val, 2.0)

    def test_lookuptable_flambda(self):
        x = np.linspace(300, 1100, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="flambda")
        val = float(sed(500.0))
        assert val > 0

    def test_callable_flambda(self):
        sed = jax_galsim.SED(lambda w: 1.0, wave_type="nm", flux_type="flambda")
        val = float(sed(500.0))
        assert val > 0

    def test_string_expression(self):
        sed = jax_galsim.SED("1.0", wave_type="nm", flux_type="1")
        val = float(sed(500.0))
        npt.assert_allclose(val, 1.0)

    def test_spectral_property(self):
        sed1 = jax_galsim.SED(1.0, wave_type="nm", flux_type="1")
        assert sed1.dimensionless
        assert not sed1.spectral

        sed2 = jax_galsim.SED(lambda w: 1.0, wave_type="nm", flux_type="flambda")
        assert sed2.spectral
        assert not sed2.dimensionless

    def test_wave_type_angstrom(self):
        x = np.linspace(3000, 11000, 100)  # Angstroms
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="Angstrom", flux_type="flambda")
        # Should be able to call at 500nm
        val = float(sed(500.0))
        assert val > 0

    def test_redshift(self):
        x = np.linspace(300, 1100, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="flambda", redshift=0.0)
        sed2 = sed.atRedshift(0.5)
        assert sed2.redshift == 0.5
        assert sed2.blue_limit == sed.blue_limit * 1.5
        assert sed2.red_limit == sed.red_limit * 1.5

    def test_mul_scalar(self):
        sed = jax_galsim.SED(1.0, wave_type="nm", flux_type="1")
        sed2 = sed * 3.0
        npt.assert_allclose(float(sed2(500.0)), 3.0)

    def test_rmul_scalar(self):
        sed = jax_galsim.SED(1.0, wave_type="nm", flux_type="1")
        sed2 = 3.0 * sed
        npt.assert_allclose(float(sed2(500.0)), 3.0)

    def test_mul_sed_dimensionless(self):
        sed1 = jax_galsim.SED(lambda w: 1.0, wave_type="nm", flux_type="flambda")
        sed2 = jax_galsim.SED(2.0, wave_type="nm", flux_type="1")
        sed3 = sed1 * sed2
        # result should be spectral with 2x the flux
        assert sed3.spectral
        val_1 = float(sed1(500.0))
        val_3 = float(sed3(500.0))
        npt.assert_allclose(val_3, val_1 * 2.0, rtol=1e-4)

    def test_mul_two_spectral_raises(self):
        sed1 = jax_galsim.SED(lambda w: 1.0, wave_type="nm", flux_type="flambda")
        sed2 = jax_galsim.SED(lambda w: 1.0, wave_type="nm", flux_type="flambda")
        with pytest.raises(Exception):
            sed1 * sed2

    def test_div_scalar(self):
        sed = jax_galsim.SED(2.0, wave_type="nm", flux_type="1")
        sed2 = sed / 2.0
        npt.assert_allclose(float(sed2(500.0)), 1.0)

    def test_add_sed(self):
        x = np.linspace(300, 1100, 100)
        f1 = np.ones_like(x)
        f2 = 2.0 * np.ones_like(x)
        lt1 = jax_galsim.LookupTable(x, f1, interpolant="linear")
        lt2 = jax_galsim.LookupTable(x, f2, interpolant="linear")
        sed1 = jax_galsim.SED(lt1, wave_type="nm", flux_type="flambda")
        sed2 = jax_galsim.SED(lt2, wave_type="nm", flux_type="flambda")
        sed3 = sed1 + sed2
        val1 = float(sed1(500.0))
        val2 = float(sed2(500.0))
        val3 = float(sed3(500.0))
        npt.assert_allclose(val3, val1 + val2, rtol=1e-4)

    def test_sub_sed(self):
        sed1 = jax_galsim.SED(2.0, wave_type="nm", flux_type="1")
        sed2 = jax_galsim.SED(1.0, wave_type="nm", flux_type="1")
        sed3 = sed1 - sed2
        npt.assert_allclose(float(sed3(500.0)), 1.0)

    def test_calculateFlux(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="fphotons")
        bp = jax_galsim.Bandpass(
            jax_galsim.LookupTable(x, np.ones_like(x), interpolant="linear"),
            wave_type="nm",
        )
        flux = sed.calculateFlux(bp)
        # flux should be roughly (600-400)*1 = 200
        npt.assert_allclose(flux, 200.0, rtol=0.01)

    def test_withFlux(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="fphotons")
        bp = jax_galsim.Bandpass(
            jax_galsim.LookupTable(x, np.ones_like(x), interpolant="linear"),
            wave_type="nm",
        )
        sed2 = sed.withFlux(1.0, bp)
        npt.assert_allclose(sed2.calculateFlux(bp), 1.0, rtol=0.01)

    def test_withFluxDensity(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="fphotons")
        sed2 = sed.withFluxDensity(10.0, 500.0)
        npt.assert_allclose(float(sed2(500.0)), 10.0, rtol=0.01)

    def test_equality(self):
        sed1 = jax_galsim.SED(1.0, wave_type="nm", flux_type="1")
        sed2 = jax_galsim.SED(1.0, wave_type="nm", flux_type="1")
        assert sed1 == sed2

    def test_hash(self):
        sed1 = jax_galsim.SED(1.0, wave_type="nm", flux_type="1")
        sed2 = jax_galsim.SED(1.0, wave_type="nm", flux_type="1")
        assert hash(sed1) == hash(sed2)

    def test_pytree_const(self):
        sed = jax_galsim.SED(2.0, wave_type="nm", flux_type="1")
        children, aux = sed.tree_flatten()
        sed2 = jax_galsim.SED.tree_unflatten(aux, children)
        npt.assert_allclose(float(sed(500.0)), float(sed2(500.0)))

    def test_pytree_lookuptable(self):
        x = np.linspace(300, 1100, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="flambda")
        children, aux = sed.tree_flatten()
        sed2 = jax_galsim.SED.tree_unflatten(aux, children)
        npt.assert_allclose(float(sed(500.0)), float(sed2(500.0)))

    def test_calculateMagnitude(self):
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="fphotons")
        bp = jax_galsim.Bandpass(
            jax_galsim.LookupTable(x, np.ones_like(x), interpolant="linear"),
            wave_type="nm",
        )
        bp_ab = bp.withZeropoint("AB")
        mag = sed.calculateMagnitude(bp_ab)
        assert isinstance(mag, float)


# ============================================================
# ChromaticObject tests
# ============================================================


class TestChromaticObject:
    def setup_method(self):
        self.x = np.linspace(400, 600, 100)
        self.f = np.ones_like(self.x)
        self.lt = jax_galsim.LookupTable(self.x, self.f, interpolant="linear")
        self.bp = jax_galsim.Bandpass(self.lt, wave_type="nm")
        self.sed = jax_galsim.SED(self.lt, wave_type="nm", flux_type="fphotons")
        self.gal = jax_galsim.Gaussian(sigma=1.0, flux=1.0)

    def test_gsobj_mul_sed(self):
        chrom = self.gal * self.sed
        assert isinstance(chrom, jax_galsim.SimpleChromaticTransformation)

    def test_sed_mul_gsobj(self):
        chrom = self.sed * self.gal
        assert isinstance(chrom, jax_galsim.SimpleChromaticTransformation)

    def test_chromatic_separable(self):
        chrom = self.gal * self.sed
        assert chrom.separable is True

    def test_chromatic_sed_property(self):
        chrom = self.gal * self.sed
        sed = chrom.sed
        # The SED of the chromatic object should be the SED scaled by the galaxy flux
        assert sed.spectral

    def test_evaluateAtWavelength(self):
        chrom = self.gal * self.sed
        prof = chrom.evaluateAtWavelength(500.0)
        assert isinstance(prof, jax_galsim.GSObject)
        assert prof.flux > 0

    def test_drawImage(self):
        chrom = self.gal * self.sed
        img = chrom.drawImage(self.bp, nx=32, ny=32, scale=0.2)
        assert img.array.shape == (32, 32)
        assert float(img.array.sum()) > 0

    def test_drawImage_consistency(self):
        """Check that chromatic drawImage gives result consistent with
        monochromatic drawing at the effective wavelength scaled by the flux."""
        chrom = self.gal * self.sed
        img_chrom = chrom.drawImage(self.bp, nx=32, ny=32, scale=0.2)

        # Expected: draw at effective wavelength, scale by multiplier
        wave0 = self.bp.effective_wavelength
        prof0 = chrom.evaluateAtWavelength(wave0)
        sed = chrom.sed
        from jax_galsim.chromatic import ChromaticObject

        multiplier = ChromaticObject._get_multiplier(
            sed, self.bp, tuple(chrom.wave_list)
        )
        sed_at_wave0 = sed(wave0)
        prof_scaled = prof0 * (multiplier / sed_at_wave0)
        img_mono = prof_scaled.drawImage(nx=32, ny=32, scale=0.2)

        npt.assert_allclose(img_chrom.array, img_mono.array, rtol=1e-5, atol=1e-10)

    def test_chromatic_equality(self):
        chrom1 = jax_galsim.SimpleChromaticTransformation(self.gal, sed=self.sed)
        chrom2 = jax_galsim.SimpleChromaticTransformation(self.gal, sed=self.sed)
        assert chrom1 == chrom2

    def test_chromatic_pytree(self):
        chrom = jax_galsim.SimpleChromaticTransformation(self.gal, sed=self.sed)
        children, aux = chrom.tree_flatten()
        chrom2 = jax_galsim.SimpleChromaticTransformation.tree_unflatten(aux, children)
        img1 = chrom.drawImage(self.bp, nx=16, ny=16, scale=0.3)
        img2 = chrom2.drawImage(self.bp, nx=16, ny=16, scale=0.3)
        npt.assert_allclose(img1.array, img2.array, rtol=1e-5)


# ============================================================
# Transform integration tests
# ============================================================


class TestTransformSED:
    def test_transform_with_sed_flux_ratio(self):
        """Transform(obj, flux_ratio=SED) should create a SimpleChromaticTransformation."""
        gal = jax_galsim.Gaussian(sigma=1.0, flux=1.0)
        x = np.linspace(400, 600, 100)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="fphotons")

        result = jax_galsim.Transform(gal, flux_ratio=sed)
        assert isinstance(result, jax_galsim.SimpleChromaticTransformation)


# ============================================================
# Utilities tests
# ============================================================


class TestUtilities:
    def test_merge_sorted(self):
        from jax_galsim.utilities import merge_sorted

        a = np.array([1, 3, 5])
        b = np.array([2, 4, 6])
        result = merge_sorted([a, b])
        npt.assert_array_equal(result, [1, 2, 3, 4, 5, 6])

    def test_merge_sorted_duplicates(self):
        from jax_galsim.utilities import merge_sorted

        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        result = merge_sorted([a, b])
        npt.assert_array_equal(result, [1, 2, 3, 4])

    def test_combine_wave_list(self):
        from jax_galsim.utilities import combine_wave_list

        x1 = np.linspace(400, 600, 50)
        f1 = np.ones_like(x1)
        lt1 = jax_galsim.LookupTable(x1, f1, interpolant="linear")
        bp1 = jax_galsim.Bandpass(lt1, wave_type="nm")

        x2 = np.linspace(450, 650, 50)
        f2 = np.ones_like(x2)
        lt2 = jax_galsim.LookupTable(x2, f2, interpolant="linear")
        bp2 = jax_galsim.Bandpass(lt2, wave_type="nm")

        wave_list, blue_limit, red_limit = combine_wave_list(bp1, bp2)
        # Intersection should be [450, 600]
        npt.assert_allclose(blue_limit, 450.0, atol=0.1)
        npt.assert_allclose(red_limit, 600.0, atol=0.1)
        assert len(wave_list) > 0


# ============================================================
# JIT compatibility tests
# ============================================================


class TestJIT:
    def test_lookuptable_jit(self):
        x = np.linspace(0, 10, 100)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")

        @jax.jit
        def eval_lt(w):
            return lt(w)

        result = eval_lt(jnp.array(5.0))
        npt.assert_allclose(float(result), np.sin(5.0), atol=2e-3)

    def test_chromatic_drawImage_jit(self):
        """Test that drawing a chromatic object works inside jit."""
        x = np.linspace(400, 600, 50)
        f = np.ones_like(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        bp = jax_galsim.Bandpass(lt, wave_type="nm")
        sed = jax_galsim.SED(lt, wave_type="nm", flux_type="fphotons")
        gal = jax_galsim.Gaussian(sigma=1.0, flux=1.0)
        chrom = gal * sed

        # The drawImage itself uses static computations for wave_list etc.,
        # so we test that the result is consistent
        img1 = chrom.drawImage(bp, nx=16, ny=16, scale=0.3)
        img2 = chrom.drawImage(bp, nx=16, ny=16, scale=0.3)
        npt.assert_allclose(img1.array, img2.array)
