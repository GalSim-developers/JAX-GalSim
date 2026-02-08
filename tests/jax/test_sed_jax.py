"""JAX-specific tests for SED, Bandpass, LookupTable, and ChromaticObject.

Tests JIT compilation, autodiff (grad), vmap, and pytree round-tripping.
API compliance and numerical precision are covered by the GalSim test suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import jax_galsim


def _make_bandpass():
    x = np.linspace(400, 600, 50)
    f = np.ones_like(x)
    lt = jax_galsim.LookupTable(x, f, interpolant="linear")
    return jax_galsim.Bandpass(lt, wave_type="nm")


def _make_sed():
    x = np.linspace(400, 600, 50)
    f = np.ones_like(x)
    lt = jax_galsim.LookupTable(x, f, interpolant="linear")
    return jax_galsim.SED(lt, wave_type="nm", flux_type="fphotons")


def _make_chromatic():
    gal = jax_galsim.Gaussian(sigma=1.0, flux=1.0)
    return gal * _make_sed()


# ============================================================
# PyTree round-tripping
# ============================================================


class TestPyTree:
    def test_lookuptable_pytree(self):
        x = np.linspace(0, 10, 50)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")
        children, aux = lt.tree_flatten()
        lt2 = jax_galsim.LookupTable.tree_unflatten(aux, children)
        npt.assert_allclose(float(lt(5.0)), float(lt2(5.0)))

    def test_lookuptable_spline_pytree(self):
        x = np.linspace(0, 10, 50)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="spline")
        children, aux = lt.tree_flatten()
        lt2 = jax_galsim.LookupTable.tree_unflatten(aux, children)
        npt.assert_allclose(float(lt(5.0)), float(lt2(5.0)))

    def test_bandpass_pytree(self):
        bp = _make_bandpass()
        children, aux = bp.tree_flatten()
        bp2 = jax_galsim.Bandpass.tree_unflatten(aux, children)
        npt.assert_allclose(float(bp(500)), float(bp2(500)))

    def test_sed_const_pytree(self):
        sed = jax_galsim.SED(2.0, wave_type="nm", flux_type="1")
        children, aux = sed.tree_flatten()
        sed2 = jax_galsim.SED.tree_unflatten(aux, children)
        npt.assert_allclose(float(sed(500.0)), float(sed2(500.0)))

    def test_sed_lookuptable_pytree(self):
        sed = _make_sed()
        children, aux = sed.tree_flatten()
        sed2 = jax_galsim.SED.tree_unflatten(aux, children)
        npt.assert_allclose(float(sed(500.0)), float(sed2(500.0)))

    def test_chromatic_pytree(self):
        bp = _make_bandpass()
        chrom = _make_chromatic()
        children, aux = chrom.tree_flatten()
        chrom2 = jax_galsim.SimpleChromaticTransformation.tree_unflatten(aux, children)
        img1 = chrom.drawImage(bp, nx=16, ny=16, scale=0.3)
        img2 = chrom2.drawImage(bp, nx=16, ny=16, scale=0.3)
        npt.assert_allclose(img1.array, img2.array, rtol=1e-5)


# ============================================================
# JIT compilation
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
        npt.assert_allclose(float(result), float(lt(5.0)))

    def test_lookuptable_spline_jit(self):
        x = np.linspace(0, 10, 100)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="spline")

        @jax.jit
        def eval_lt(w):
            return lt(w)

        result = eval_lt(jnp.array(5.0))
        npt.assert_allclose(float(result), float(lt(5.0)))

    def test_bandpass_jit(self):
        bp = _make_bandpass()

        @jax.jit
        def eval_bp(w):
            return bp(w)

        result = eval_bp(jnp.array(500.0))
        npt.assert_allclose(float(result), float(bp(500.0)))

    def test_sed_const_jit(self):
        sed = jax_galsim.SED(2.0, wave_type="nm", flux_type="1")

        @jax.jit
        def eval_sed(w):
            return sed(w)

        result = eval_sed(jnp.array(500.0))
        npt.assert_allclose(float(result), 2.0)

    def test_sed_lookuptable_jit(self):
        sed = _make_sed()

        @jax.jit
        def eval_sed(w):
            return sed(w)

        result = eval_sed(jnp.array(500.0))
        npt.assert_allclose(float(result), float(sed(500.0)))

    def test_chromatic_drawImage_deterministic(self):
        """Drawing a chromatic object produces consistent results."""
        bp = _make_bandpass()
        chrom = _make_chromatic()
        img1 = chrom.drawImage(bp, nx=16, ny=16, scale=0.3)
        img2 = chrom.drawImage(bp, nx=16, ny=16, scale=0.3)
        npt.assert_allclose(img1.array, img2.array)


# ============================================================
# Autodiff (grad)
# ============================================================


class TestGrad:
    def test_lookuptable_grad(self):
        """Gradient of LookupTable evaluation w.r.t. input wavelength."""
        x = np.linspace(0, 10, 100)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")

        grad_fn = jax.grad(lambda w: lt(w))
        g = grad_fn(jnp.array(5.0))
        # For linear interpolation, gradient should approximate cos(5)
        npt.assert_allclose(float(g), np.cos(5.0), atol=0.1)

    def test_lookuptable_spline_grad(self):
        """Gradient of spline LookupTable should approximate derivative."""
        x = np.linspace(0, 10, 100)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="spline")

        grad_fn = jax.grad(lambda w: lt(w))
        g = grad_fn(jnp.array(5.0))
        npt.assert_allclose(float(g), np.cos(5.0), atol=0.05)

    def test_bandpass_grad(self):
        """Gradient of Bandpass evaluation w.r.t. wavelength."""
        bp = _make_bandpass()
        grad_fn = jax.grad(lambda w: bp(w))
        # Inside the flat bandpass, gradient should be ~0
        g = grad_fn(jnp.array(500.0))
        npt.assert_allclose(float(g), 0.0, atol=1e-6)

    def test_sed_const_grad(self):
        """Gradient of constant SED w.r.t. wavelength should be 0."""
        sed = jax_galsim.SED(2.0, wave_type="nm", flux_type="1")
        grad_fn = jax.grad(lambda w: sed(w))
        g = grad_fn(jnp.array(500.0))
        npt.assert_allclose(float(g), 0.0, atol=1e-6)

    def test_sed_lookuptable_grad(self):
        """Gradient of tabulated SED w.r.t. wavelength."""
        sed = _make_sed()
        grad_fn = jax.grad(lambda w: sed(w))
        g = grad_fn(jnp.array(500.0))
        # Just check it returns a finite number
        assert jnp.isfinite(g)


# ============================================================
# vmap
# ============================================================


class TestVmap:
    def test_lookuptable_vmap(self):
        x = np.linspace(0, 10, 100)
        f = np.sin(x)
        lt = jax_galsim.LookupTable(x, f, interpolant="linear")

        eval_fn = jax.vmap(lambda w: lt(w))
        ws = jnp.array([1.0, 3.0, 5.0, 7.0])
        result = eval_fn(ws)
        expected = jnp.array([float(lt(w)) for w in ws])
        npt.assert_allclose(result, expected, rtol=1e-6)

    def test_bandpass_vmap(self):
        bp = _make_bandpass()
        eval_fn = jax.vmap(lambda w: bp(w))
        ws = jnp.array([300.0, 450.0, 500.0, 700.0])
        result = eval_fn(ws)
        expected = jnp.array([float(bp(w)) for w in ws])
        npt.assert_allclose(result, expected)

    def test_sed_vmap(self):
        sed = _make_sed()
        eval_fn = jax.vmap(lambda w: sed(w))
        ws = jnp.array([420.0, 460.0, 500.0, 540.0])
        result = eval_fn(ws)
        expected = jnp.array([float(sed(w)) for w in ws])
        npt.assert_allclose(result, expected, rtol=1e-6)
