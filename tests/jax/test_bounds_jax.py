import jax
import jax.numpy as jnp
import numpy as np

import jax_galsim


@jax.vmap
@jax.jit
def _make_bounds_float(xmin, ymin, xmax, ymax):
    bnds = jax_galsim.BoundsD(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return bnds, bnds.isDefined()


def test_bounds_jax_vmap_isdefined_float():
    xmin = jnp.array([9, 10, 11, 12])
    xmax = jnp.array([12, 11, 10, 9])
    ymin = jnp.array([9, 11, 10, 12])
    ymax = jnp.array([10, 10, 10, 10])
    bnds, isdef = _make_bounds_float(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)


@jax.vmap
@jax.jit
def _and_bounds_empty_float(bnds):
    bnds = bnds & jax_galsim.BoundsD()
    return bnds, bnds.isDefined()


@jax.vmap
@jax.jit
def _and_bounds_float(bnds):
    bnds = bnds & jax_galsim.BoundsD(xmin=10, xmax=11, ymin=10, ymax=11)
    return bnds, bnds.isDefined()


@jax.vmap
@jax.jit
def _and_bounds_far_away_float(bnds):
    bnds = bnds & jax_galsim.BoundsD(xmin=100, xmax=110, ymin=100, ymax=110)
    return bnds, bnds.isDefined()


def test_bounds_jax_vmap_and_isdefined_float():
    xmin = jnp.array([9, 10, 11, 12])
    xmax = jnp.array([12, 11, 10, 9])
    ymin = jnp.array([9, 11, 10, 12])
    ymax = jnp.array([10, 10, 10, 10])

    bnds, isdef = _make_bounds_float(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    bnds, isdef = _and_bounds_empty_float(bnds)
    assert bnds.isDefined().shape == (4,)
    assert not jnp.any(bnds.isDefined())
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    np.testing.assert_array_equal(bnds.isDefined(), False)

    bnds, isdef = _make_bounds_float(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    bnds, isdef = _and_bounds_float(bnds)
    assert bnds.isDefined().shape == (4,)
    np.testing.assert_array_equal(
        bnds.isDefined(), jnp.array([True, False, False, False])
    )
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    assert bnds.xmin[0] == 10
    assert bnds.xmax[0] == 11
    assert bnds.ymin[0] == 10
    assert bnds.ymax[0] == 10

    bnds, isdef = _make_bounds_float(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    bnds, isdef = _and_bounds_far_away_float(bnds)
    assert bnds.isDefined().shape == (4,)
    assert not jnp.any(bnds.isDefined())
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    np.testing.assert_array_equal(bnds.isDefined(), False)


@jax.vmap
@jax.jit
def _plus_bounds_far_away_float(bnds):
    bnds = bnds + jax_galsim.BoundsD(xmin=100, xmax=110, ymin=100, ymax=110)
    return bnds, bnds.isDefined()


def test_bounds_jax_vmap_plus_float():
    xmin = jnp.array([9, 10, 11, 12])
    xmax = jnp.array([12, 11, 10, 9])
    ymin = jnp.array([9, 11, 10, 12])
    ymax = jnp.array([10, 10, 10, 10])

    bnds, isdef = _make_bounds_float(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    bnds, isdef = _plus_bounds_far_away_float(bnds)
    assert bnds.isDefined().shape == (4,)
    np.testing.assert_array_equal(bnds.isDefined(), True)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    assert bnds.xmin[0] == 9
    assert bnds.xmax[0] == 110
    assert bnds.ymin[0] == 9
    assert bnds.ymax[0] == 110

    np.testing.assert_array_equal(bnds.xmin[1:], 100)
    np.testing.assert_array_equal(bnds.xmax[1:], 110)
    np.testing.assert_array_equal(bnds.ymin[1:], 100)
    np.testing.assert_array_equal(bnds.ymax[1:], 110)
