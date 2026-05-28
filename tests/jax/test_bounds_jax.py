from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


@jax.vmap
@jax.jit
def _plus_bounds_pos_far_away_float(bnds):
    bnds = bnds + jax_galsim.PositionD(x=100, y=110)
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

    bnds, isdef = _make_bounds_float(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    bnds, isdef = _plus_bounds_pos_far_away_float(bnds)
    assert bnds.isDefined().shape == (4,)
    np.testing.assert_array_equal(bnds.isDefined(), True)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)
    assert bnds.xmin[0] == 9
    assert bnds.xmax[0] == 100
    assert bnds.ymin[0] == 9
    assert bnds.ymax[0] == 110

    np.testing.assert_array_equal(bnds.xmin[1:], 100)
    np.testing.assert_array_equal(bnds.xmax[1:], 100)
    np.testing.assert_array_equal(bnds.ymin[1:], 110)
    np.testing.assert_array_equal(bnds.ymax[1:], 110)


@jax.vmap
@jax.jit
def _make_bounds_int(xmin, ymin):
    bnds = jax_galsim.BoundsI(xmin=xmin, ymin=ymin, deltax=10, deltay=11)
    return bnds, bnds.isDefined()


def test_bounds_jax_vmap_isdefined_int():
    xmin = jnp.array([9, 10, 11, 12])
    ymin = jnp.array([9, 11, 10, 12])
    bnds, isdef = _make_bounds_int(xmin, ymin)
    np.testing.assert_array_equal(bnds.isDefined(), isdef[0], strict=True)
    np.testing.assert_array_equal(bnds.isDefined(), True)
    assert jnp.all(isdef)
    np.testing.assert_array_equal(bnds.xmin, xmin, strict=True)
    np.testing.assert_array_equal(bnds.ymin, ymin, strict=True)
    assert isinstance(bnds.deltax, int)
    assert bnds.deltax == 10
    assert isinstance(bnds.deltay, int)
    assert bnds.deltay == 11


@jax.vmap
@jax.jit
def _make_bounds_int_bad(xmin, ymin, delta):
    bnds = jax_galsim.BoundsI(xmin=xmin, ymin=ymin, deltax=delta, deltay=delta)
    return bnds, bnds.isDefined()


def test_bounds_jax_vmap_varying_shape_raises_int():
    xmin = jnp.array([9, 10, 11, 12])
    ymin = jnp.array([9, 11, 10, 12])
    delta = jnp.array([9, 11, 10, 12])
    with pytest.raises(Exception):
        _make_bounds_int_bad(xmin, ymin, delta)


@jax.vmap
@jax.jit
def _and_bounds_empty_int(bnds):
    bnds = bnds & jax_galsim.BoundsI()
    return bnds, bnds.isDefined()


def test_bounds_jax_vmap_and_raises_isdefined_int():
    xmin = jnp.array([9, 10, 11, 12])
    ymin = jnp.array([9, 11, 10, 12])
    bnds, isdef = _make_bounds_int(xmin, ymin)
    np.testing.assert_array_equal(bnds.isDefined(), isdef[0], strict=True)
    np.testing.assert_array_equal(bnds.isDefined(), True)
    assert jnp.all(isdef)

    with pytest.raises(Exception):
        _and_bounds_empty_int(bnds)


@jax.vmap
@jax.jit
def _plus_bounds_far_away_int(bnds):
    bnds = bnds + jax_galsim.BoundsI(xmin=100, deltax=110, ymin=100, deltay=110)
    return bnds, bnds.isDefined()


@jax.vmap
@jax.jit
def _plus_bounds_pos_far_away_int(bnds):
    bnds = bnds + jax_galsim.PositionI(x=100, y=110)
    return bnds, bnds.isDefined()


def test_bounds_jax_vmap_plus_raises_int():
    xmin = jnp.array([9, 10, 11, 12])
    ymin = jnp.array([9, 11, 10, 12])
    bnds, isdef = _make_bounds_int(xmin, ymin)
    np.testing.assert_array_equal(bnds.isDefined(), isdef[0], strict=True)
    np.testing.assert_array_equal(bnds.isDefined(), True)
    assert jnp.all(isdef)

    with pytest.raises(Exception):
        _plus_bounds_far_away_int(bnds)

    with pytest.raises(Exception):
        _plus_bounds_pos_far_away_float(bnds)


def test_bounds_jax_int_set_static():
    bnds = jax_galsim.BoundsI(xmin=1, ymin=1, deltax=10, deltay=11)

    bnds.xmin = 11.0
    assert isinstance(bnds.xmin, int)
    assert bnds.xmin == 11
    bnds.xmin = jnp.array(12, dtype=float)
    assert isinstance(bnds.xmin, int)
    assert bnds.xmin == 12

    bnds.ymin = 12.0
    assert isinstance(bnds.ymin, int)
    assert bnds.ymin == 12
    bnds.ymin = jnp.array(13, dtype=float)
    assert isinstance(bnds.ymin, int)
    assert bnds.ymin == 13

    bnds.deltax = 11.0
    assert isinstance(bnds.deltax, int)
    assert bnds.deltax == 11
    bnds.deltax = jnp.array(12, dtype=float)
    assert isinstance(bnds.deltax, int)
    assert bnds.deltax == 12

    bnds.deltay = 12.0
    assert isinstance(bnds.deltay, int)
    assert bnds.deltay == 12
    bnds.deltay = jnp.array(13, dtype=float)
    assert isinstance(bnds.deltay, int)
    assert bnds.deltay == 13


def test_bounds_jax_int_set_dynamic():
    bnds = jax_galsim.BoundsI(
        xmin=jnp.array(1), ymin=jnp.array(2), deltax=10, deltay=11
    )

    bnds.xmin = 11.0
    assert isinstance(bnds.xmin, jnp.ndarray)
    assert bnds.xmin == 11
    bnds.xmin = jnp.array(12, dtype=float)
    assert isinstance(bnds.xmin, jnp.ndarray)
    assert bnds.xmin == 12

    bnds.ymin = 12.0
    assert isinstance(bnds.ymin, jnp.ndarray)
    assert bnds.ymin == 12
    bnds.ymin = jnp.array(13, dtype=float)
    assert isinstance(bnds.ymin, jnp.ndarray)
    assert bnds.ymin == 13

    bnds.deltax = 11.0
    assert isinstance(bnds.deltax, int)
    assert bnds.deltax == 11
    bnds.deltax = jnp.array(12, dtype=float)
    assert isinstance(bnds.deltax, int)
    assert bnds.deltax == 12

    bnds.deltay = 12.0
    assert isinstance(bnds.deltay, int)
    assert bnds.deltay == 12
    bnds.deltay = jnp.array(13, dtype=float)
    assert isinstance(bnds.deltay, int)
    assert bnds.deltay == 13


def test_bounds_jax_int_set_jit_raises():
    @jax.jit
    def _make_bnds_bad_xmin(xmin):
        bnds = jax_galsim.BoundsI(xmin=1, ymin=1, deltax=10, deltay=11)
        bnds.xmin = xmin
        return bnds

    @jax.jit
    def _make_bnds_bad_ymin(ymin):
        bnds = jax_galsim.BoundsI(xmin=1, ymin=1, deltax=10, deltay=11)
        bnds.ymin = ymin
        return bnds

    with pytest.raises(Exception):
        _make_bnds_bad_xmin(2)

    with pytest.raises(Exception):
        _make_bnds_bad_ymin(2)

    @jax.jit
    def _make_bnds_bad_deltax(deltax):
        bnds = jax_galsim.BoundsI(xmin=1, ymin=1, deltax=10, deltay=11)
        bnds.deltax = deltax
        return bnds

    @jax.jit
    def _make_bnds_bad_deltay(deltay):
        bnds = jax_galsim.BoundsI(xmin=1, ymin=1, deltax=10, deltay=11)
        bnds.deltay = deltay
        return bnds

    with pytest.raises(Exception):
        _make_bnds_bad_deltay(2)

    with pytest.raises(Exception):
        _make_bnds_bad_deltay(2)


def test_bounds_jax_int_set_jit():
    @jax.jit
    def _make_bnds_set_xmin(xmin):
        bnds = jax_galsim.BoundsI(xmin=jnp.array(1), ymin=1, deltax=10, deltay=11)
        bnds.xmin = xmin
        return bnds

    @jax.jit
    def _make_bnds_set_ymin(ymin):
        bnds = jax_galsim.BoundsI(xmin=jnp.array(1), ymin=1, deltax=10, deltay=11)
        bnds.ymin = ymin
        return bnds

    bnds = _make_bnds_set_xmin(2)
    assert isinstance(bnds.ymin, jnp.ndarray)
    assert bnds.xmin == 2
    assert isinstance(bnds.xmin, jnp.ndarray)

    bnds = _make_bnds_set_ymin(2)
    assert isinstance(bnds.ymin, jnp.ndarray)
    assert bnds.ymin == 2
    assert isinstance(bnds.ymin, jnp.ndarray)

    @partial(jax.jit, static_argnames=("xmin",))
    def _make_bnds_set_xmin_static(xmin):
        bnds = jax_galsim.BoundsI(xmin=1, ymin=1, deltax=10, deltay=11)
        bnds.xmin = xmin
        return bnds

    bnds = _make_bnds_set_xmin_static(2)
    assert isinstance(bnds.xmin, int)
    assert isinstance(bnds.ymin, int)
    assert bnds.xmin == 2


@jax.vmap
@jax.jit
def _make_bounds_int_nodef(xmin, ymin):
    bnds = jax_galsim.BoundsI(xmin=xmin, ymin=ymin, deltax=0, deltay=0)
    return bnds, bnds.isDefined()


@jax.vmap
@jax.jit
def _bounds_includes_bounds_float(bnds):
    return bnds.includes(jax_galsim.BoundsD(9.5, 9.75, 9.5, 9.75))


@jax.vmap
@jax.jit
def _bounds_includes_bounds_nodef_float(bnds):
    return bnds.includes(jax_galsim.BoundsD())


@jax.vmap
@jax.jit
def _bounds_includes_pos_float(bnds):
    return bnds.includes(jax_galsim.PositionD(9.5, 9.5))


@jax.vmap
@jax.jit
def _bounds_includes_xy_float(bnds):
    return bnds.includes(9.5, 9.5)


def test_bounds_jax_vmap_includes_float():
    xmin = jnp.array([8, 9, 10, 11, 12])
    xmax = jnp.array([9, 12, 11, 10, 9])
    ymin = jnp.array([7, 9, 11, 10, 12])
    ymax = jnp.array([8, 10, 10, 10, 10])
    bnds, isdef = _make_bounds_float(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bnds.isDefined(), isdef, strict=True)

    incs = _bounds_includes_bounds_float(bnds)
    np.testing.assert_array_equal(incs, [False, True, False, False, False], strict=True)

    incs = _bounds_includes_bounds_nodef_float(bnds)
    np.testing.assert_array_equal(
        incs, [False, False, False, False, False], strict=True
    )

    incs = _bounds_includes_pos_float(bnds)
    np.testing.assert_array_equal(incs, [False, True, False, False, False], strict=True)

    incs = _bounds_includes_xy_float(bnds)
    np.testing.assert_array_equal(incs, [False, True, False, False, False], strict=True)


@jax.vmap
@jax.jit
def _bounds_includes_bounds_int(bnds):
    return bnds.includes(jax_galsim.BoundsI(9, 9, 10, 10))


@jax.vmap
@jax.jit
def _bounds_includes_bounds_nodef_int(bnds):
    return bnds.includes(jax_galsim.BoundsI())


@jax.vmap
@jax.jit
def _bounds_includes_pos_int(bnds):
    return bnds.includes(jax_galsim.PositionD(9, 10))


@jax.vmap
@jax.jit
def _bounds_includes_xy_int(bnds):
    return bnds.includes(9.5, 9.7)


def test_bounds_jax_vmap_includes_int():
    xmin = jnp.array([8, 9, 10, 11, 12])
    ymin = jnp.array([7, 9, 11, 10, 12])

    bnds, isdef = _make_bounds_int(xmin, ymin)
    np.testing.assert_array_equal(bnds.isDefined(), isdef)

    incs = _bounds_includes_bounds_int(bnds)
    np.testing.assert_array_equal(incs, [True, True, False, False, False], strict=True)

    incs = _bounds_includes_bounds_nodef_int(bnds)
    np.testing.assert_array_equal(
        incs, [False, False, False, False, False], strict=True
    )

    incs = _bounds_includes_pos_int(bnds)
    np.testing.assert_array_equal(incs, [True, True, False, False, False], strict=True)

    incs = _bounds_includes_xy_int(bnds)
    np.testing.assert_array_equal(incs, [True, True, False, False, False], strict=True)

    bnds, isdef = _make_bounds_int_nodef(xmin, ymin)
    np.testing.assert_array_equal(bnds.isDefined(), isdef)

    incs = _bounds_includes_bounds_int(bnds)
    np.testing.assert_array_equal(
        incs, [False, False, False, False, False], strict=True
    )

    incs = _bounds_includes_bounds_nodef_int(bnds)
    np.testing.assert_array_equal(
        incs, [False, False, False, False, False], strict=True
    )

    incs = _bounds_includes_pos_int(bnds)
    np.testing.assert_array_equal(
        incs, [False, False, False, False, False], strict=True
    )

    incs = _bounds_includes_xy_int(bnds)
    np.testing.assert_array_equal(
        incs, [False, False, False, False, False], strict=True
    )


def test_bounds_jax_init_bounds():
    assert not jax_galsim.BoundsD().isDefined()
    assert not jax_galsim.BoundsD(jax_galsim.BoundsD()).isDefined()

    assert not jax_galsim.BoundsI().isDefined()
    assert not jax_galsim.BoundsI(jax_galsim.BoundsI()).isDefined()
