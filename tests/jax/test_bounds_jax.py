import jax
import jax.numpy as jnp
import numpy as np

import jax_galsim


@jax.vmap
@jax.jit
def _make_bounds_int(xmin, ymin, xmax, ymax):
    bds = jax_galsim.BoundsI(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return bds, bds.isDefined()


def test_bounds_jax_vmap_isdefined_int():
    xmin = jnp.array([9, 10, 11, 12])
    xmax = jnp.array([12, 11, 10, 9])
    ymin = jnp.array([9, 11, 10, 12])
    ymax = jnp.array([10, 10, 11, 10])
    bds, isdef = _make_bounds_int(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bds.isDefined(), isdef, strict=True)

    # turn a bounds of arrays into a list of bounds
    # see https://github.com/jax-ml/jax/discussions/35711
    list_of_bnds = jax.tree.transpose(
        jax.tree.structure(bds),
        None,
        jax.tree.map(list, bds)
    )
    assert list_of_bnds[0] != list_of_bnds[2]
    assert list_of_bnds[1] == list_of_bnds[2]
    assert list_of_bnds[2] == list_of_bnds[3]
    assert all(not bnds.isStatic() for bnds in list_of_bnds)


@jax.vmap
@jax.jit
def _make_bounds_float(xmin, ymin, xmax, ymax):
    bds = jax_galsim.BoundsD(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return bds, bds.isDefined()


def test_bounds_jax_vmap_isdefined_float():
    xmin = jnp.array([9, 10, 11, 12])
    xmax = jnp.array([12, 11, 10, 9])
    ymin = jnp.array([9, 11, 10, 12])
    ymax = jnp.array([10, 10, 10, 10])
    bds, isdef = _make_bounds_float(xmin, ymin, xmax, ymax)
    np.testing.assert_array_equal(bds.isDefined(), isdef, strict=True)
