import jax
import jax.numpy as jnp
import numpy as np

import jax_galsim


@jax.vmap
@jax.jit
def _make_bounds(xmin, ymin):
    bds = jax_galsim.BoundsI(xmin=xmin, ymin=ymin, deltax=10, deltay=10)
    return bds, bds.isDefined()


def test_bounds_jax_vmap_isdefined():
    xmin = jnp.array([9, 10, 11])
    ymin = jnp.array([9, 10, 11])

    bds, isdef = _make_bounds(xmin, ymin)
    print(isdef, bds.isDefined())
    np.testing.assert_array_equal(bds.isDefined(), isdef, strict=True)
