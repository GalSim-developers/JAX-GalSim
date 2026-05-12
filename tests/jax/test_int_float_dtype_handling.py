import galsim as _galsim
import jax.numpy as jnp
import numpy as np

import jax_galsim


def test_int_float_dtype_handling_invertSelf():
    gim = _galsim.Image(np.arange(20).reshape(4, 5), dtype=int)
    gim.invertSelf()

    assert gim[1, 1] == 0
    assert gim[2, 1] == 1
    assert gim[4, 4] == 0

    jgim = jax_galsim.Image(jnp.arange(20).reshape(4, 5), dtype=int)
    jgim.invertSelf()

    assert jgim[1, 1] == 0
    assert jgim[2, 1] == 1
    assert jgim[4, 4] == 0

    np.testing.assert_array_equal(gim.array, jgim.array)
