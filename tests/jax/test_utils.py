import jax.numpy as jnp
import numpy as np
import pytest

from jax_galsim.core.utils import cast_numpy_array_to_native_byte_order


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(10),
        np.arange(10, dtype="<i4"),
        np.arange(10, dtype=">f8"),
        jnp.arange(10),
    ],
)
def test_cast_numpy_array_to_native_byte_order(arr):
    output = cast_numpy_array_to_native_byte_order(arr)
    assert output.dtype.isnative
    np.testing.assert_array_equal(arr, output)
