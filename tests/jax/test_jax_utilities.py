import jax.numpy as jnp
import numpy as np

import jax_galsim


def test_jax_utilities_horner_dtype():
    rng = np.random.RandomState(1234)
    x = rng.uniform(size=20)
    coef = [1.2332, 3.43242, 4.1231, -0.2342, 0.4242]
    truth = coef[0] + coef[1] * x + coef[2] * x**2 + coef[3] * x**3 + coef[4] * x**4

    result = jax_galsim.utilities.horner(x, coef, dtype=int)
    np.testing.assert_almost_equal(result, truth.astype(int))

    result = jax_galsim.utilities.horner(x, coef, dtype=float)
    np.testing.assert_almost_equal(result, truth)

    result = jax_galsim.utilities.horner(x, coef, dtype=complex)
    np.testing.assert_almost_equal(result.real, truth)
    np.testing.assert_almost_equal(result.imag, np.zeros_like(truth))

    result = jax_galsim.utilities.horner(x, coef, dtype=jnp.int32)
    assert result.dtype == jnp.int32
    np.testing.assert_almost_equal(result, truth.astype(np.int32))

    result = jax_galsim.utilities.horner(x, coef, dtype=jnp.int64)
    assert result.dtype == jnp.int64
    np.testing.assert_almost_equal(result, truth.astype(np.int64))

    result = jax_galsim.utilities.horner(x, coef, dtype=jnp.float32)
    assert result.dtype == jnp.float32
    np.testing.assert_almost_equal(result, truth.astype(np.float32))

    result = jax_galsim.utilities.horner(x, coef, dtype=jnp.float64)
    assert result.dtype == jnp.float64
    np.testing.assert_almost_equal(result, truth)

    result = jax_galsim.utilities.horner(x, coef, dtype=jnp.complex64)
    assert result.dtype == jnp.complex64
    np.testing.assert_almost_equal(result.real, truth.astype(np.complex64))
    np.testing.assert_almost_equal(result.imag, np.zeros_like(truth))

    result = jax_galsim.utilities.horner(x, coef, dtype=jnp.complex128)
    assert result.dtype == jnp.complex128
    np.testing.assert_almost_equal(result.real, truth)
    np.testing.assert_almost_equal(result.imag, np.zeros_like(truth))
