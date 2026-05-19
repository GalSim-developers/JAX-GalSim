import jax
import jax.numpy as jnp
import pytest

import jax_galsim


def test_moffat_jax_beta_raises():

    @jax.jit
    def make_moffat(beta):
        return jax_galsim.Moffat(beta, fwhm=1.0)

    with pytest.raises(Exception):
        make_moffat(jnp.array(1.1))

    with pytest.raises(Exception):
        make_moffat(0.9)
