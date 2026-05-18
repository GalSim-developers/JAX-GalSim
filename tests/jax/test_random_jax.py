import jax
import pytest

import jax_galsim


def test_random_jax_gaussian_pos_sigma_jit():
    @jax.jit
    def _make_gauss(sigma):
        return jax_galsim.GaussianDeviate(seed=10, sigma=sigma)

    with pytest.raises(Exception):
        _make_gauss(-1.0)

    @jax.jit
    def _make_gauss_again(sigma):
        return jax_galsim.GaussianDeviate(seed=10, sigma=sigma)

    _make_gauss_again(1.0)

    with pytest.raises(Exception):
        _make_gauss_again(-1)

    def _make_gauss_again_again(sigma):
        return jax_galsim.GaussianDeviate(seed=10, sigma=sigma)

    _make_gauss_again_again(1.0)

    with pytest.raises(Exception):
        jax.jit(_make_gauss_again_again)(-1)
