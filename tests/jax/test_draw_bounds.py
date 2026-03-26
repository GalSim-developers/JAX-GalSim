import jax
import numpy as np

import jax_galsim


def test_draw_bounds_center():

    def _draw(center, flux):
        return jax_galsim.Gaussian(
            fwhm=1.5,
            flux=flux,
            gsparams=jax_galsim.GSParams(minimum_fft_size=1024, maximum_fft_size=1024),
        ).drawImage(nx=52, ny=52, center=center, scale=0.2)

    img = _draw(jax_galsim.PositionD(5.7, -2.1), 10)
    np.testing.assert_allclose(img.array.sum(), 10, rtol=1e-5, atol=1e-5)
    assert img.bounds.xmin != 1
    assert img.bounds.ymin != 1


def test_draw_bounds_center_jit():

    @jax.jit
    def _draw(center, flux):
        return jax_galsim.Gaussian(
            fwhm=1.5,
            flux=flux,
            gsparams=jax_galsim.GSParams(minimum_fft_size=1024, maximum_fft_size=1024),
        ).drawImage(nx=52, ny=52, center=center, scale=0.2)

    img = _draw(jax_galsim.PositionD(5.7, -2.1), 10)
    np.testing.assert_allclose(img.array.sum(), 10, rtol=1e-5, atol=1e-5)
    assert img.bounds.xmin != 1
    assert img.bounds.ymin != 1


def test_draw_bounds_center_jit_vmap():

    @jax.jit
    def _draw(center, flux):
        return jax_galsim.Gaussian(
            fwhm=1.5,
            flux=flux,
            gsparams=jax_galsim.GSParams(minimum_fft_size=1024, maximum_fft_size=1024),
        ).drawImage(nx=101, ny=101, center=center, scale=0.2)

    ng = 7
    rng = np.random.default_rng(seed=10)
    pos_x = rng.uniform(low=-10, high=10, size=ng)
    pos_y = rng.uniform(low=-10, high=10, size=ng)
    flux = rng.uniform(low=1, high=10, size=ng)
    pos = jax.vmap(lambda x, y: jax_galsim.PositionD(x, y))(pos_x, pos_y)
    img = jax.jit(jax.vmap(_draw))(pos, flux)
    assert img.array.shape == (ng, 101, 101)
    assert not any(xmin == 1 for xmin in img.bounds.xmin)
    assert not any(ymin == 1 for ymin in img.bounds.ymin)
    for i in range(ng):
        for j in range(i + 1, ng):
            assert not np.array_equal(img.array[i, ...], img.array[j, ...])

    fluxes = img.array.sum(axis=(1, 2))
    np.testing.assert_allclose(fluxes, flux, rtol=1e-5, atol=1e-5)
