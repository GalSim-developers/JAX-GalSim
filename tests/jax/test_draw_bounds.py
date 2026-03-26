import jax
import numpy as np

import jax_galsim


def test_draw_bounds_center():

    @jax.jit
    def _draw(center):
        return jax_galsim.Gaussian(
            fwhm=1.5,
            gsparams=jax_galsim.GSParams(minimum_fft_size=1024, maximum_fft_size=1024),
        ).drawImage(nx=52, ny=52, center=center, scale=0.2)

    img = _draw(jax_galsim.PositionD(5.7, -2.1))
    print(img)


def test_draw_bounds_center_jit():

    @jax.jit
    def _draw(center):
        return jax_galsim.Gaussian(
            fwhm=1.5,
            gsparams=jax_galsim.GSParams(minimum_fft_size=1024, maximum_fft_size=1024),
        ).drawImage(nx=52, ny=52, center=center, scale=0.2)

    img = _draw(jax_galsim.PositionD(5.7, -2.1))
    print(img)


def test_draw_bounds_center_jit_vmap():

    @jax.jit
    def _draw(center):
        return jax_galsim.Gaussian(
            fwhm=1.5,
            gsparams=jax_galsim.GSParams(minimum_fft_size=1024, maximum_fft_size=1024),
        ).drawImage(nx=52, ny=52, center=center, scale=0.2)

    ng = 7
    rng = np.random.default_rng(seed=10)
    pos_x = rng.uniform(low=-10, high=10, size=ng)
    pos_y = rng.uniform(low=-10, high=10, size=ng)
    pos = jax.vmap(lambda x, y: jax_galsim.PositionD(x, y))(pos_x, pos_y)
    img = jax.jit(jax.vmap(_draw))(pos)
    assert img.array.shape == (ng, 52, 52)
    for i in range(ng):
        for j in range(i + 1, ng):
            assert not np.array_equal(img.array[i, ...], img.array[j, ...])
