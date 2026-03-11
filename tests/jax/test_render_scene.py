from functools import partial

import jax
import jax.random as jrng

import jax_galsim as jgs


def _generate_stamp(rng_key, psf):
    rng_key, use_key = jrng.split(rng_key)
    flux = jrng.uniform(use_key, minval=1.5, maxval=2.5)
    rng_key, use_key = jrng.split(rng_key)
    hlr = jrng.uniform(use_key, minval=0.5, maxval=2.5)
    rng_key, use_key = jrng.split(rng_key)
    g1 = jrng.uniform(use_key, minval=-0.1, maxval=0.1)
    rng_key, use_key = jrng.split(rng_key)
    g2 = jrng.uniform(use_key, minval=-0.1, maxval=0.1)

    rng_key, use_key = jrng.split(rng_key)
    dx = jrng.uniform(use_key, minval=-10, maxval=10)
    rng_key, use_key = jrng.split(rng_key)
    dy = jrng.uniform(use_key, minval=-10, maxval=10)

    return (
        jgs.Convolve(
            [
                jgs.Exponential(half_light_radius=hlr)
                .shear(g1=g1, g2=g2)
                .shift(dx, dy)
                .withFlux(flux),
                psf,
            ]
        )
        .withGSParams(minimum_fft_size=1024, maximum_fft_size=1024)
        .drawImage(nx=200, ny=200, scale=0.2)
    )


@partial(jax.jit, static_argnames=("n_obj"))
def _generate_stamps(rng_key, psf, n_obj):
    use_keys = jrng.split(rng_key, num=n_obj + 1)
    rng_key = use_keys[0]
    use_keys = use_keys[1:]

    return jax.vmap(_generate_stamp, in_axes=(0, None))(use_keys, psf)


def test_render_scene_draw_many_ffts_full_img():
    psf = jgs.Gaussian(fwhm=0.9)
    img = _generate_stamps(jrng.key(42), psf, 5)

    assert img.array.shape == (5, 200, 200)
    assert img.array.sum() > 5.0
