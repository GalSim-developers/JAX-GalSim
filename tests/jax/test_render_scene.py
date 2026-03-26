from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import pytest

import jax_galsim as jgs


def _generate_image_one(rng_key, psf):
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
def _generate_image(rng_key, psf, n_obj):
    use_keys = jrng.split(rng_key, num=n_obj + 1)
    rng_key = use_keys[0]
    use_keys = use_keys[1:]

    return jax.vmap(_generate_image_one, in_axes=(0, None))(use_keys, psf)


def test_render_scene_draw_many_ffts_full_img():
    psf = jgs.Gaussian(fwhm=0.9)
    img = _generate_image(jrng.key(10), psf, 5)

    if False:
        import pdb

        import matplotlib.pyplot as plt

        plt.imshow(img.array.sum(axis=0))
        pdb.set_trace()

    assert img.array.shape == (5, 200, 200)
    assert img.array.sum() > 5.0


def _get_bd_jgs(
    flux_d,
    flux_b,
    hlr_b,
    hlr_d,
    q_b,
    q_d,
    beta,
    *,
    psf_hlr=0.7,
):
    components = []

    # disk
    disk = jgs.Exponential(flux=flux_d, half_light_radius=hlr_d).shear(
        q=q_d, beta=beta * jgs.degrees
    )
    components.append(disk)

    # bulge
    bulge = jgs.Spergel(nu=-0.6, flux=flux_b, half_light_radius=hlr_b).shear(
        q=q_b, beta=beta * jgs.degrees
    )
    components.append(bulge)

    galaxy = jgs.Add(components)

    # psf
    psf = jgs.Moffat(2, flux=1.0, half_light_radius=0.7)

    gal_conv = jgs.Convolve([galaxy, psf])
    return gal_conv


@partial(jax.jit, static_argnames=("fft_size", "slen"))
def _draw_stamp_jgs(
    galaxy_params: dict,
    image_pos: jgs.PositionD,
    local_wcs: jgs.PixelScale,
    fft_size: int,
    slen: int,
) -> jax.Array:
    gsparams = jgs.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)

    convolved_object = _get_bd_jgs(**galaxy_params).withGSParams(gsparams)

    stamp = convolved_object.drawImage(
        nx=slen,
        ny=slen,
        center=image_pos,
        wcs=local_wcs,
        dtype=jnp.float64,
    )

    return stamp


@partial(jax.jit, static_argnames=("slen",))
def _add_to_image(carry, x, slen):
    image = carry[0]
    stamp = x

    image[stamp.bounds] += stamp

    return (image,), None


@partial(jax.jit, static_argnames=("fft_size", "slen", "ilen", "ng"))
def _render_scene_stamps_jax_galsim(
    galaxy_params: dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    fft_size: int,
    slen: int,
    ilen: int,
    ng: int,
):
    image = jgs.Image(ncol=ilen, nrow=ilen, scale=0.2, dtype=jnp.float64)
    wcs = image.wcs

    image_positions = jax.vmap(lambda x, y: jgs.PositionD(x=x, y=y))(x, y)
    local_wcss = jax.vmap(lambda x: wcs.local(image_pos=x))(image_positions)

    stamps = jax.jit(jax.vmap(partial(_draw_stamp_jgs, slen=slen, fft_size=fft_size)))(
        galaxy_params, image_positions, local_wcss
    )

    pad_image = jgs.ImageD(
        jnp.pad(image.array, slen), wcs=image.wcs, bounds=image.bounds.withBorder(slen)
    )

    final_pad_image = jax.lax.scan(
        partial(_add_to_image, slen=slen),
        (pad_image,),
        xs=stamps,
        length=ng,
    )[0][0]

    return stamps, final_pad_image


def _get_bd_gs(
    flux_d,
    flux_b,
    hlr_b,
    hlr_d,
    q_b,
    q_d,
    beta,
    *,
    psf_hlr=0.7,
):
    components = []

    # disk
    disk = _galsim.Exponential(flux=flux_d, half_light_radius=hlr_d).shear(
        q=q_d, beta=beta * _galsim.degrees
    )
    components.append(disk)

    # bulge
    bulge = _galsim.Spergel(nu=-0.6, flux=flux_b, half_light_radius=hlr_b).shear(
        q=q_b, beta=beta * _galsim.degrees
    )
    components.append(bulge)

    galaxy = _galsim.Add(components)

    # psf
    psf = _galsim.Moffat(2, flux=1.0, half_light_radius=0.7)

    gal_conv = _galsim.Convolve([galaxy, psf])
    return gal_conv


def _render_scene_stamps_galsim(
    galaxy_params: dict,
    x: np.ndarray,
    y: np.ndarray,
    fft_size: int,
    slen: int,
    ilen: int,
    ng: int,
):
    image = _galsim.Image(ncol=ilen, nrow=ilen, scale=0.2, dtype=np.float64)
    wcs = image.wcs

    image_pos = list(
        map(lambda tup: _galsim.PositionD(x=tup[0], y=tup[1]), zip(x, y))
    )
    local_wcs = list(map(lambda x: wcs.local(image_pos=x), image_pos))

    gsparams = _galsim.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)

    for i in range(ng):
        gpars = {k: v[i] for k, v in galaxy_params.items()}
        convolved_object = _get_bd_gs(**gpars).withGSParams(gsparams)

        stamp = convolved_object.drawImage(
            nx=slen,
            ny=slen,
            center=(image_pos[i].x, image_pos[i].y),
            wcs=local_wcs[i],
            dtype=np.float64,
        )

        b = stamp.bounds & image.bounds
        if b.isDefined():
            image[b] += stamp[b]

    return image


@pytest.mark.parametrize("slen", [51, 52])
def test_render_scene_stamps(slen):
    rng = np.random.default_rng(seed=10)
    ng = 5
    fft_size = 2048
    ilen = 200

    galaxy_params = {
        "flux_d": rng.uniform(low=0, high=1.0, size=ng),
        "flux_b": rng.uniform(low=0, high=1.0, size=ng),
        "hlr_b": rng.uniform(low=0.3, high=0.5, size=ng),
        "hlr_d": rng.uniform(low=0.5, high=0.7, size=ng),
        "q_b": rng.uniform(low=0.1, high=0.9, size=ng),
        "q_d": rng.uniform(low=0.1, high=0.9, size=ng),
        "beta": rng.uniform(low=0, high=360, size=ng),
        "x": rng.uniform(low=10, high=190, size=ng),
        "y": rng.uniform(low=10, high=190, size=ng),
    }

    x = galaxy_params.pop("x")
    y = galaxy_params.pop("y")

    stamps, final_pad_image = _render_scene_stamps_jax_galsim(
        galaxy_params,
        x,
        y,
        fft_size,
        slen,
        ilen,
        ng,
    )
    assert stamps.array.shape == (ng, slen, slen)
    assert stamps.array.sum() > 0

    np.testing.assert_allclose(final_pad_image.array.sum(), stamps.array.sum())

    if False:
        import pdb

        import matplotlib.pyplot as plt

        plt.imshow(final_pad_image.array)
        pdb.set_trace()

    gs_image = _render_scene_stamps_galsim(
        galaxy_params,
        x,
        y,
        fft_size,
        slen,
        ilen,
        ng,
    )

    gs_image_mo = _render_scene_stamps_galsim(
        galaxy_params,
        x,
        y,
        fft_size,
        slen + 1,
        ilen,
        ng,
    )

    abs_eps = np.max(np.abs(gs_image_mo.array - gs_image.array))
    rel_eps = 0.0

    if False:
        import pdb

        import matplotlib.pyplot as plt

        plt.imshow(gs_image_mo.array - gs_image.array)
        pdb.set_trace()

    if False:
        import pdb

        import matplotlib.pyplot as plt

        plt.imshow(gs_image.array)
        pdb.set_trace()

    if False:
        import pdb

        import matplotlib.pyplot as plt

        plt.imshow(final_pad_image.array[slen:-slen, slen:-slen] - gs_image.array)
        pdb.set_trace()

    np.testing.assert_allclose(
        final_pad_image.array[slen:-slen, slen:-slen].sum(),
        gs_image.array.sum(),
        atol=abs_eps,
        rtol=rel_eps,
    )

    np.testing.assert_allclose(
        final_pad_image.array[slen:-slen, slen:-slen],
        gs_image.array,
        atol=abs_eps,
        rtol=rel_eps,
    )
