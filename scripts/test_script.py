#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

import jax_galsim as jgs


def _get_bd_jgs(
    flux_d,
    flux_b,
    hlr_b,
    hlr_d,
    q_b,
    q_d,
    beta,
    *,
    psf: jgs.GSObject,
):

    disk = jgs.Exponential(flux=flux_d, half_light_radius=hlr_d).shear(
        q=q_d, beta=beta * jgs.degrees
    )
    bulge = jgs.Spergel(nu=-0.6, flux=flux_b, half_light_radius=hlr_b).shear(
        q=q_b, beta=beta * jgs.degrees
    )
    galaxy = disk + bulge
    gal_conv = jgs.Convolve([galaxy, psf])
    return gal_conv


def _draw_stamp_jgs(
    galaxy_params: dict,
    image_pos: jgs.PositionD,
    local_wcs: jgs.PixelScale,
    psf: jgs.GSObject,
    slen: int,
    fft_size: int,
) -> jax.Array:
    gsparams = jgs.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)
    convolved_object = _get_bd_jgs(**galaxy_params, psf=psf).withGSParams(gsparams)
    stamp = convolved_object.drawImage(
        nx=slen, ny=slen, center=image_pos, wcs=local_wcs, dtype=jnp.float64
    )
    return stamp


def _draw_stamp_and_add_to_image(carry, x, *, psf, fft_size, slen):
    # scan already jits so a bit overkill
    image = carry[0]
    gparams, image_pos, lwcs = x
    stamp = _draw_stamp_jgs(
        galaxy_params=gparams,
        image_pos=image_pos,
        local_wcs=lwcs,
        psf=psf,
        slen=slen,
        fft_size=fft_size,
    )
    image[stamp.bounds] += stamp
    return (image,), None


def draw_jgs_scan_stamps(
    galaxy_params: dict,
    *,
    psf: jgs.GSObject,
    ilen: int,
    slen: int,
    fft_size: int,
    max_n_gals: int,
):
    # I think this version will be better in CPU than vmap

    # create big image
    image = jgs.ImageD(ncol=ilen, nrow=ilen, scale=0.2)
    wcs = image.wcs

    x = galaxy_params.pop("x")
    y = galaxy_params.pop("y")

    image_positions = vmap(lambda x, y: jgs.PositionD(x=x, y=y))(x, y)
    local_wcss = vmap(lambda x: wcs.local(image_pos=x))(image_positions)

    pad_image = jgs.ImageD(
        jnp.pad(image.array, slen), wcs=image.wcs, bounds=image.bounds.withBorder(slen)
    )

    _func_to_scan = partial(
        _draw_stamp_and_add_to_image, psf=psf, fft_size=fft_size, slen=slen
    )

    final_pad_image = jax.lax.scan(
        _func_to_scan,
        (pad_image,),
        xs=(galaxy_params, image_positions, local_wcss),
        length=max_n_gals,
    )[0][0]

    return final_pad_image.array[slen:-slen, slen:-slen]


DEVICE = jax.devices()[0]
max_n_gals_bins = [10, 20, 30]
stamp_slen_bins = [52, 55, 58]
xpsf = jgs.Gaussian(half_light_radius=0.7, flux=1.0)
fft_size = 128
image_slen = 250


draw_fncs = []
_draw_fnc_raw = draw_jgs_scan_stamps
for _max_n_gals, _batch_slen in zip(max_n_gals_bins, stamp_slen_bins):
    draw_fncs.append(
        jit(
            partial(
                _draw_fnc_raw,
                psf=xpsf,
                ilen=image_slen,
                fft_size=fft_size,
                max_n_gals=_max_n_gals,
                slen=_batch_slen,
            )
        )
    )

for _max_n_gals, _draw_fnc in zip(max_n_gals_bins, draw_fncs):
    # stupid test
    _test_sample = sample_jax = {
        "flux_b": jnp.array([10.0] * _max_n_gals),
        "flux_d": jnp.array([10.0] * _max_n_gals),
        "hlr_b": jnp.array([1.0] * _max_n_gals),
        "hlr_d": jnp.array([1.0] * _max_n_gals),
        "q_d": jnp.array([1.0] * _max_n_gals),
        "q_b": jnp.array([1.0] * _max_n_gals),
        "beta": jnp.array([0.0] * _max_n_gals),
        "x": jnp.array([1.0] * _max_n_gals),
        "y": jnp.array([0.0] * _max_n_gals),
    }
    # _new_dict_jax = device_put(
    #     {p: sample[p][:_max_n_gals] for p in sample}, device=device
    # )
    _draw_fnc(_test_sample).block_until_ready()
