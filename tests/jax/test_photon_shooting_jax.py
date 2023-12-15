from functools import partial

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import register_pytree_node_class

import jax_galsim
from jax_galsim.core.draw import calculate_n_photons
from jax_galsim.core.testing import time_code_block
from jax_galsim.photon_array import fixed_photon_array_size


def test_photon_shooting_jax_make_from_image_notranspose():
    # this test uses a very assymetric array to ensure there is not a transpose
    # error in the code
    ref_array = np.array(
        [
            [0.01, 0.08, 0.07, 0.02],
            [0.13, 0.38, 10.52, 0.06],
            [0.09, 0.41, 0.44, 0.09],
            [0.04, 0.11, 0.10, 0.01],
            [0.04, 0.11, 0.10, 0.01],
        ]
    )
    image = jax_galsim.Image(ref_array)

    photons = jax_galsim.PhotonArray.makeFromImage(image, max_flux=0.1)

    image2 = jax_galsim.Image(np.zeros_like(ref_array))
    photons.addTo(image2)


    np.testing.assert_allclose(image2.array, ref_array)


@register_pytree_node_class
class NoShootingExponential(jax_galsim.Exponential):
    def _shoot(self, *args, **kwargs):
        raise NotImplementedError("this is a test")

    def tree_flatten(self):
        return super().tree_flatten()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return super().tree_unflatten(aux_data, children)


def test_photon_shooting_jax_raises():
    obj = NoShootingExponential(half_light_radius=1.0, flux=1.0)
    with pytest.raises(jax_galsim.errors.GalSimNotImplementedError):
        obj.drawImage(nx=33, ny=33, scale=0.2, method="phot", n_photons=1000)

    @jax.jit
    def _jitted():
        obj = NoShootingExponential(half_light_radius=1.0, flux=1.0)
        return obj.drawImage(nx=33, ny=33, scale=0.2, method="phot", n_photons=1000)

    with pytest.raises(jax_galsim.errors.GalSimNotImplementedError):
        _jitted()


@pytest.mark.parametrize(
    "offset",
    [
        (0, 0),
        (5, 5),
        (0, 5),
        (0, -5),
        (5, 0),
        (-5, 0),
        (-5, -5),
        (-5, 5),
        (5, -5),
    ],
)
def test_photon_shooting_jax_offset(offset):
    gal = jax_galsim.Exponential(
        half_light_radius=0.5, flux=1000.0
    ) + jax_galsim.Exponential(half_light_radius=1.0, flux=100.0)
    psf = jax_galsim.Gaussian(fwhm=0.9, flux=1.0)
    obj = jax_galsim.Convolve([gal, psf])
    img = obj.drawImage(scale=0.2)
    iobj = jax_galsim.InterpolatedImage(img, scale=0.2, offset=offset)

    img_fft = iobj.drawImage(nx=33, ny=33, scale=0.2, dtype=np.float64)

    # these magic equations come from the do_shoot routine in the
    # GalSim file galsim/tests/galsim_test_helpers.py
    rtol = 1e-2
    flux_tot = iobj.flux
    flux_max = iobj.max_sb * 0.2**2
    atol = flux_max * rtol * 3
    nphot = int((flux_tot / flux_max / rtol**2).item())
    rtol *= 3

    with time_code_block():
        img_phot = iobj.drawImage(
            nx=33,
            ny=33,
            scale=0.2,
            method="phot",
            dtype=np.float64,
            n_photons=nphot,
            maxN=10000,
            rng=jax_galsim.BaseDeviate(1234),
        )

    print(
        "fft|phot argmax:",
        jnp.argmax(img_fft.array),
        jnp.argmax(img_phot.array),
    )

    print(
        "fft|phot max:",
        jnp.max(img_fft.array),
        jnp.max(img_phot.array),
    )

    print(
        "fft|phot sum:",
        jnp.sum(img_fft.array),
        jnp.sum(img_phot.array),
    )

    print(
        "fft  moments:",
        " ".join(
            "%s:% 15.7e" % (k, v)
            for k, v in jax_galsim.utilities.unweighted_moments(img_fft).items()
        ),
    )
    print(
        "phot moments:",
        " ".join(
            "%s:% 15.7e" % (k, v)
            for k, v in jax_galsim.utilities.unweighted_moments(img_phot).items()
        ),
    )


    np.testing.assert_almost_equal(
        jnp.argmax(img_fft.array),
        jnp.argmax(img_phot.array),
    )

    np.testing.assert_allclose(img_fft.array, img_phot.array, rtol=rtol, atol=atol)


@pytest.mark.parametrize("max_n_phot", [10, 100, 1000])
def test_photon_shooting_jax_vmapping(max_n_phot):
    n_stamps = 100
    rng = np.random.RandomState(1234)
    shifts = jnp.array(rng.uniform(-1, 1, size=(n_stamps, 2)))
    hlrs = jnp.array(rng.uniform(0.1, 1.0, size=(n_stamps,)))
    fwhms = jnp.array(rng.uniform(0.9, 1.0, size=(n_stamps,)))
    fluxes = jnp.array(rng.uniform(100, 1000, size=(n_stamps,)))
    rng = jax_galsim.BaseDeviate(1234)
    seeds = []
    for i in range(n_stamps):
        seeds.append(jax.random.key(i + 1))
    seeds = jnp.array(seeds)

    @jax.jit
    def _draw(hlr, fwhm, shift, flux, seed):
        obj = jax_galsim.Convolve(
            [
                jax_galsim.Exponential(half_light_radius=hlr, flux=flux).shift(*shift),
                jax_galsim.Gaussian(fwhm=fwhm, flux=1.0),
            ]
        )
        with fixed_photon_array_size(max_n_phot):
            return obj.drawImage(
                nx=53,
                ny=53,
                scale=0.2,
                method="phot",
                rng=jax_galsim.BaseDeviate(seed),
            )

    with time_code_block("one warmup"):
        img = _draw(hlrs[0], fwhms[0], shifts[0], fluxes[0], seeds[0])
    with time_code_block("one"):
        img = _draw(hlrs[0], fwhms[0], shifts[0], fluxes[0], seeds[0])

    _vmap_draw = jax.jit(jax.vmap(_draw, in_axes=(0, 0, 0, 0, 0)))
    with time_code_block("vmap warmup"):
        imgs = _vmap_draw(hlrs, fwhms, shifts, fluxes, seeds)
    with time_code_block("vmap"):
        imgs = _vmap_draw(hlrs, fwhms, shifts, fluxes, seeds)

    np.testing.assert_allclose(img.array.sum(), imgs.array[0].sum())

    def _draw_galsim(hlr, fwhm, shift, flux, seed):
        obj = _galsim.Convolve(
            [
                _galsim.Exponential(half_light_radius=hlr, flux=flux).shift(*shift),
                _galsim.Gaussian(fwhm=fwhm, flux=1.0),
            ]
        )
        return obj.drawImage(
            nx=53,
            ny=53,
            scale=0.2,
            method="phot",
            rng=_galsim.BaseDeviate(seed),
        )

    with time_code_block("galsim"):
        for i in range(n_stamps):
            _draw_galsim(hlrs[i], fwhms[i], shifts[i], fluxes[i], i + 1)


def test_photon_shooting_jax_rng_seed():
    def _build_and_draw(hlr, fwhm, jit=True, new_seed=False):
        gal = jax_galsim.Exponential(
            half_light_radius=hlr, flux=1000.0
        ) + jax_galsim.Exponential(half_light_radius=hlr * 2.0, flux=100.0)
        psf = jax_galsim.Gaussian(fwhm=fwhm, flux=1.0)
        final = jax_galsim.Convolve(
            [gal, psf],
        )
        n = final.getGoodImageSize(0.2).item()
        n += 1
        n_photons = calculate_n_photons(
            final.flux,
            final._flux_per_photon,
            final.max_sb,
            poisson_flux=True,
            rng=jax_galsim.BaseDeviate(1234),
        )[0].item()
        if jit:
            if new_seed:
                return _draw_it_jit_new_seed(final, n_photons)
            else:
                return _draw_it_jit(final, n_photons)
        else:
            return final.makePhot(
                local_wcs=jax_galsim.PixelScale(0.2),
                n_photons=n_photons,
                poisson_flux=False,
                rng=jax_galsim.BaseDeviate(42),
            )

    @partial(jax.jit, static_argnums=(1,))
    def _draw_it_jit(obj, n_photons):
        return obj.makePhot(
            local_wcs=jax_galsim.PixelScale(0.2),
            n_photons=n_photons,
            poisson_flux=False,
            rng=jax_galsim.BaseDeviate(42),
        )

    @partial(jax.jit, static_argnums=(1,))
    def _draw_it_jit_new_seed(obj, n_photons):
        return obj.makePhot(
            local_wcs=jax_galsim.PixelScale(0.2),
            n_photons=n_photons,
            poisson_flux=False,
            rng=jax_galsim.BaseDeviate(2),
        )

    with time_code_block("warmup no-jit"):
        pa1 = _build_and_draw(0.5, 1.0, jit=False)

    with time_code_block("no-jit"):
        pa2 = _build_and_draw(0.5, 1.0, jit=False)

    with time_code_block("warmup jit"):
        pa3 = _build_and_draw(0.5, 1.0)

    with time_code_block("jit"):
        pa4 = _build_and_draw(0.5, 1.0)

    with time_code_block("warmup jit + new seed"):
        pa5 = _build_and_draw(0.5, 1.0, new_seed=True, jit=True)

    with time_code_block("jit + new seed"):
        pa6 = _build_and_draw(0.5, 1.0, new_seed=True, jit=True)

    for attr in ["x", "y"]:
        np.testing.assert_allclose(getattr(pa1, attr), getattr(pa2, attr))
        np.testing.assert_allclose(getattr(pa1, attr), getattr(pa3, attr))
        np.testing.assert_allclose(getattr(pa1, attr), getattr(pa4, attr))

        assert not np.allclose(getattr(pa1, attr), getattr(pa5, attr))

        np.testing.assert_allclose(getattr(pa5, attr), getattr(pa6, attr))
