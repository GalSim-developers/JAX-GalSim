from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import jax_galsim as galsim
from jax_galsim.core.draw import calculate_n_photons
from jax_galsim.core.testing import time_code_block
from jax_galsim.photon_array import fixed_photon_array_size

# Defining jitting identity
identity = jax.jit(lambda x: x)
gsparams = galsim.GSParams(minimum_fft_size=32)


def test_gaussian_jitting():
    # Test Gaussian objects
    objects = [
        galsim.Gaussian(half_light_radius=1.0, flux=0.2, gsparams=gsparams),
        galsim.Gaussian(sigma=0.1, flux=0.2, gsparams=gsparams),
        galsim.Gaussian(fwhm=1.0, flux=0.2, gsparams=gsparams),
    ]

    # Test equality function from original galsim gaussian.py
    def test_eq(self, other):
        return self.sigma == other.sigma and self.flux == other.flux and self.gsparams == other.gsparams

    # Check that after jitting the oject is still the same
    assert all([test_eq(identity(o), o) for o in objects])


def test_exponential_jitting():
    # Test Exponential objects
    objects = [
        galsim.Exponential(half_light_radius=1.0, flux=0.2, gsparams=gsparams),
        galsim.Exponential(scale_radius=0.1, flux=0.2, gsparams=gsparams),
    ]

    # Test equality function from original galsim exponential.py
    def test_eq(self, other):
        return self.scale_radius == other.scale_radius and self.flux == other.flux and self.gsparams == other.gsparams

    # Check that after jitting the oject is still the same
    assert all([test_eq(identity(o), o) for o in objects])


def test_moffat_jitting():
    # Test Moffat objects
    fwhm_backwards_compatible = 1.3178976627539716
    objects = [
        galsim.Moffat(beta=5.0, flux=0.2, scale_radius=1.0, gsparams=gsparams),
        galsim.Moffat(
            beta=2.0,
            half_light_radius=1.0,
            trunc=5 * fwhm_backwards_compatible,
            flux=1.0,
            gsparams=gsparams,
        ),
    ]

    # Test equality function from original galsim moffat.py
    def test_eq(self, other):
        return self.scale_radius == other.scale_radius and self.flux == other.flux and self.gsparams == other.gsparams

    # Check that after jitting the oject is still the same
    assert all([test_eq(identity(o), o) for o in objects])


def test_spergel_jitting():
    # Test Spergel objects
    objects = [
        galsim.Spergel(nu=-0.85, flux=1.0, scale_radius=1.0, gsparams=gsparams),
        galsim.Spergel(nu=4.0, flux=0.2, half_light_radius=1.0, gsparams=gsparams),
    ]

    # Test equality function from original galsim spergel.py
    def test_eq(self, other):
        return self.scale_radius == other.scale_radius and self.flux == other.flux and self.gsparams == other.gsparams

    # Check that after jitting the oject is still the same
    assert all([test_eq(identity(o), o) for o in objects])


def test_pixel_jitting():
    objects = [
        galsim.Pixel(scale=0.2, flux=100, gsparams=gsparams),
        galsim.Pixel(scale=0.2, flux=1000, gsparams=gsparams),
    ]

    def test_eq(self, other):
        return (
            self.width == other.width
            and self.height == other.height
            and self.gsparams == other.gsparams
            and self.scale == other.scale
        )

    # Check that after jitting the oject is still the same
    assert all([test_eq(identity(o), o) for o in objects])


def test_sum_jitting():
    obj1 = galsim.Gaussian(half_light_radius=1.0, flux=0.2, gsparams=gsparams)
    obj2 = galsim.Exponential(half_light_radius=1.0, flux=0.2, gsparams=gsparams)

    obj = obj1 + obj2

    def test_eq(self, other):
        return (
            self.obj_list == other.obj_list
            and self.gsparams == other.gsparams
            and self._propagate_gsparams == other._propagate_gsparams
        )

    assert test_eq(identity(obj), obj)


def test_affine_transform_jitting():
    obj = galsim.AffineTransform(
        1.0,
        0.0,
        0.0,
        1.0,
        origin=galsim.PositionD(1.0, 2.0),
        world_origin=galsim.PositionD(1.0, 2.0),
    )

    def test_eq(self, other):
        return self._local_wcs == other._local_wcs and self.origin == other.origin and self.world_origin == other.world_origin

    assert test_eq(identity(obj), obj)


def test_bounds_jitting():
    obj = galsim.BoundsD(0.0, 1.0, 0.0, 1.0)

    objI = galsim.BoundsI(0.0, 1.0, 0.0, 1.0)

    assert identity(obj) == obj
    assert identity(objI) == objI


def test_image_jitting():
    ref_array = jnp.array(
        [
            [11, 21, 31, 41, 51, 61, 71],
            [12, 22, 32, 42, 52, 62, 72],
            [13, 23, 33, 43, 53, 63, 73],
            [14, 24, 34, 44, 54, 64, 74],
            [15, 25, 35, 45, 55, 65, 75],
        ]
    ).astype(dtype=jnp.float32)
    im1 = galsim.Image(ref_array, wcs=galsim.PixelScale(0.2), dtype=jnp.int32)
    assert identity(im1) == im1


def test_position_jitting():
    obj = galsim.PositionD(1.0, 2.0)

    def test_eq(self, other):
        return self.x == other.x and self.y == other.y

    assert test_eq(identity(obj), obj)


def test_shear_jitting():
    g = galsim.Shear(g1=0.1, g2=0.2)
    e = galsim.Shear(e1=0.1, e2=0.2)

    def test_eq(self, other):
        return self._g == other._g

    assert test_eq(identity(g), g)
    assert test_eq(identity(e), e)


def test_jitting_draw_fft():
    def _build_and_draw(hlr, fwhm, jit=True):
        gal = galsim.Exponential(half_light_radius=hlr, flux=1000.0)
        psf = galsim.Gaussian(fwhm=fwhm, flux=1.0)
        final = galsim.Convolve(
            [gal, psf],
        )
        n = final.getGoodImageSize(0.2).item()
        n += 1
        nfft = galsim.Image.good_fft_size(4 * n)
        if jit:
            return _draw_it_jit(final, n, nfft)
        else:
            final = final.withGSParams(
                minimum_fft_size=128,
                maximum_fft_size=128,
            )
            return final.drawImage(
                nx=n,
                ny=n,
                scale=0.2,
            )

    @partial(jax.jit, static_argnums=(1, 2))
    def _draw_it_jit(obj, n, nfft):
        obj = obj.withGSParams(
            minimum_fft_size=128,
            maximum_fft_size=128,
        )
        return obj.drawImage(
            nx=n,
            ny=n,
            scale=0.2,
        )

    with time_code_block("warmup no-jit"):
        img = _build_and_draw(0.5, 1.0, jit=False)
    np.testing.assert_array_almost_equal(img.array.sum(), 1000.0, 0)

    with time_code_block("no-jit"):
        img = _build_and_draw(0.5, 1.0, jit=False)
    np.testing.assert_array_almost_equal(img.array.sum(), 1000.0, 0)

    with time_code_block("warmup jit"):
        img = _build_and_draw(0.5, 1.0)
    np.testing.assert_array_almost_equal(img.array.sum(), 1000.0, 0)

    with time_code_block("jit"):
        img = _build_and_draw(0.5, 1.0)
    np.testing.assert_array_almost_equal(img.array.sum(), 1000.0, 0)


def test_jitting_draw_phot():
    def _build_and_draw(hlr, fwhm, jit=True, maxn=False):
        gal = galsim.Exponential(half_light_radius=hlr, flux=1000.0) + galsim.Exponential(
            half_light_radius=hlr * 2.0, flux=100.0
        )
        psf = galsim.Gaussian(fwhm=fwhm, flux=1.0)
        final = galsim.Convolve(
            [gal, psf],
        )
        n = final.getGoodImageSize(0.2).item()
        n += 1
        n_photons = calculate_n_photons(
            final.flux,
            final._flux_per_photon,
            final.max_sb,
            poisson_flux=False,
            rng=galsim.BaseDeviate(1234),
        )[0].item()
        gain = 1.0
        if jit:
            if maxn:
                return _draw_it_jit_maxn(final, n, n_photons, gain)
            else:
                return _draw_it_jit(final, n, n_photons, gain)
        else:
            return final.drawImage(
                nx=n,
                ny=n,
                scale=0.2,
                method="phot",
                n_photons=n_photons,
                poisson_flux=False,
                gain=gain,
                rng=galsim.BaseDeviate(42),
            )

    @partial(jax.jit, static_argnums=(1, 2))
    def _draw_it_jit_maxn(obj, n, nphotons, gain):
        return obj.drawImage(
            nx=n,
            ny=n,
            scale=0.2,
            n_photons=nphotons,
            method="phot",
            poisson_flux=False,
            gain=gain,
            maxN=101,
            rng=galsim.BaseDeviate(2),
        )

    @partial(jax.jit, static_argnums=(1, 2))
    def _draw_it_jit(obj, n, nphotons, gain):
        return obj.drawImage(
            nx=n,
            ny=n,
            scale=0.2,
            n_photons=nphotons,
            method="phot",
            poisson_flux=False,
            gain=gain,
            maxN=None,
            rng=galsim.BaseDeviate(42),
        )

    with time_code_block("warmup no-jit"):
        img1 = _build_and_draw(0.5, 1.0, jit=False)

    with time_code_block("no-jit"):
        img2 = _build_and_draw(0.5, 1.0, jit=False)

    with time_code_block("warmup jit"):
        img3 = _build_and_draw(0.5, 1.0)

    with time_code_block("jit"):
        img4 = _build_and_draw(0.5, 1.0)

    with time_code_block("warmup jit"):
        img5 = _build_and_draw(0.5, 1.0, maxn=True)

    with time_code_block("jit"):
        img6 = _build_and_draw(0.5, 1.0, maxn=True)

    np.testing.assert_allclose(img1.array, img2.array)
    np.testing.assert_allclose(img1.array, img3.array)
    np.testing.assert_allclose(img1.array, img4.array)

    assert not np.allclose(img1.array, img5.array)

    np.testing.assert_allclose(img5.array, img6.array)

    np.testing.assert_allclose(img1.array.sum(), 1100.0)
    np.testing.assert_allclose(img5.array.sum(), 1100.0)


def test_jitting_draw_phot_fixed():
    def _build_and_draw(hlr, fwhm, jit=True, maxn=False):
        gal = galsim.Exponential(half_light_radius=hlr, flux=1000.0) + galsim.Exponential(
            half_light_radius=hlr * 2.0, flux=100.0
        )
        psf = galsim.Gaussian(fwhm=fwhm, flux=1.0)
        final = galsim.Convolve(
            [gal, psf],
        )
        n = final.getGoodImageSize(0.2).item()
        n += 1
        n_photons = calculate_n_photons(
            final.flux,
            final._flux_per_photon,
            final.max_sb,
            poisson_flux=False,
            rng=galsim.BaseDeviate(1234),
        )[0].item()
        gain = 1.0
        if jit:
            if maxn:
                return _draw_it_jit_maxn(final, n, n_photons, gain)
            else:
                return _draw_it_jit(final, n, n_photons, gain)
        else:
            with fixed_photon_array_size(2048):
                return final.drawImage(
                    nx=n,
                    ny=n,
                    scale=0.2,
                    method="phot",
                    n_photons=n_photons,
                    poisson_flux=False,
                    gain=gain,
                    rng=galsim.BaseDeviate(42),
                )

    @partial(jax.jit, static_argnums=(1, 2))
    def _draw_it_jit(obj, n, nphotons, gain):
        with fixed_photon_array_size(2048):
            return obj.drawImage(
                nx=n,
                ny=n,
                scale=0.2,
                n_photons=nphotons,
                method="phot",
                poisson_flux=False,
                gain=gain,
                rng=galsim.BaseDeviate(42),
            )

    @partial(jax.jit, static_argnums=(1, 2))
    def _draw_it_jit_maxn(obj, n, nphotons, gain):
        with fixed_photon_array_size(2048):
            return obj.drawImage(
                nx=n,
                ny=n,
                scale=0.2,
                n_photons=nphotons,
                method="phot",
                poisson_flux=False,
                gain=gain,
                maxN=101,
                rng=galsim.BaseDeviate(42),
            )

    with time_code_block("warmup no-jit"):
        img1 = _build_and_draw(0.5, 1.0, jit=False)

    with time_code_block("no-jit"):
        img2 = _build_and_draw(0.5, 1.0, jit=False)

    with time_code_block("warmup jit"):
        img3 = _build_and_draw(0.5, 1.0)

    with time_code_block("jit"):
        img4 = _build_and_draw(0.5, 1.0)

    with time_code_block("warmup jit+maxn"):
        img5 = _build_and_draw(0.5, 1.0, maxn=True)

    with time_code_block("jit+maxn"):
        img6 = _build_and_draw(0.5, 1.0, maxn=True)

    np.testing.assert_allclose(img1.array, img2.array)
    np.testing.assert_allclose(img1.array, img3.array)
    np.testing.assert_allclose(img1.array, img4.array)

    assert not np.allclose(img1.array, img5.array)

    np.testing.assert_allclose(img5.array, img6.array)

    np.testing.assert_allclose(img1.array.sum(), 1100.0)
    np.testing.assert_allclose(img5.array.sum(), 1100.0)
