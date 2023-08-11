import jax
import jax.numpy as jnp

import jax_galsim as galsim

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
        return (
            self.sigma == other.sigma
            and self.flux == other.flux
            and self.gsparams == other.gsparams
        )

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
        return (
            self.scale_radius == other.scale_radius
            and self.flux == other.flux
            and self.gsparams == other.gsparams
        )

    # Check that after jitting the oject is still the same
    assert all([test_eq(identity(o), o) for o in objects])


def test_moffat_jiiting():
    # Test Moffat objects
    fwhm_backwards_compatible = 1.3178976627539716
    objects = [
        galsim.Moffat(beta=5.0, flux=0.2, half_light_radius=1.0, gsparams=gsparams),
        galsim.Moffat(beta=2.0, half_light_radius=1.0, trunc=5*fwhm_backwards_compatible, flux=1.0, gsparams=gsparams)
    ]
    # Test equality function from original galsim moffat.py
    def test_eq(self, other):
        return (
            self.scale_radius == other.scale_radius
            and self.flux == other.flux
            and self.gsparams == other.gsparams
        )

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
        return (
            self._local_wcs == other._local_wcs
            and self.origin == other.origin
            and self.world_origin == other.world_origin
        )

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
