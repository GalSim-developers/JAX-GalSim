import jax
import jax.numpy as jnp

import jax_galsim as galsim

e = jnp.ones(2)


def duplicate(params):
    return {x: y * e for x, y in params.items()}


def test_gaussian_vmapping():
    # Test Gaussian objects
    objects = [
        galsim.Gaussian(half_light_radius=1.0, flux=0.2),
        galsim.Gaussian(sigma=0.1, flux=0.2),
        galsim.Gaussian(fwhm=1.0, flux=0.2),
    ]

    # Test equality function from original galsim gaussian.py
    def test_eq(self, other):
        return (self.sigma == jnp.array([other.sigma, other.sigma])).all() and (
            self.flux == jnp.array([other.flux, other.flux])
        ).all()

    # Check that after vmapping the oject is still the same
    assert all([test_eq(jax.vmap(galsim.Gaussian)(**duplicate(o.params)), o) for o in objects])


def test_exponential_vmapping():
    # Test Exponential objects
    objects = [
        galsim.Exponential(half_light_radius=1.0, flux=0.2),
        galsim.Exponential(scale_radius=0.1, flux=0.2),
    ]

    # Test equality function from original galsim exponential.py
    def test_eq(self, other):
        return (
            self.scale_radius == jnp.array([other.scale_radius, other.scale_radius])
        ).all() and (self.flux == jnp.array([other.flux, other.flux])).all()

    # Check that after vmapping the oject is still the same
    assert all([test_eq(jax.vmap(galsim.Exponential)(**duplicate(o.params)), o) for o in objects])


def eq_pos(pos, other):
    return (pos.x == jnp.array([other.x, other.x])).all() and (
        pos.y == jnp.array([other.y, other.y])
    ).all()


def test_position_vmapping():
    obj = galsim.PositionD(1.0, 2.0)

    assert eq_pos(jax.vmap(galsim.PositionD)(1.0 * e, 2.0 * e), obj)


def test_affine_transform_vmapping():
    obj = galsim.AffineTransform(
        1.0,
        0.0,
        0.0,
        1.0,
        origin=galsim.PositionD(1.0, 2.0),
        world_origin=galsim.PositionD(1.0, 2.0),
    )

    print("now vmap")

    obj_duplicated = jax.vmap(galsim.AffineTransform)(
        1.0 * e,
        0.0 * e,
        0.0 * e,
        1.0 * e,
        origin=jax.vmap(galsim.PositionD)(1.0 * e, 2.0 * e),
        world_origin=jax.vmap(galsim.PositionD)(1.0 * e, 2.0 * e),
    )

    def test_eq(self, other):
        return (
            (
                self._local_wcs.dudx == jnp.array([other._local_wcs.dudx, other._local_wcs.dudx])
            ).all()
            and eq_pos(self.origin, other.origin)
            and eq_pos(self.world_origin, other.world_origin)
        )

    assert test_eq(obj_duplicated, obj)


def test_bounds_vmapping():
    obj = galsim.BoundsD(0.0, 1.0, 0.0, 1.0)
    obj_d = jax.vmap(galsim.BoundsD)(0.0 * e, 1.0 * e, 0.0 * e, 1.0 * e)

    objI = galsim.BoundsI(0.0, 1.0, 0.0, 1.0)
    objI_d = jax.vmap(galsim.BoundsI)(0.0 * e, 1.0 * e, 0.0 * e, 1.0 * e)

    def test_eq(self, other):
        return (
            (self.xmin == jnp.array([other.xmin, other.xmin])).all()
            and (self.xmax == jnp.array([other.xmax, other.xmax])).all()
            and (self.ymin == jnp.array([other.ymin, other.ymin])).all()
            and (self.ymax == jnp.array([other.ymax, other.ymax])).all()
        )

    assert test_eq(obj_d, obj)
    assert test_eq(objI_d, objI)
