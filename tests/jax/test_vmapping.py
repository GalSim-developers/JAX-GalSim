import jax
import jax_galsim as galsim

import jax.numpy as jnp


def duplicate(params):
    return {x: y * jnp.ones(2) for x, y in params.items()}


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
    assert all(
        [test_eq(jax.vmap(galsim.Gaussian)(**duplicate(o.params)), o) for o in objects]
    )


def test_exponential_vmapping():
    # Test Exponential objects
    objects = [
        galsim.Exponential(half_light_radius=1.0, flux=0.2),
        galsim.Exponential(scale_radius=0.1, flux=0.2),
    ]

    # Test equality function from original galsim exponential.py
    def test_eq(self, other):
        return (self.scale_radius == other.scale_radius).all() and (
            self.flux == other.flux
        ).all()

    # Check that after vmapping the oject is still the same
    assert all(
        [
            test_eq(jax.vmap(galsim.Exponential)(**duplicate(o.params)), o)
            for o in objects
        ]
    )
