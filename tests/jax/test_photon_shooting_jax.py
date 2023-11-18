import jax
import numpy as np
import pytest
from jax.tree_util import register_pytree_node_class

import jax_galsim


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

    # code for testing
    # if not np.allclose(image2.array, ref_array) and False:
    #     import proplot as pplt

    #     fig, axs = pplt.subplots(nrows=1, ncols=3)
    #     axs[0].imshow(ref_array)
    #     axs[1].imshow(image2.array)
    #     axs[2].imshow(image2.array - ref_array)

    #     import pdb

    #     pdb.set_trace()

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
