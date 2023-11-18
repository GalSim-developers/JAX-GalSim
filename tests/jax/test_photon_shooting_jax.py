import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import register_pytree_node_class

import jax_galsim
from jax_galsim.core.testing import time_code_block


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

    # code for testing
    if not np.allclose(img_fft.array, img_phot.array, rtol=rtol, atol=atol):
        import proplot as pplt

        fig, axs = pplt.subplots(nrows=1, ncols=3)
        axs[0].imshow(img_fft.array, origin="lower")
        axs[1].imshow(img_phot.array, origin="lower")
        axs[2].imshow(img_fft.array - img_phot.array, origin="lower")
        fig.show()

        import pdb

        pdb.set_trace()

    np.testing.assert_almost_equal(
        jnp.argmax(img_fft.array),
        jnp.argmax(img_phot.array),
    )

    np.testing.assert_allclose(img_fft.array, img_phot.array, rtol=rtol, atol=atol)
