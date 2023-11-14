import jax_galsim
import numpy as np


def test_photon_array_make_from_image_notranspose():
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

    if not np.allclose(image2.array, ref_array) and False:
        import proplot as pplt

        fig, axs = pplt.subplots(nrows=1, ncols=3)
        axs[0].imshow(ref_array)
        axs[1].imshow(image2.array)
        axs[2].imshow(image2.array - ref_array)

        import pdb

        pdb.set_trace()

    np.testing.assert_allclose(image2.array, ref_array)
