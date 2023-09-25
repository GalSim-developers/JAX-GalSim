import galsim as _galsim
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim
from jax_galsim.interpolant import (  # SincInterpolant,
    Cubic,
    Lanczos,
    Linear,
    Nearest,
    Quintic,
)
from jax_galsim.interpolatedimage import _draw_with_interpolant_xval


@pytest.mark.parametrize(
    "interp",
    [
        Nearest(),
        Linear(),
        # this is really slow right now and I am not sure why will fix later
        # SincInterpolant(),
        Linear(),
        Cubic(),
        Quintic(),
        Lanczos(3, conserve_dc=False),
        Lanczos(5, conserve_dc=True),
    ],
)
def test_interpolatedimage_utils_draw_with_interpolant_xval(interp):
    zp = jnp.array(
        [
            [0.01, 0.08, 0.07, 0.02],
            [0.13, 0.38, 0.52, 0.06],
            [0.09, 0.41, 0.44, 0.09],
            [0.04, 0.11, 0.10, 0.01],
        ]
    )
    for xmin in [-3, 0, 2]:
        for ymin in [-5, 0, 1]:
            for x in range(4):
                for y in range(4):
                    np.testing.assert_allclose(
                        _draw_with_interpolant_xval(
                            jnp.array([x + xmin], dtype=float),
                            jnp.array([y + ymin], dtype=float),
                            xmin,
                            ymin,
                            zp,
                            interp,
                        ),
                        zp[y, x],
                    )


def test_interpolatedimage_utils_stepk_maxk():
    ref_array = np.array(
        [
            [0.01, 0.08, 0.07, 0.02],
            [0.13, 0.38, 0.52, 0.06],
            [0.09, 0.41, 0.44, 0.09],
            [0.04, 0.11, 0.10, 0.01],
        ]
    )
    test_scale = 2.0
    gimage_in = _galsim.Image(ref_array)
    jgimage_in = jax_galsim.Image(ref_array)
    gii = _galsim.InterpolatedImage(gimage_in, scale=test_scale)
    jgii = jax_galsim.InterpolatedImage(jgimage_in, scale=test_scale)

    np.testing.assert_allclose(gii.stepk, jgii.stepk, rtol=0.2, atol=0)
    np.testing.assert_allclose(gii.maxk, jgii.maxk, rtol=0.2, atol=0)
