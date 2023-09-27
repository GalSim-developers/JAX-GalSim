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
from jax_galsim.interpolatedimage import (
    _draw_with_interpolant_kval,
    _draw_with_interpolant_xval,
)


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
def test_interpolatedimage_utils_draw_with_interpolant_kval(interp):
    zp = jnp.array(
        [
            [0.01, 0.08, 0.07, 0.02],
            [0.13, 0.38, 0.52, 0.06],
            [0.09, 0.41, 0.44, 0.09],
            [0.04, 0.11, 0.10, 0.01],
        ]
    )
    kim = jax_galsim.Image(zp, scale=1.0).calculate_fft()
    zpherm = jnp.concatenate(
        [kim.array[:, 1:][::-1, ::-1].conjugate(), kim.array], axis=1
    )
    nherm = kim.array.shape[0]
    minherm = kim.bounds.ymin

    for x in range(nherm):
        for y in range(nherm):
            np.testing.assert_allclose(
                _draw_with_interpolant_kval(
                    jnp.array([x + minherm], dtype=float),
                    jnp.array([y + minherm], dtype=float),
                    minherm,
                    minherm,
                    kim.array,
                    interp,
                ),
                zpherm[y, x],
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


@pytest.mark.parametrize("method", ["xValue", "kValue"])
def test_interpolated_image_utils_comp_to_galsim(method):
    ref_array = np.array(
        [
            [0.01, 0.08, 0.07, 0.02, 0.0, 0.0],
            [0.13, 0.38, 0.52, 0.06, 0.0, 0.05],
            [0.09, 0.41, 0.44, 0.09, 0.0, 0.2],
            [0.04, 0.11, 0.10, 0.01, 0.0, 0.5],
            [0.04, 0.11, 0.10, 0.01, 0.0, 0.3],
            [0.04, 0.11, 0.10, 0.01, 0.0, 0.1],
        ]
    )
    gimage_in = _galsim.Image(ref_array, scale=1)
    jgimage_in = jax_galsim.Image(ref_array, scale=1)

    for wcs in [
        _galsim.PixelScale(1.0),
        _galsim.JacobianWCS(2.1, 0.3, -0.4, 2.3),
        _galsim.AffineTransform(-0.3, 2.1, 1.8, 0.1, _galsim.PositionD(0.3, -0.4)),
    ]:
        gii = _galsim.InterpolatedImage(gimage_in, wcs=wcs)
        jgii = jax_galsim.InterpolatedImage(
            jgimage_in, wcs=jax_galsim.BaseWCS.from_galsim(wcs)
        )

        np.testing.assert_allclose(gii.stepk, jgii.stepk, rtol=0.5, atol=0)
        np.testing.assert_allclose(gii.maxk, jgii.maxk, rtol=0.5, atol=0)
        kxvals = [(0, 0), (1, 1), (1, -2), (-3, 4)]
        for x, y in kxvals:
            if method == "kValue":
                dk = jgii._original._kim.scale
                np.testing.assert_allclose(
                    gii.kValue(x * dk, y * dk),
                    jgii.kValue(x * dk, y * dk),
                    err_msg=f"kValue mismatch: wcs={wcs}, x={x}, y={y} kim={jgii._original._kim(x, y)}",
                )
            else:
                dx = jnp.sqrt(jgii._original._wcs.pixelArea())
                np.testing.assert_allclose(
                    gii.xValue(x * dx, y * dx),
                    jgii.xValue(x * dx, y * dx),
                    err_msg=f"xValue mismatch: wcs={wcs}, x={x}, y={y}",
                )


def _compute_fft_with_numpy_jax_galsim(im):
    import numpy as np

    from jax_galsim import BoundsI, Image

    No2 = max(-im.bounds.xmin, im.bounds.xmax + 1, -im.bounds.ymin, im.bounds.ymax + 1)

    full_bounds = BoundsI(-No2, No2 - 1, -No2, No2 - 1)
    if im.bounds == full_bounds:
        # Then the image is already in the shape we need.
        ximage = im
    else:
        # Then we pad out with zeros
        ximage = Image(full_bounds, dtype=im.dtype, init_value=0)
        ximage[im.bounds] = im[im.bounds]

    dx = im.scale
    # dk = 2pi / (N dk)
    dk = np.pi / (No2 * dx)

    out = Image(BoundsI(0, No2, -No2, No2 - 1), dtype=np.complex128, scale=dk)
    out._array = np.fft.fftshift(np.fft.rfft2(ximage.array), axes=0)
    out *= dx * dx
    out.setOrigin(0, -No2)
    return out


@pytest.mark.parametrize("n", [5, 4])
def test_jax_galsim_fft_vs_numpy(n):
    import numpy as np

    import jax_galsim as galsim

    rng = np.random.RandomState(42)
    arr = rng.normal(size=(n, n))
    im = galsim.Image(arr, scale=1)
    kim = im.calculate_fft()
    xkim = kim.calculate_inverse_fft()

    np.testing.assert_allclose(im.array, xkim[im.bounds].array)

    np_kim = _compute_fft_with_numpy_jax_galsim(im)
    print("ratio real:\n", np_kim.array.real / kim.array.real)
    print("ratio imag:\n", np_kim.array.imag / kim.array.imag)
    np.testing.assert_allclose(kim.array, np_kim.array)


def _compute_fft_with_numpy_galsim(im):
    import numpy as np
    from galsim import BoundsI, Image

    No2 = max(-im.bounds.xmin, im.bounds.xmax + 1, -im.bounds.ymin, im.bounds.ymax + 1)

    full_bounds = BoundsI(-No2, No2 - 1, -No2, No2 - 1)
    if im.bounds == full_bounds:
        # Then the image is already in the shape we need.
        ximage = im
    else:
        # Then we pad out with zeros
        ximage = Image(full_bounds, dtype=im.dtype, init_value=0)
        ximage[im.bounds] = im[im.bounds]

    dx = im.scale
    # dk = 2pi / (N dk)
    dk = np.pi / (No2 * dx)

    out = Image(BoundsI(0, No2, -No2, No2 - 1), dtype=np.complex128, scale=dk)
    out._array = np.fft.fftshift(np.fft.rfft2(ximage.array), axes=0)
    out *= dx * dx
    out.setOrigin(0, -No2)
    return out


@pytest.mark.parametrize("n", [5, 4])
def test_galsim_fft_vs_numpy(n):
    import galsim
    import numpy as np

    rng = np.random.RandomState(42)
    arr = rng.normal(size=(n, n))
    im = galsim.Image(arr, scale=1)
    kim = im.calculate_fft()
    xkim = kim.calculate_inverse_fft()

    np.testing.assert_allclose(im.array, xkim[im.bounds].array)

    np_kim = _compute_fft_with_numpy_galsim(im)
    print("ratio real:\n", np_kim.array.real / kim.array.real)
    print("ratio imag:\n", np_kim.array.imag / kim.array.imag)
    np.testing.assert_allclose(kim.array, np_kim.array)
