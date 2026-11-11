import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim


def _gaussian_image(dtype, n=32, scale=0.2):
    return jax_galsim.Gaussian(sigma=1.0).drawImage(
        nx=n, ny=n, scale=scale, dtype=dtype, method="no_pixel"
    )


@pytest.mark.parametrize(
    "dtype,fft_dtype,ifft_dtype",
    [
        (jnp.float32, jnp.complex64, jnp.float32),
        (jnp.int32, jnp.complex128, jnp.float64),
    ],
)
def test_calculate_fft_dtype(dtype, fft_dtype, ifft_dtype):
    im = jax_galsim.Image(
        np.arange(64).reshape(8, 8).astype(dtype), scale=0.2, dtype=dtype
    )
    kim = im.calculate_fft()
    assert kim.dtype == fft_dtype
    assert kim.array.dtype == fft_dtype

    inv_fft_im = kim.calculate_inverse_fft()
    assert inv_fft_im.dtype == ifft_dtype
    assert inv_fft_im.array.dtype == ifft_dtype


@pytest.mark.parametrize(
    "dtype,kdtype",
    [
        (jnp.float32, jnp.complex64),
    ],
)
def test_interpolated_image_preserves_dtype(dtype, kdtype):
    im = _gaussian_image(dtype)
    ii = jax_galsim.InterpolatedImage(im, x_interpolant="lanczos15")
    assert ii._original._image.dtype == dtype
    assert ii._original._kim.dtype == kdtype


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.complex64, jnp.complex64),
        (jnp.complex64, jnp.complex64),
        (np.complex128, jnp.complex128),
        (jnp.complex128, jnp.complex128),
        (complex, jnp.complex128),
    ],
)
def test_drawkimage_dtype_kwarg(dtype, expected):
    obj = jax_galsim.Gaussian(sigma=1.0)
    kim = obj.drawKImage(nx=32, ny=32, scale=0.1, dtype=dtype)
    assert kim.dtype == expected
    assert kim.array.dtype == expected


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_drawkimage_image_kwarg_dtype(dtype):
    obj = jax_galsim.Gaussian(sigma=1.0)
    image_in = jax_galsim.Image(32, 32, scale=0.1, dtype=dtype)

    # below dtype argument shout be ignored
    kim = obj.drawKImage(image=image_in)
    for im in (image_in, kim):
        assert im.dtype == dtype
        assert im.array.dtype == dtype

    kim = obj.drawKImage(image=image_in, dtype=dtype)
    for im in (image_in, kim):
        assert im.dtype == dtype
        assert im.array.dtype == dtype


@pytest.mark.test_in_float32
def test_image_and_array_dtype_are_always_consistent():
    image = jax_galsim.Image(2, 2, dtype=jnp.float64)
    assert image.dtype == image.array.dtype.type

    leaves, treedef = jax.tree_util.tree_flatten(image)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.dtype == rebuilt.array.dtype.type
