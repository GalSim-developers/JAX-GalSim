import jax
import numpy as np
import pytest
from galsim_test_helpers import timer

import jax_galsim as galsim
from jax_galsim.core.wrap_image import (
    _block_reduce_index,
    _block_reduce_loop,
    contract_hermitian_x,
    contract_hermitian_y,
    expand_hermitian_x,
    expand_hermitian_y,
)


@timer
def test_image_wrapping_expand_contract():
    # For complex images (in particular k-space images), we often want the image to be implicitly
    # Hermitian, so we only need to keep around half of it.
    M = 38
    N = 25
    im = galsim.ImageCD(2 * M + 1, 2 * N + 1, xmin=-M, ymin=-N)  # Explicitly Hermitian
    im2 = galsim.ImageCD(
        2 * M + 1, N + 1, xmin=-M, ymin=0
    )  # Implicitly Hermitian across y axis
    im3 = galsim.ImageCD(
        M + 1, 2 * N + 1, xmin=0, ymin=-N
    )  # Implicitly Hermitian across x axis
    # print('im = ',im)
    # print('im2 = ',im2)
    # print('im3 = ',im3)
    for i in range(-M, M + 1):
        for j in range(-N, N + 1):
            # An arbitrary, complicated Hermitian function.
            val = (
                np.exp((i / (2.3 * M)) ** 2 + 1j * (2.8 * i - 1.3 * j))
                + ((2 + 3j * j) / (1.9 * N)) ** 3
            )
            # val = 2*(i-j)**2 + 3j*(i+j)

            im[i, j] = val
            if j >= 0:
                im2[i, j] = val
            if i >= 0:
                im3[i, j] = val

    # print("im = ",im.array)

    # Confirm that the image is Hermitian.
    for i in range(-M, M + 1):
        for j in range(-N, N + 1):
            assert im(i, j) == im(-i, -j).conjugate()

    im_exp = expand_hermitian_x(im3.array)
    np.testing.assert_allclose(
        im_exp,
        im.array,
        err_msg="expand_hermitian_x() did not match expectation",
    )

    im_cnt = contract_hermitian_x(im.array)
    np.testing.assert_allclose(
        im_cnt,
        im3.array,
        err_msg="contract_hermitian_x() did not match expectation",
    )

    im_exp = expand_hermitian_y(im2.array)
    np.testing.assert_allclose(
        im_exp,
        im.array,
        err_msg="expand_hermitian_x() did not match expectation",
    )

    im_cnt = contract_hermitian_y(im.array)
    np.testing.assert_allclose(
        im_cnt,
        im2.array,
        err_msg="contract_hermitian_x() did not match expectation",
    )


# the sizes here are to exercise both branches of the wrapping code which
# diverge for the number of blocks tilling the image greater than 50
@pytest.mark.parametrize("func", ["wrap", "vjp", "jvp", "vjp-jit", "jvp-jit"])
@pytest.mark.parametrize("K,L", [(5, 3), (19, 17)])
@timer
def test_image_wrapping_autodiff(func, K, L):
    # For complex images (in particular k-space images), we often want the image to be implicitly
    # Hermitian, so we only need to keep around half of it.
    M = 38
    N = 25
    im = galsim.ImageCD(2 * M + 1, 2 * N + 1, xmin=-M, ymin=-N)  # Explicitly Hermitian
    im2 = galsim.ImageCD(
        2 * M + 1, N + 1, xmin=-M, ymin=0
    )  # Implicitly Hermitian across y axis
    im3 = galsim.ImageCD(
        M + 1, 2 * N + 1, xmin=0, ymin=-N
    )  # Implicitly Hermitian across x axis
    # print('im = ',im)
    # print('im2 = ',im2)
    # print('im3 = ',im3)
    b = galsim.BoundsI(-K + 1, K, -L + 1, L)
    im_test = galsim.ImageCD(b, init_value=0)
    for i in range(-M, M + 1):
        for j in range(-N, N + 1):
            # An arbitrary, complicated Hermitian function.
            val = (
                np.exp((i / (2.3 * M)) ** 2 + 1j * (2.8 * i - 1.3 * j))
                + ((2 + 3j * j) / (1.9 * N)) ** 3
            )
            # val = 2*(i-j)**2 + 3j*(i+j)

            im[i, j] = val
            if j >= 0:
                im2[i, j] = val
            if i >= 0:
                im3[i, j] = val

            ii = (i - b.xmin) % (b.xmax - b.xmin + 1) + b.xmin
            jj = (j - b.ymin) % (b.ymax - b.ymin + 1) + b.ymin
            im_test.addValue(ii, jj, val)
    # print("im = ",im.array)

    # Confirm that the image is Hermitian.
    for i in range(-M, M + 1):
        for j in range(-N, N + 1):
            assert im(i, j) == im(-i, -j).conjugate()

    # make sure these run without error
    if func == "wrap":
        b3 = galsim.BoundsI(0, K, -L + 1, L)
        im.wrap(b3)
    elif func == "vjp-jit" or func == "jvp-jit":

        @jax.jit
        def _wrapit(im):
            b3 = galsim.BoundsI(0, K, -L + 1, L)
            return im.wrap(b3)

        if func == "vjp-jit":
            p, grad = jax.vjp(_wrapit, im3)
            grad = jax.jit(grad)
            grad(p)
        else:
            jax.jvp(_wrapit, (im3,), (im3 * 2,))
    elif func == "vjp" or func == "jvp":

        def _wrapit(im):
            b3 = galsim.BoundsI(0, K, -L + 1, L)
            return im.wrap(b3)

        if func == "vjp":
            p, grad = jax.vjp(_wrapit, im3)
            grad(p)
        else:
            jax.jvp(_wrapit, (im3,), (im3 * 2,))
    else:
        raise ValueError(
            f"Unknown function {func} for testing image wrapping w/ grads!"
        )


def test_image_wrapping_block_vs_index_reduce():
    nxwrap, nywrap = 16, 10
    nx = 3
    ny = 5
    Nx, Ny = nxwrap * nx, nywrap * ny
    im = jax.random.normal(jax.random.PRNGKey(1), (Ny, Nx))
    np.testing.assert_allclose(
        _block_reduce_index(im, nxwrap, nywrap),
        _block_reduce_loop(im, nx, ny, nxwrap, nywrap),
        err_msg="block_reduce_index() did not match block_reduce_loop()",
    )
