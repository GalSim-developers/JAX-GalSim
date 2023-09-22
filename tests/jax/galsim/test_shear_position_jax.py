import numpy as np

import galsim
from galsim_test_helpers import timer, assert_raises
from jax_galsim.core.wrap_image import (
    expand_hermitian_x,
    contract_hermitian_x,
    expand_hermitian_y,
    contract_hermitian_y,
)


@timer
def test_shear_position_image_integration_pixelwcs_jax():
    wcs = galsim.PixelScale(0.3)
    obj1 = galsim.Gaussian(sigma=3)
    obj2 = galsim.Gaussian(sigma=2)
    pos2 = galsim.PositionD(3, 5)
    sum = obj1 + obj2.shift(pos2)
    shear = galsim.Shear(g1=0.1, g2=0.18)
    im1 = galsim.Image(50, 50, wcs=wcs)
    sum.shear(shear).drawImage(im1, center=im1.center)

    # Equivalent to shear each object separately and drawing at the sheared position.
    im2 = galsim.Image(50, 50, wcs=wcs)
    obj1.shear(shear).drawImage(im2, center=im2.center)
    obj2.shear(shear).drawImage(
        im2,
        add_to_image=True,
        center=im2.center + wcs.toImage(pos2.shear(shear)),
    )

    print("err:", np.max(np.abs(im1.array - im2.array)))
    np.testing.assert_allclose(im1.array, im2.array, rtol=0, atol=5e-8)


@timer
def test_wrap_jax_simple_real():
    """Test the image.wrap() function."""
    # Start with a fairly simple test where the image is 4 copies of the same data:
    im_orig = galsim.Image(
        [
            [11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 13.0, 14.0],
            [21.0, 22.0, 23.0, 24.0, 21.0, 22.0, 23.0, 24.0],
            [31.0, 32.0, 33.0, 34.0, 31.0, 32.0, 33.0, 34.0],
            [41.0, 42.0, 43.0, 44.0, 41.0, 42.0, 43.0, 44.0],
            [11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 13.0, 14.0],
            [21.0, 22.0, 23.0, 24.0, 21.0, 22.0, 23.0, 24.0],
            [31.0, 32.0, 33.0, 34.0, 31.0, 32.0, 33.0, 34.0],
            [41.0, 42.0, 43.0, 44.0, 41.0, 42.0, 43.0, 44.0],
        ]
    )
    im = im_orig.copy()
    b = galsim.BoundsI(1, 4, 1, 4)
    im_quad = im_orig[b]
    im_wrap = im.wrap(b)
    np.testing.assert_allclose(im_wrap.array, 4.0 * im_quad.array)

    # The same thing should work no matter where the lower left corner is:
    for xmin, ymin in ((1, 5), (5, 1), (5, 5), (2, 3), (4, 1)):
        b = galsim.BoundsI(xmin, xmin + 3, ymin, ymin + 3)
        im_quad = im_orig[b]
        im = im_orig.copy()
        im_wrap = im.wrap(b)
        np.testing.assert_allclose(
            im_wrap.array,
            4.0 * im_quad.array,
            err_msg="image.wrap(%s) did not match expectation" % b,
        )
        np.testing.assert_allclose(
            im_wrap.array,
            im[b].array,
            err_msg="image.wrap(%s) did not return the right subimage" % b,
        )
        # this test passes even though we do not get a view
        im[b].fill(0)
        np.testing.assert_allclose(
            im_wrap.array,
            im[b].array,
            err_msg="image.wrap(%s) did not return a view of the original" % b,
        )


@timer
def test_wrap_jax_weird_real():
    # Now test where the subimage is not a simple fraction of the original, and all the
    # sizes are different.
    im = galsim.ImageD(17, 23, xmin=0, ymin=0)
    b = galsim.BoundsI(7, 9, 11, 18)
    im_test = galsim.ImageD(b, init_value=0)
    for i in range(17):
        for j in range(23):
            val = np.exp(i / 7.3) + (j / 12.9) ** 3  # Something randomly complicated...
            im[i, j] = val
            # Find the location in the sub-image for this point.
            ii = (i - b.xmin) % (b.xmax - b.xmin + 1) + b.xmin
            jj = (j - b.ymin) % (b.ymax - b.ymin + 1) + b.ymin
            im_test.addValue(ii, jj, val)
    im_wrap = im.wrap(b)
    np.testing.assert_allclose(
        im_wrap.array,
        im_test.array,
        err_msg="image.wrap(%s) did not match expectation" % b,
    )
    np.testing.assert_array_equal(
        im_wrap.array, im[b].array, "image.wrap(%s) did not return the right subimage"
    )
    np.testing.assert_equal(
        im_wrap.bounds, b, "image.wrap(%s) does not have the correct bounds"
    )


@timer
def test_wrap_jax_complex():
    # For complex images (in particular k-space images), we often want the image to be implicitly
    # Hermitian, so we only need to keep around half of it.
    M = 38
    N = 25
    K = 8
    L = 5
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
    b2 = galsim.BoundsI(-K + 1, K, 0, L)
    b3 = galsim.BoundsI(0, K, -L + 1, L)
    # print('b = ',b)
    # print('b2 = ',b2)
    # print('b3 = ',b3)
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

    im_wrap = im.wrap(b)
    # print("im_wrap = ",im_wrap.array)
    np.testing.assert_allclose(
        im_wrap.array,
        im_test.array,
        err_msg="image.wrap(%s) did not match expectation" % b,
    )
    np.testing.assert_array_equal(
        im_wrap.array,
        im[b].array,
        "image.wrap(%s) did not return the right subimage" % b,
    )
    np.testing.assert_equal(
        im_wrap.bounds, b, "image.wrap(%s) does not have the correct bounds" % b
    )

    im2_wrap = im2.wrap(b2, hermitian="y")
    # print('im_test = ',im_test[b2].array)
    # print('im2_wrap = ',im2_wrap.array)
    # print('diff = ',im2_wrap.array-im_test[b2].array)
    np.testing.assert_allclose(
        im2_wrap.array,
        im_test[b2].array,
        err_msg="image.wrap(%s) did not match expectation" % b,
    )
    np.testing.assert_array_equal(
        im2_wrap.array,
        im2[b2].array,
        "image.wrap(%s) did not return the right subimage",
    )
    np.testing.assert_equal(
        im2_wrap.bounds, b2, "image.wrap(%s) does not have the correct bounds"
    )

    im3_wrap = im3.wrap(b3, hermitian="x")
    # print('im_test = ',im_test[b3].array)
    # print('im3_wrap = ',im3_wrap.array)
    # print('diff = ',im3_wrap.array-im_test[b3].array)
    np.testing.assert_allclose(
        im3_wrap.array,
        im_test[b3].array,
        err_msg="image.wrap(%s, hermitian='x') did not match expectation" % b3,
    )
    np.testing.assert_array_equal(
        im3_wrap.array,
        im3[b3].array,
        "image.wrap(%s, hermitian='x') did not match expectation" % b3,
    )
    np.testing.assert_equal(
        im3_wrap.bounds,
        b3,
        "image.wrap(%s, hermitian='x') did not match expectation" % b3,
    )

    b = galsim.BoundsI(-K + 1, K, -L + 1, L)
    b2 = galsim.BoundsI(-K + 1, K, 0, L)
    b3 = galsim.BoundsI(0, K, -L + 1, L)
    assert_raises(TypeError, im.wrap, bounds=None)
    assert_raises(ValueError, im.wrap, b2, hermitian="y")
    assert_raises(ValueError, im.wrap, b, hermitian="invalid")
    assert_raises(ValueError, im.wrap, b3, hermitian="x")

    assert_raises(ValueError, im2.wrap, b, hermitian="y")
    assert_raises(ValueError, im2.wrap, b3, hermitian="y")
    assert_raises(ValueError, im2.wrap, b2, hermitian="invalid")
    assert_raises(ValueError, im3.wrap, b3, hermitian="invalid")
    assert_raises(ValueError, im3.wrap, b, hermitian="x")
    assert_raises(ValueError, im3.wrap, b2, hermitian="x")
