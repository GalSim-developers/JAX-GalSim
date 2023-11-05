import galsim as ref_galsim
import numpy as np
import pytest

import jax_galsim


def test_convolve_temp():
    """Validates convolutions against reference GalSim
    This test is to be removed once we can execute test_convolve.py in
    its entirety.
    """
    dx = 0.2
    fwhm_backwards_compatible = 1.0927449310213702

    # First way to do a convolution
    def conv1(galsim):
        psf = galsim.Gaussian(fwhm=fwhm_backwards_compatible, flux=1)
        pixel = galsim.Pixel(scale=dx, flux=1.0)
        conv = galsim.Convolve([psf, pixel], real_space=False)
        return conv.drawImage(
            nx=32, ny=32, scale=dx, method="sb", use_true_center=False
        ).array

    np.testing.assert_array_almost_equal(
        conv1(ref_galsim),
        conv1(jax_galsim),
        5,
        err_msg="Gaussian convolved with Pixel disagrees with expected result",
    )

    # Second way of doing a convolution
    def conv2(galsim):
        psf = galsim.Gaussian(fwhm=fwhm_backwards_compatible, flux=1)
        pixel = galsim.Pixel(scale=dx, flux=1.0)
        conv = galsim.Convolve(psf, pixel, real_space=False)
        return conv.drawImage(
            nx=32, ny=32, scale=dx, method="sb", use_true_center=False
        ).array

    np.testing.assert_array_almost_equal(
        conv2(ref_galsim),
        conv2(jax_galsim),
        5,
        err_msg=" GSObject Convolve(psf,pixel) disagrees with expected result",
    )


def test_shearconvolve_temp():
    """Verifies that a convolution of a Sheared Gaussian and a Box Profile
    return the expected results.
    """
    e1 = 0.05
    e2 = 0.0
    dx = 0.2

    def func(galsim):
        psf = (
            galsim.Gaussian(flux=1, sigma=1)
            .shear(e1=e1, e2=e2)
            .rotate(33 * galsim.degrees)
            .shift(0.1, 0.4)
            * 1.1
        )
        pixel = galsim.Pixel(scale=dx, flux=1.0)
        conv = galsim.Convolve([psf, pixel])
        return conv.drawImage(
            nx=32, ny=32, scale=dx, method="sb", use_true_center=False
        ).array

    np.testing.assert_array_almost_equal(
        func(ref_galsim),
        func(jax_galsim),
        5,
        err_msg="Using GSObject Convolve([psf,pixel]) disagrees with expected result",
    )


@pytest.mark.parametrize(
    "obj1",
    [
        jax_galsim.Gaussian(fwhm=1.0),
        jax_galsim.Pixel(scale=1.0),
        jax_galsim.Exponential(scale_radius=1.0),
        jax_galsim.Exponential(half_light_radius=1.0),
        jax_galsim.Moffat(fwhm=1.0, beta=3),
        jax_galsim.Moffat(scale_radius=1.0, beta=3),
        jax_galsim.Shear(g1=0.1, g2=0.2),
        jax_galsim.PositionD(x=0.1, y=0.2),
        jax_galsim.BoundsI(xmin=0, xmax=1, ymin=0, ymax=1),
        jax_galsim.BoundsD(xmin=0, xmax=1, ymin=0, ymax=1),
        jax_galsim.ShearWCS(0.2, jax_galsim.Shear(g1=0.1, g2=0.2)),
        jax_galsim.Delta(),
        jax_galsim.Nearest(),
        jax_galsim.Lanczos(3),
        jax_galsim.Lanczos(3, conserve_dc=False),
        jax_galsim.Quintic(),
        jax_galsim.Linear(),
        jax_galsim.Cubic(),
        jax_galsim.SincInterpolant(),
    ],
)
def test_pickling_eval_repr(obj1):
    """This test is here until we run all of the galsim tests which cover this one."""
    # test copied from galsim
    import copy
    import pickle
    from collections.abc import Hashable
    from numbers import Complex, Integral, Real  # noqa: F401

    # In case the repr uses these:
    from numpy import (  # noqa: F401
        array,
        complex64,
        complex128,
        float32,
        float64,
        int16,
        int32,
        ndarray,
        uint16,
        uint32,
    )

    def func(x):
        return x

    print("Try pickling ", str(obj1))

    # print('pickled obj1 = ',pickle.dumps(obj1))
    obj2 = pickle.loads(pickle.dumps(obj1))
    assert obj2 is not obj1
    # print('obj1 = ',repr(obj1))
    # print('obj2 = ',repr(obj2))
    f1 = func(obj1)
    f2 = func(obj2)
    # print('func(obj1) = ',repr(f1))
    # print('func(obj2) = ',repr(f2))
    assert f1 == f2

    # Check that == works properly if the other thing isn't the same type.
    assert f1 != object()
    assert object() != f1

    # Test the hash values are equal for two equivalent objects.
    if isinstance(obj1, Hashable):
        # print('hash = ',hash(obj1),hash(obj2))
        assert hash(obj1) == hash(obj2)

    obj3 = copy.copy(obj1)
    assert obj3 is not obj1
    random = hasattr(obj1, "rng") or "rng" in repr(obj1)
    if not random:  # Things with an rng attribute won't be identical on copy.
        f3 = func(obj3)
        assert f3 == f1

    obj4 = copy.deepcopy(obj1)
    assert obj4 is not obj1
    f4 = func(obj4)
    if random:
        f1 = func(obj1)
    # print('func(obj1) = ',repr(f1))
    # print('func(obj4) = ',repr(f4))
    assert f4 == f1  # But everything should be identical with deepcopy.

    # Also test that the repr is an accurate representation of the object.
    # The gold standard is that eval(repr(obj)) == obj.  So check that here as well.
    # A few objects we don't expect to work this way in GalSim; when testing these, we set the
    # `irreprable` kwarg to true.  Also, we skip anything with random deviates since these don't
    # respect the eval/repr roundtrip.

    if not random:
        # A further complication is that the default numpy print options do not lead to sufficient
        # precision for the eval string to exactly reproduce the original object, and start
        # truncating the output for relatively small size arrays.  So we temporarily bump up the
        # precision and truncation threshold for testing.
        # print(repr(obj1))
        with ref_galsim.utilities.printoptions(precision=20, threshold=np.inf):
            obj5 = eval(repr(obj1))
        # print('obj1 = ',repr(obj1))
        # print('obj5 = ',repr(obj5))
        f5 = func(obj5)
        # print('f1 = ',f1)
        # print('f5 = ',f5)
        assert f5 == f1, "func(obj1) = %r\nfunc(obj5) = %r" % (f1, f5)
    else:
        # Even if we're not actually doing the test, still make the repr to check for syntax errors.
        repr(obj1)
