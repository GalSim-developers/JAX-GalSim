import hashlib

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

FRAC_TEST_TO_KEEP = 0.5


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
    nherm = kim.array.shape[0]
    minherm = kim.bounds.ymin
    kimherm = jax_galsim.Image(
        jnp.zeros((kim.array.shape[0], kim.array.shape[0]), dtype=complex),
        xmin=minherm,
        ymin=minherm,
    )
    for y in range(kimherm.bounds.ymin, kimherm.bounds.ymax + 1):
        for x in range(kimherm.bounds.xmin, kimherm.bounds.xmax + 1):
            if x >= 0:
                kimherm[x, y] = kim[x, y]
            else:
                if y == minherm:
                    kimherm[x, y] = kim[-x, y].conj()
                else:
                    kimherm[x, y] = kim[-x, -y].conj()
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
                kimherm(x + minherm, y + minherm),
            )


def test_interpolatedimage_utils_stepk_maxk():
    hlr = 0.5
    fwhm = 0.9
    scale = 0.2

    ref_array = (
        _galsim.Convolve(
            _galsim.Exponential(half_light_radius=hlr),
            _galsim.Gaussian(fwhm=fwhm),
        )
        .drawImage(
            nx=53,
            ny=53,
            scale=scale,
        )
        .array.astype(np.float64)
    )

    gimage_in = _galsim.Image(ref_array)
    jgimage_in = jax_galsim.Image(ref_array)
    gii = _galsim.InterpolatedImage(gimage_in, scale=scale)
    jgii = jax_galsim.InterpolatedImage(jgimage_in, scale=scale)

    np.testing.assert_allclose(jgii.stepk, gii.stepk, rtol=0, atol=1e-6)
    np.testing.assert_allclose(jgii.maxk, gii.maxk, rtol=0, atol=1e-6)


@pytest.mark.parametrize("x_interp", ["lanczos15", "quintic"])
@pytest.mark.parametrize("normalization", ["sb", "flux"])
@pytest.mark.parametrize("use_true_center", [True, False])
@pytest.mark.parametrize(
    "wcs",
    [
        _galsim.PixelScale(0.2),
        _galsim.JacobianWCS(0.21, 0.03, -0.04, 0.23),
        _galsim.AffineTransform(-0.03, 0.21, 0.18, 0.01, _galsim.PositionD(0.3, -0.4)),
    ],
)
@pytest.mark.parametrize(
    "offset_x",
    [
        -4.35,
        -0.45,
        0.0,
        0.67,
        3.78,
    ],
)
@pytest.mark.parametrize(
    "offset_y",
    [
        -2.12,
        -0.33,
        0.0,
        0.12,
        1.45,
    ],
)
@pytest.mark.parametrize(
    "ref_array",
    [
        _galsim.Gaussian(fwhm=0.9).drawImage(nx=33, ny=33, scale=0.2).array,
        _galsim.Gaussian(fwhm=0.9).drawImage(nx=32, ny=32, scale=0.2).array,
    ],
)
@pytest.mark.parametrize("method", ["kValue", "xValue"])
def test_interpolatedimage_utils_comp_to_galsim(
    method,
    ref_array,
    offset_x,
    offset_y,
    wcs,
    use_true_center,
    normalization,
    x_interp,
):
    seed = max(
        abs(
            int(
                hashlib.sha1(
                    f"{method}{ref_array}{offset_x}{offset_y}{wcs}{use_true_center}{normalization}{x_interp}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                16,
            )
        )
        % (10**7),
        1,
    )

    rng = np.random.RandomState(seed=seed)
    if rng.uniform() < FRAC_TEST_TO_KEEP:
        pytest.skip(
            "Skipping `test_interpolatedimage_utils_comp_to_galsim` case at random to save time."
        )

    gimage_in = _galsim.Image(ref_array, scale=0.2)
    jgimage_in = jax_galsim.Image(ref_array, scale=0.2)

    gii = _galsim.InterpolatedImage(
        gimage_in,
        wcs=wcs,
        offset=_galsim.PositionD(offset_x, offset_y),
        use_true_center=use_true_center,
        normalization=normalization,
        x_interpolant=x_interp,
    )
    jgii = jax_galsim.InterpolatedImage(
        jgimage_in,
        wcs=jax_galsim.BaseWCS.from_galsim(wcs),
        offset=jax_galsim.PositionD(offset_x, offset_y),
        use_true_center=use_true_center,
        normalization=normalization,
        x_interpolant=x_interp,
    )

    np.testing.assert_allclose(jgii.stepk, gii.stepk, rtol=0, atol=1e-6)
    np.testing.assert_allclose(jgii.maxk, gii.maxk, rtol=0.5, atol=0)
    kxvals = [
        (0, 0),
        (-5, -5),
        (-10, 10),
        (1, 1),
        (1, -2),
        (-1, 0),
        (0, -1),
        (-1, -1),
        (-2, 2),
        (-5, 0),
        (3, -4),
        (-3, 4),
    ]
    for x, y in kxvals:
        if method == "kValue":
            dk = jgii._original._kim.scale * rng.uniform(low=0.5, high=1.5)
            np.testing.assert_allclose(
                jgii.kValue(x * dk, y * dk),
                gii.kValue(x * dk, y * dk),
                err_msg=f"kValue mismatch: wcs={wcs}, x={x}, y={y}",
            )
        else:
            dx = jnp.sqrt(jgii._original._wcs.pixelArea()) * rng.uniform(
                low=0.5, high=1.5
            )
            np.testing.assert_allclose(
                jgii.xValue(x * dx, y * dx),
                gii.xValue(x * dx, y * dx),
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
    out._array = jnp.asarray(
        np.fft.fftshift(np.fft.rfft2(np.fft.fftshift(ximage.array)), axes=0)
    )
    out *= dx * dx
    out.setOrigin(0, -No2)
    return out


@pytest.mark.parametrize("n", [5, 4])
def test_interpolatedimage_utils_jax_galsim_fft_vs_galsim_fft(n):
    rng = np.random.RandomState(42)
    arr = rng.normal(size=(n, n))
    im = jax_galsim.Image(arr, scale=1)
    kim = im.calculate_fft()
    xkim = kim.calculate_inverse_fft()
    np.testing.assert_allclose(im.array, xkim[im.bounds].array)

    np_kim = _compute_fft_with_numpy_jax_galsim(im)
    np.testing.assert_allclose(kim.array, np_kim.array)

    rng = np.random.RandomState(42)
    arr = rng.normal(size=(n, n))
    gim = _galsim.Image(arr, scale=1)
    gkim = gim.calculate_fft()
    gxkim = gkim.calculate_inverse_fft()
    np.testing.assert_allclose(gxkim[gim.bounds].array, gim.array, rtol=0, atol=1e-12)
    np.testing.assert_allclose(im.array, gim.array, rtol=0, atol=1e-12)
    np.testing.assert_allclose(kim.array, gkim.array, rtol=0, atol=1e-12)
    np.testing.assert_allclose(xkim.array, gxkim.array, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "interp",
    [
        Nearest(),
        Linear(),
        Cubic(),
        Quintic(),
        Lanczos(3, conserve_dc=True),
        Lanczos(3, conserve_dc=False),
        Lanczos(7),
    ],
)
def test_interpolatedimage_interpolant_sample(interp):
    """Sample from the interpolation kernel and compare a histogram of the samples to the expected fistribution."""
    from jax_galsim.photon_array import PhotonArray
    from jax_galsim.random import BaseDeviate

    rng = BaseDeviate(999)

    ntot = 1000000
    photons = PhotonArray(ntot)
    interp._shoot(photons, rng)

    h, bins = jnp.histogram(photons.x, bins=500)
    mid = (bins[1:] + bins[:-1]) / 2.0
    dx = bins[1:] - bins[:-1]
    yv = (
        jnp.abs(interp._xval_noraise(mid))
        * dx
        * ntot
        * 1.0
        / (interp.positive_flux + interp.negative_flux)
    )
    msk = yv > 100
    fdev = np.abs(h - yv) / np.abs(np.sqrt(yv))
    np.testing.assert_allclose(fdev[msk], 0, rtol=0, atol=5.0, err_msg=f"{interp}")
    np.testing.assert_allclose(fdev[~msk], 0, rtol=0, atol=15.0, err_msg=f"{interp}")


@pytest.mark.parametrize("x_interp", ["lanczos15", "quintic"])
@pytest.mark.parametrize("normalization", ["sb", "flux"])
@pytest.mark.parametrize("use_true_center", [True, False])
@pytest.mark.parametrize(
    "wcs",
    [
        _galsim.PixelScale(0.2),
        _galsim.JacobianWCS(0.21, 0.03, -0.04, 0.23),
        _galsim.AffineTransform(-0.03, 0.21, 0.18, 0.01, _galsim.PositionD(0.3, -0.4)),
    ],
)
@pytest.mark.parametrize(
    "offset_x",
    [
        -4.35,
        -0.45,
        0.0,
        0.67,
        3.78,
    ],
)
@pytest.mark.parametrize(
    "offset_y",
    [
        -2.12,
        -0.33,
        0.0,
        0.12,
        1.45,
    ],
)
@pytest.mark.parametrize(
    "ref_array",
    [
        _galsim.Gaussian(fwhm=0.9)
        .shear(g1=0.3, g2=-0.2)
        .drawImage(nx=33, ny=33, scale=0.2)
        .array,
        _galsim.Gaussian(fwhm=0.9)
        .shear(g1=-0.03, g2=0.1)
        .drawImage(nx=32, ny=32, scale=0.2)
        .array,
    ],
)
def test_interpolatedimage_utils_comp_stepk_maxk_to_galsim(
    ref_array,
    offset_x,
    offset_y,
    wcs,
    use_true_center,
    normalization,
    x_interp,
):
    seed = max(
        abs(
            int(
                hashlib.sha1(
                    f"{ref_array}{offset_x}{offset_y}{wcs}{use_true_center}{normalization}{x_interp}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                16,
            )
        )
        % (10**7),
        1,
    )

    rng = np.random.RandomState(seed=seed)
    if rng.uniform() < FRAC_TEST_TO_KEEP:
        pytest.skip(
            "Skipping `test_interpolatedimage_utils_comp_stepk_maxk_to_galsim` case at random to save time."
        )

    nse = rng.uniform(size=ref_array.shape) * ref_array.max() * 0.05

    gimage_in = _galsim.Image(ref_array + nse, scale=0.2)
    jgimage_in = jax_galsim.Image(ref_array + nse, scale=0.2)

    np.testing.assert_allclose(gimage_in.center.x, jgimage_in.center.x)
    np.testing.assert_allclose(gimage_in.center.y, jgimage_in.center.y)

    gii = _galsim.InterpolatedImage(
        gimage_in,
        wcs=wcs,
        offset=_galsim.PositionD(offset_x, offset_y),
        use_true_center=use_true_center,
        normalization=normalization,
        x_interpolant=x_interp,
        flux=20,
    )
    jgii = jax_galsim.InterpolatedImage(
        jgimage_in,
        wcs=jax_galsim.BaseWCS.from_galsim(wcs),
        offset=jax_galsim.PositionD(offset_x, offset_y),
        use_true_center=use_true_center,
        normalization=normalization,
        x_interpolant=x_interp,
        flux=20,
    )

    gthresh = (1.0 - gii.gsparams.folding_threshold) * gii._image_flux
    gR = _galsim._galsim.CalculateSizeContainingFlux(gii._image._image, gthresh)

    from jax_galsim.interpolatedimage import _calculate_size_containing_flux

    jgthresh = (
        1.0 - jgii._original.gsparams.folding_threshold
    ) * jgii._original._image_flux
    jgR = _calculate_size_containing_flux(jgii._original.image, jgthresh)

    lgR = _galsim_stepk_loop(gii._image, gthresh)
    ljgR = _galsim_stepk_loop(jgii._original.image, jgthresh)

    np.testing.assert_allclose(jgii._original.image.center.x, gii._image.center.x)
    np.testing.assert_allclose(jgii._original.image.center.y, gii._image.center.y)
    np.testing.assert_allclose(jgii._original.image(0, 0), gii._image(0, 0))
    np.testing.assert_allclose(jgii._original.image.array.sum(), gii._image.array.sum())
    np.testing.assert_allclose(jgthresh, gthresh, rtol=0, atol=1e-6)
    np.testing.assert_allclose(jgR, gR, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ljgR, gR, rtol=0, atol=1e-6)
    np.testing.assert_allclose(gR, lgR, rtol=0, atol=1e-6)

    np.testing.assert_allclose(jgii.stepk, gii.stepk, rtol=0, atol=1e-6)
    # FIXME: make maxk match
    np.testing.assert_allclose(jgii.maxk, gii.maxk, rtol=0.5, atol=0)


# this is a copy of the galsim C++ algorithm in a pure python
# loop to help with debugging and testing
def _galsim_stepk_loop(im, target_flux):
    if target_flux > 0:
        p = 1.0
    else:
        p = -1.0

    b = im.bounds
    dmax = int(min((b.getXMax() - b.getXMin()) / 2, (b.getYMax() - b.getYMin()) / 2))

    flux = im(0, 0)
    d = 1
    while d <= dmax:
        # Add the left, right, top and bottom sides of box
        for x in range(-d, d):
            # Note: All 4 corners are added exactly once by including x=-d but omitting
            #  x=d from the loop.
            flux += im(x, -d)  # bottom
            flux += im(d, x)  # right
            flux += im(-x, d)  # top
            flux += im(-d, -x)  # left

        if p * flux >= p * target_flux:
            break

        d += 1

    return d + 0.5
