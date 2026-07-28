import os

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from galsim.utilities import timer

import jax_galsim as galsim

DES_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "GalSim", "tests", "des_data"
)
PSFEX_FILE = "DECam_00154912_12_psfcat.psf"

# A few positions spread across the DECam chip.
POSITIONS = [(100.0, 100.0), (456.0, 789.0), (1024.0, 2048.0), (1700.0, 3500.0)]


def _have_des_data():
    return os.path.isfile(os.path.join(DES_DATA_DIR, PSFEX_FILE))


requires_des_data = pytest.mark.skipif(
    not _have_des_data(),
    reason="DES test data (tests/GalSim submodule) not available",
)


@requires_des_data
@timer
def test_des_psfex_getPSFArray_vs_galsim():
    """The interpolated PSF array should match reference GalSim."""
    ref = _galsim.des.DES_PSFEx(PSFEX_FILE, dir=DES_DATA_DIR)
    jgs = galsim.des.DES_PSFEx(PSFEX_FILE, dir=DES_DATA_DIR)

    assert jgs.fit_order == ref.fit_order
    assert jgs.fit_size == ref.fit_size
    np.testing.assert_allclose(jgs.sample_scale, ref.sample_scale)

    for x, y in POSITIONS:
        a = np.asarray(jgs.getPSFArray(galsim.PositionD(x, y)))
        b = ref.getPSFArray(_galsim.PositionD(x, y))
        # float32 interpolation, so compare at ~single precision.
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


@requires_des_data
@timer
def test_des_psfex_getPSF_drawn_image_vs_galsim():
    """The effective-PSF image (drawn with no_pixel) should match GalSim."""
    ref = _galsim.des.DES_PSFEx(PSFEX_FILE, dir=DES_DATA_DIR)
    jgs = galsim.des.DES_PSFEx(PSFEX_FILE, dir=DES_DATA_DIR)

    for x, y in POSITIONS:
        # PSFEx PSFs already include the pixel, so draw with method='no_pixel'.
        gimg = ref.getPSF(_galsim.PositionD(x, y)).drawImage(
            nx=25, ny=25, scale=0.2, method="no_pixel"
        )
        jimg = jgs.getPSF(galsim.PositionD(x, y)).drawImage(
            nx=25, ny=25, scale=0.2, method="no_pixel"
        )
        np.testing.assert_allclose(
            np.asarray(jimg.array), gimg.array, rtol=0, atol=1e-6
        )


@requires_des_data
@timer
def test_des_psfex_pytree_roundtrip_and_traced_arg():
    """DES_PSFEx is a registered PyTree: it round-trips through flatten/
    unflatten (without re-reading the file) and can be passed as an argument to
    a transformed function, including two distinct-but-equal instances."""
    jgs = galsim.des.DES_PSFEx(PSFEX_FILE, dir=DES_DATA_DIR)

    leaves, treedef = jax.tree_util.tree_flatten(jgs)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(np.asarray(rebuilt.basis), np.asarray(jgs.basis))
    for x, y in POSITIONS:
        np.testing.assert_allclose(
            np.asarray(rebuilt.getPSFArray(galsim.PositionD(x, y))),
            np.asarray(jgs.getPSFArray(galsim.PositionD(x, y))),
            rtol=0,
            atol=1e-6,
        )

    # Pass the object itself as a jitted argument. Using two distinct instances
    # exercises the hashability of the (auxiliary) PSFEx data in the treedef.
    f = jax.jit(lambda obj, x, y: obj.getPSFArray(galsim.PositionD(x, y)))
    jgs2 = galsim.des.DES_PSFEx(PSFEX_FILE, dir=DES_DATA_DIR)
    a1 = f(jgs, 456.0, 789.0)
    a2 = f(jgs2, 456.0, 789.0)
    np.testing.assert_allclose(np.asarray(a1), np.asarray(a2), rtol=0, atol=1e-6)


@requires_des_data
@timer
def test_des_psfex_is_jittable_vmappable_differentiable():
    """getPSFArray should support jit, vmap, and grad over the image position."""
    jgs = galsim.des.DES_PSFEx(PSFEX_FILE, dir=DES_DATA_DIR)

    def psf_sum(x, y):
        return jnp.sum(jgs.getPSFArray(galsim.PositionD(x, y)))

    # jit
    jitted = jax.jit(lambda x, y: jgs.getPSFArray(galsim.PositionD(x, y)))
    arr = jitted(456.0, 789.0)
    ref = jgs.getPSFArray(galsim.PositionD(456.0, 789.0))
    np.testing.assert_allclose(np.asarray(arr), np.asarray(ref), rtol=0, atol=1e-6)

    # vmap over a batch of positions
    xs = jnp.array([p[0] for p in POSITIONS])
    ys = jnp.array([p[1] for p in POSITIONS])
    batched = jax.vmap(lambda x, y: jgs.getPSFArray(galsim.PositionD(x, y)))(xs, ys)
    assert batched.shape[0] == len(POSITIONS)

    # grad w.r.t. position must be finite (the PSF is differentiable in position)
    gx = jax.grad(psf_sum, argnums=0)(456.0, 789.0)
    assert np.isfinite(float(gx))


if __name__ == "__main__":
    test_des_psfex_getPSFArray_vs_galsim()
    test_des_psfex_getPSF_drawn_image_vs_galsim()
    test_des_psfex_pytree_roundtrip_and_traced_arg()
    test_des_psfex_is_jittable_vmappable_differentiable()
    print("all DES_PSFEx tests passed")
