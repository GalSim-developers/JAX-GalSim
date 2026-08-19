import os
import shutil
import tempfile

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

    # file_name is not carried through the tree, so a rebuilt instance loses it
    assert rebuilt.file_name is None

    # Pass the object itself as an argument to a jitted function. Two separate
    # instances share a tree structure, so the second call reuses the trace.
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


@requires_des_data
@timer
def test_des_psfex_vmap_over_multiple_psf_models():
    """A batch of *different* PSFEx models can be evaluated in one jit/vmap.

    The PSFEx data is traced, so several models -- including ones read from
    different files -- stack into a single PyTree and are evaluated by one
    compiled kernel. This is the case where a simulation draws galaxies whose
    PSFs come from different exposures. Note that ``fit_order``/``fit_size``
    are static, so a batch must share a polynomial degree.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # A second file on disk, so the two models genuinely differ by file
        # name (which must not be part of the tree for this to work).
        other_name = "other_psfcat.psf"
        shutil.copyfile(
            os.path.join(DES_DATA_DIR, PSFEX_FILE), os.path.join(tmpdir, other_name)
        )

        psf_a = galsim.des.DES_PSFEx(PSFEX_FILE, dir=DES_DATA_DIR)
        psf_b = galsim.des.DES_PSFEx(other_name, dir=tmpdir)
        # Only one PSFEx file ships with the test data, so perturb the second
        # model's basis to stand in for a different exposure's solution.
        psf_b.basis = psf_b.basis * 1.05

        # Stacking would raise if any per-model data were static auxiliary data.
        batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), psf_a, psf_b)

        xs = jnp.array([456.0, 1024.0])
        ys = jnp.array([789.0, 2048.0])
        out = jax.jit(
            jax.vmap(lambda model, x, y: model.getPSFArray(galsim.PositionD(x, y)))
        )(batch, xs, ys)

        assert out.shape[0] == 2

        # Each batch element matches that model evaluated on its own.
        for i, (model, x, y) in enumerate(
            ((psf_a, 456.0, 789.0), (psf_b, 1024.0, 2048.0))
        ):
            np.testing.assert_allclose(
                np.asarray(out[i]),
                np.asarray(model.getPSFArray(galsim.PositionD(x, y))),
                rtol=0,
                atol=1e-6,
            )

        # The two models must give different answers, i.e. the basis really is
        # batched over rather than baked in as a constant.
        assert not np.allclose(np.asarray(out[0]), np.asarray(out[1]))


if __name__ == "__main__":
    test_des_psfex_getPSFArray_vs_galsim()
    test_des_psfex_getPSF_drawn_image_vs_galsim()
    test_des_psfex_pytree_roundtrip_and_traced_arg()
    test_des_psfex_is_jittable_vmappable_differentiable()
    test_des_psfex_vmap_over_multiple_psf_models()
    print("all DES_PSFEx tests passed")
