import jax.numpy as jnp
import numpy as np

import jax_galsim


def _gen_photon_array(n_photons, rng):
    pa = jax_galsim.PhotonArray(n_photons)
    pa.x = rng.uniform(size=n_photons)
    pa.y = rng.uniform(size=n_photons)
    pa.flux = rng.uniform(size=n_photons)
    pa.wavelength = rng.uniform(size=n_photons)
    pa.dxdz = rng.uniform(size=n_photons)
    pa.dydz = rng.uniform(size=n_photons)
    pa.pupil_u = rng.uniform(size=n_photons)
    pa.pupil_v = rng.uniform(size=n_photons)
    pa.time = rng.uniform(size=n_photons)
    pa._nokeep = jnp.array(
        rng.uniform(size=n_photons) > 0.5,
        dtype=bool,
    )
    return pa


def test_photon_array_masking_sort():
    rng = np.random.RandomState(seed=42)
    pa = _gen_photon_array(10, rng)
    pas = jax_galsim.PhotonArray(10)
    pas = pas.copyFrom(pa)

    sinds = jnp.argsort(pas._nokeep)
    pas._sort_by_nokeep()
    np.testing.assert_allclose(pas.x, pa.x[sinds])
    np.testing.assert_allclose(pas.y, pa.y[sinds])
    np.testing.assert_allclose(pas.flux, pa.flux[sinds])
    np.testing.assert_allclose(pas.wavelength, pa.wavelength[sinds])
    np.testing.assert_allclose(pas.dxdz, pa.dxdz[sinds])
    np.testing.assert_allclose(pas.dydz, pa.dydz[sinds])
    np.testing.assert_allclose(pas.pupil_u, pa.pupil_u[sinds])
    np.testing.assert_allclose(pas.pupil_v, pa.pupil_v[sinds])
    np.testing.assert_allclose(pas.time, pa.time[sinds])
    np.testing.assert_allclose(pas._nokeep, pa._nokeep[sinds])

    pas._set_self_at_inds(sinds)
    np.testing.assert_allclose(pas.x, pa.x)
    np.testing.assert_allclose(pas.y, pa.y)
    np.testing.assert_allclose(pas.flux, pa.flux)
    np.testing.assert_allclose(pas.wavelength, pa.wavelength)
    np.testing.assert_allclose(pas.dxdz, pa.dxdz)
    np.testing.assert_allclose(pas.dydz, pa.dydz)
    np.testing.assert_allclose(pas.pupil_u, pa.pupil_u)
    np.testing.assert_allclose(pas.pupil_v, pa.pupil_v)
    np.testing.assert_allclose(pas.time, pa.time)
    np.testing.assert_allclose(pas._nokeep, pa._nokeep)


def test_photon_array_masking_set_num_keep():
    rng = np.random.RandomState(seed=42)
    pa = _gen_photon_array(10, rng)
    pas = jax_galsim.PhotonArray(10)
    pas = pas.copyFrom(pa)

    pas._num_keep = 2
    np.testing.assert_allclose(pas.x, pa.x)
    np.testing.assert_allclose(pas.y, pa.y)
    assert not np.allclose(pas.flux, pa.flux)
    np.testing.assert_allclose(pas.wavelength, pa.wavelength)
    np.testing.assert_allclose(pas.dxdz, pa.dxdz)
    np.testing.assert_allclose(pas.dydz, pa.dydz)
    np.testing.assert_allclose(pas.pupil_u, pa.pupil_u)
    np.testing.assert_allclose(pas.pupil_v, pa.pupil_v)
    np.testing.assert_allclose(pas.time, pa.time)
    assert not np.allclose(pas._nokeep, pa._nokeep)

    assert pas._num_keep == 2
    assert pas._Ntot == 10
    assert pa._num_keep == pa._Ntot - np.sum(pa._nokeep)
    assert pa._num_keep != pas._num_keep


def test_photon_array_masking_copyFrom_flux_handling():
    rng = np.random.RandomState(seed=42)
    pal = _gen_photon_array(6, rng)
    pal._nokeep = jnp.array(
        [True] * 2 + [False] * 4,
        dtype=bool,
    )

    par = _gen_photon_array(4, rng)
    par._nokeep = jnp.array(
        [False] * 1 + [True] * 3,
        dtype=bool,
    )

    pa = jax_galsim.PhotonArray(10)
    pa.copyFrom(pal, slice(0, 6))
    pa.copyFrom(par, slice(6, 10))

    np.testing.assert_allclose(pa.x, np.hstack([pal.x, par.x]))
    np.testing.assert_allclose(pa.y, np.hstack([pal.y, par.y]))
    np.testing.assert_allclose(pa.flux, np.hstack([pal.flux, par.flux]))
    np.testing.assert_allclose(pa.wavelength, np.hstack([pal.wavelength, par.wavelength]))
    np.testing.assert_allclose(pa.dxdz, np.hstack([pal.dxdz, par.dxdz]))
    np.testing.assert_allclose(pa.dydz, np.hstack([pal.dydz, par.dydz]))
    np.testing.assert_allclose(pa.pupil_u, np.hstack([pal.pupil_u, par.pupil_u]))
    np.testing.assert_allclose(pa.pupil_v, np.hstack([pal.pupil_v, par.pupil_v]))
    np.testing.assert_allclose(pa.time, np.hstack([pal.time, par.time]))
    np.testing.assert_allclose(pa._nokeep, np.hstack([pal._nokeep, par._nokeep]))

    assert np.sum(pal.flux) + np.sum(par.flux) == np.sum(pa.flux)
