import galsim as gs
import numpy as np
import pytest

import jax_galsim as jgs


@pytest.mark.parametrize(
    "attr",
    [
        "nu",
        "scale_radius",
        "maxk",
        pytest.param(
            "stepk",
            marks=pytest.mark.xfail(
                reason="GalSim has a bug in its stepk routine. See https://github.com/GalSim-developers/GalSim/issues/1324"
            ),
        ),
        "half_light_radius",
    ],
)
@pytest.mark.parametrize("nu", [-0.6, -0.5, 0.0, 0.1, 0.5, 1.0, 1.1, 1.5, 2, 2.7])
@pytest.mark.parametrize("scale_radius", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_spergel_comp_galsim_properties(nu, scale_radius, attr):
    s_jgs = jgs.Spergel(nu=nu, scale_radius=scale_radius)
    s_gs = gs.Spergel(nu=nu, scale_radius=scale_radius)

    assert s_jgs.gsparams.folding_threshold == s_gs.gsparams.folding_threshold
    assert s_jgs.gsparams.stepk_minimum_hlr == s_gs.gsparams.stepk_minimum_hlr

    np.testing.assert_allclose(getattr(s_jgs, attr), getattr(s_gs, attr), rtol=1e-5)


@pytest.mark.parametrize("nu", [-0.6, -0.5, 0.0, 0.1, 0.5, 1.0, 1.1, 1.5, 2, 2.7])
@pytest.mark.parametrize("scale_radius", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_spergel_comp_galsim_flux_radius(nu, scale_radius):
    s_jgs = jgs.Spergel(nu=nu, scale_radius=scale_radius)
    s_gs = gs.Spergel(nu=nu, scale_radius=scale_radius)

    np.testing.assert_allclose(
        s_jgs.calculateFluxRadius(0.8),
        s_gs.calculateFluxRadius(0.8),
        rtol=1e-5,
    )


@pytest.mark.parametrize("nu", [-0.6, -0.5, 0.0, 0.1, 0.5, 1.0, 1.1, 1.5, 2, 2.7])
@pytest.mark.parametrize("scale_radius", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_spergel_comp_galsim_integ_flux(nu, scale_radius):
    s_jgs = jgs.Spergel(nu=nu, scale_radius=scale_radius)
    s_gs = gs.Spergel(nu=nu, scale_radius=scale_radius)

    np.testing.assert_allclose(
        s_jgs.calculateIntegratedFlux(0.8),
        s_gs.calculateIntegratedFlux(0.8),
        rtol=1e-5,
    )


@pytest.mark.parametrize("kx", np.linspace(-1, 1, 13).tolist())
@pytest.mark.parametrize("ky", np.linspace(-1, 1, 13).tolist())
@pytest.mark.parametrize("nu", [-0.6, -0.5, 0.0, 0.1, 0.5, 1.0, 1.1, 1.5, 2, 2.7])
@pytest.mark.parametrize("scale_radius", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_spergel_comp_galsim_kvalue(nu, scale_radius, kx, ky):
    s_jgs = jgs.Spergel(nu=nu, scale_radius=scale_radius)
    s_gs = gs.Spergel(nu=nu, scale_radius=scale_radius)

    np.testing.assert_allclose(s_jgs.kValue(kx, ky), s_gs.kValue(kx, ky), rtol=1e-5)


@pytest.mark.parametrize("x", np.linspace(-1, 1, 13).tolist())
@pytest.mark.parametrize("y", np.linspace(-1, 1, 13).tolist())
@pytest.mark.parametrize("nu", [-0.6, -0.5, 0.0, 0.1, 0.5, 1.0, 1.1, 1.5, 2, 2.7])
@pytest.mark.parametrize("scale_radius", [0.5, 1.0, 1.5, 2.0, 2.5])
def test_spergel_comp_galsim_xvalue(nu, scale_radius, x, y):
    s_jgs = jgs.Spergel(nu=nu, scale_radius=scale_radius)
    s_gs = gs.Spergel(nu=nu, scale_radius=scale_radius)

    np.testing.assert_allclose(s_jgs.xValue(x, y), s_gs.xValue(x, y), rtol=1e-5)
