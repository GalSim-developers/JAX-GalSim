import galsim as _galsim
import galsim as gs
import jax
import numpy as np
import pytest
from test_benchmarks import (
    _run_spergel_bench_conv,
    _run_spergel_bench_conv_jit,
    _run_spergel_bench_kvalue,
    _run_spergel_bench_kvalue_jit,
    _run_spergel_bench_xvalue,
    _run_spergel_bench_xvalue_jit,
)

import jax_galsim as jgs
from jax_galsim.core.testing import time_code_block


@pytest.mark.parametrize(
    "attr",
    [
        "nu",
        "scale_radius",
        "maxk",
        "stepk",
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


def _run_time_test(kind, func):
    if kind == "compile":

        def _run():
            jax.clear_caches()
            func()

    elif kind == "run":
        # run once to compile
        func()

        def _run():
            func()

    else:
        raise ValueError(f"kind={kind} not recognized")

    tot_time = 0
    for _ in range(3):
        with time_code_block(quiet=True) as tr:
            _run()
        tot_time += tr.dt

    return tot_time / 3


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_spergel_comp_galsim_perf_conv(kind):
    dt = _run_time_test(kind, lambda: _run_spergel_bench_conv_jit().block_until_ready())
    print(f"\njax-galsim time: {dt:0.4g} ms")

    dt = _run_time_test(
        kind,
        lambda: _run_spergel_bench_conv(_galsim),
    )
    print(f"    galsim time: {dt:0.4g} ms")


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_spergel_comp_galsim_perf_kvalue(kind):
    dt = _run_time_test(
        kind, lambda: _run_spergel_bench_kvalue_jit().block_until_ready()
    )
    print(f"\njax-galsim time: {dt:0.4g} ms")

    dt = _run_time_test(
        kind,
        lambda: _run_spergel_bench_kvalue(_galsim),
    )
    print(f"    galsim time: {dt:0.4g} ms")


@pytest.mark.parametrize("kind", ["compile", "run"])
def test_spergel_comp_galsim_perf_xvalue(kind):
    dt = _run_time_test(
        kind, lambda: _run_spergel_bench_xvalue_jit().block_until_ready()
    )
    print(f"\njax-galsim time: {dt:0.4g} ms")

    dt = _run_time_test(
        kind,
        lambda: _run_spergel_bench_xvalue(_galsim),
    )
    print(f"    galsim time: {dt:0.4g} ms")
