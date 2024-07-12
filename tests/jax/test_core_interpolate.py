import numpy as np

import pytest

from jax_galsim.core.interpolate import akima_interp, akima_interp_coeffs


@pytest.mark.parametrize("fixed_spacing", [True, False])
@pytest.mark.parametrize("use_jax", [True, False])
def test_core_interpolate_akima_interp_linear(use_jax, fixed_spacing):
    x = np.linspace(0, 10, 100)
    y = 10 + 5.0 * x
    x_interp = np.linspace(1, 9, 10)
    y_interp_true = 10 + 5.0 * x_interp
    coeffs = akima_interp_coeffs(x, y, use_jax=use_jax)

    y_interp = akima_interp(x_interp, x, y, coeffs, fixed_spacing=fixed_spacing)
    np.testing.assert_allclose(y_interp, y_interp_true)

    y_interp = akima_interp(x, x, y, coeffs, fixed_spacing=fixed_spacing)
    np.testing.assert_allclose(y_interp, y)


@pytest.mark.parametrize("use_jax", [True, False])
def test_core_interpolate_akima_interp_linear_nonuniform(use_jax):
    x = np.logspace(-5, 1, 100)
    y = 10 + 5.0 * x
    x_interp = np.linspace(1, 9, 10)
    y_interp_true = 10 + 5.0 * x_interp
    coeffs = akima_interp_coeffs(x, y, use_jax=use_jax)

    y_interp = akima_interp(x_interp, x, y, coeffs)
    np.testing.assert_allclose(y_interp, y_interp_true)

    y_interp = akima_interp(x, x, y, coeffs)
    np.testing.assert_allclose(y_interp, y)


@pytest.mark.parametrize("fixed_spacing", [True, False])
@pytest.mark.parametrize("use_jax", [True, False])
def test_core_interpolate_akima_interp_cosexp(use_jax, fixed_spacing):
    def _func(x):
        return np.cos(x) * np.exp(-0.1 * x)

    npts = [4000, 8000, 16000]
    errs = []
    for n in npts:
        x = np.linspace(0, 10, n)
        y = _func(x)
        x_interp = np.linspace(1, 9, 10)
        y_interp_true = _func(x_interp)
        coeffs = akima_interp_coeffs(x, y, use_jax=use_jax)

        y_interp = akima_interp(x_interp, x, y, coeffs, fixed_spacing=fixed_spacing)
        np.testing.assert_allclose(y_interp, y_interp_true)

        errs.append(np.max(np.abs(y_interp - y_interp_true)))

        y_interp = akima_interp(x, x, y, coeffs, fixed_spacing=fixed_spacing)
        np.testing.assert_allclose(y_interp, y)

    for i in range(1, len(errs)):
        assert errs[i] < errs[i - 1]
