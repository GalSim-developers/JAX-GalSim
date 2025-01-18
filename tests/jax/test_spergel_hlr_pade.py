import numpy as np
import pytest

from jax_galsim.core.testing import time_code_block
from jax_galsim.spergel import _spergel_hlr_pade, calculateFluxRadius


@pytest.mark.parametrize(
    "nu", [-0.85, -0.6, -0.5, 0.0, 0.1, 0.5, 1.0, 1.1, 1.5, 2, 2.7, 4.0]
)
def test_spergel_hlr_pade_correct(nu):
    np.testing.assert_allclose(_spergel_hlr_pade(nu), calculateFluxRadius(0.5, nu))


@pytest.mark.parametrize(
    "nu", [-0.85, -0.6, -0.5, 0.0, 0.1, 0.5, 1.0, 1.1, 1.5, 2, 2.7, 4.0]
)
def test_spergel_hlr_pade_time(nu):
    print("\n", end="", flush=True)

    with time_code_block("root finding") as tr_rf:
        for _ in range(100):
            calculateFluxRadius(0.5, nu)

    with time_code_block("pade") as tr_pade:
        for _ in range(100):
            _spergel_hlr_pade(nu)

    assert tr_rf.dt > tr_pade.dt
