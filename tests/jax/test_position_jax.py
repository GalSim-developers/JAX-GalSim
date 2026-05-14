import jax
import pytest

import jax_galsim


def test_position_jax_int_raises_in_jit():

    @jax.jit
    def _make_pos(x, y):
        return jax_galsim.PositionI(x, y)

    with pytest.raises(Exception):
        _make_pos(1.2, 23)

    with pytest.raises(Exception):
        _make_pos(12, 2.3)

    pos = _make_pos(1, 2)
    assert pos.x == 1
    assert pos.y == 2
