import jax_galsim
import jax


def test_position_jit():
    """Tests that positions are preserved and correctly transformed through jit"""
    pos = jax_galsim.PositionD(1.0, 2.0)

    @jax.jit
    def fun(p):
        pos2 = jax_galsim.PositionD(1.0, 1.0)
        return p + pos2

    # Because this function is jitted, the inputs and outputs
    # will have to go through flattening unflatenning.
    pos3 = fun(pos)

    # Check that the result is the same compared to not jitting
    assert pos3 == (pos + jax_galsim.PositionD(1.0, 1.0))
