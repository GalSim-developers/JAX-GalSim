import jax_galsim as galsim
import jax

identity = jax.jit(lambda x: x)


def test_position_jitting():
    obj = galsim.PositionD(1.0, 2.0)

    def test_eq(self, other):
        return self.x == other.x and self.y == other.y

    assert test_eq(identity(obj), obj)


def test_position_jit():
    """Tests that positions are preserved and correctly transformed through jit"""
    pos = galsim.PositionD(1.0, 2.0)

    @jax.jit
    def fun(p):
        pos2 = galsim.PositionD(1.0, 1.0)
        return p + pos2

    # Because this function is jitted, the inputs and outputs
    # will have to go through flattening unflatenning.
    pos3 = fun(pos)

    # Check that the result is the same compared to not jitting
    assert pos3 == (pos + galsim.PositionD(1.0, 1.0))
