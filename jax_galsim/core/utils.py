import jax


def cast_scalar_to_float(x):
    """Cast the input to a float. Works on python floats and jax arrays."""
    if isinstance(x, float):
        return float(x)
    elif hasattr(x, "astype"):
        return x.astype(float)
    else:
        try:
            return float(x)
        except TypeError:
            return x


def cast_scalar_to_int(x):
    """Cast the input to an int. Works on python floats/ints and jax arrays."""
    if isinstance(x, jax.Array):
        return x.astype(int)
    elif hasattr(x, "astype"):
        return x.astype(int)
    else:
        try:
            return int(x)
        except TypeError:
            return x


def ensure_hashable(v):
    """Ensure that the input is hashable. If it is a jax array, convert it to a tuple or float."""
    if isinstance(v, jax.Array):
        if len(v.shape) > 0:
            return tuple(v.tolist())
        else:
            return v.item()
    else:
        return v
