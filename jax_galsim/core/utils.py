import jax
import jax.numpy as jnp


def cast_scalar_to_float(x):
    """Cast the input to a float. Works on python floats and jax arrays."""
    if isinstance(x, float):
        return float(x)
    elif hasattr(x, "astype"):
        return x.astype(float)
    else:
        try:
            return float(x)
        except TypeError as e:
            # needed so that tests of jax_galsim.angle pass
            if "AngleUnit" in str(e):
                raise e

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


def is_equal_with_arrays(x, y):
    """Return True if the data is equal, False otherwise. Handles jax.Array types."""
    if isinstance(x, list):
        if isinstance(y, list) and len(x) == len(y):
            for vx, vy in zip(x, y):
                if not is_equal_with_arrays(vx, vy):
                    return False
            return True
        else:
            return False
    elif isinstance(x, tuple):
        if isinstance(y, tuple) and len(x) == len(y):
            for vx, vy in zip(x, y):
                if not is_equal_with_arrays(vx, vy):
                    return False
            return True
        else:
            return False
    elif isinstance(x, set):
        if isinstance(y, set) and len(x) == len(y):
            for vx, vy in zip(x, y):
                if not is_equal_with_arrays(vx, vy):
                    return False
            return True
        else:
            return False
    elif isinstance(x, dict):
        if isinstance(y, dict) and len(x) == len(y):
            for kx, vx in x.items():
                if kx not in y or (not is_equal_with_arrays(vx, y[kx])):
                    return False
            return True
        else:
            return False
    elif isinstance(x, jax.Array) and jnp.ndim(x) > 0:
        if isinstance(y, jax.Array) and y.shape == x.shape:
            return jnp.array_equal(x, y)
        else:
            return False
    elif (isinstance(x, jax.Array) and jnp.ndim(x) == 0) or (
        isinstance(y, jax.Array) and jnp.ndim(y) == 0
    ):
        return jnp.array_equal(x, y)
    else:
        return x == y


def _recurse_list_to_tuple(x):
    if isinstance(x, list):
        return tuple(_recurse_list_to_tuple(v) for v in x)
    else:
        return x


def ensure_hashable(v):
    """Ensure that the input is hashable. If it is a jax array,
    convert it to a possibly nested tuple or python scalar."""
    if isinstance(v, jax.Array):
        try:
            if len(v.shape) > 0:
                return _recurse_list_to_tuple(v.tolist())
            else:
                return v.item()
        except Exception:
            return v
    else:
        return v
