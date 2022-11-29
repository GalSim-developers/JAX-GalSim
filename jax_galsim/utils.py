import numpy as np
import jax
import jax.numpy as jnp


def _reshape(x, shape):
    if isinstance(x, (int, float, np.ndarray, np.generic)):
        return np.reshape(x, shape)
    else:
        return jnp.reshape(x, shape)

def promote_shapes(*args, shape=()):
    # adapted from lax.lax_numpy
    if len(args) < 2 and not shape:
        return args
    else:
        shapes = [jnp.shape(arg) for arg in args]
        num_dims = len(jax.lax.broadcast_shapes(shape, *shapes))
        return [
            _reshape(arg, (1,) * (num_dims - len(s)) + s) if len(s) < num_dims else arg
            for arg, s in zip(args, shapes)
        ]

def is_prng_key(key):
    try:
        return key.shape == (2,) and key.dtype == np.uint32
    except AttributeError:
        return False


