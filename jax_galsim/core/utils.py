from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten


def has_tracers(x):
    """Return True if the input item is a JAX tracer, False otherwise."""
    for item in tree_flatten(x)[0]:
        if (
            isinstance(item, jax.core.Tracer)
            or type(item) is object
            # or isinstance(item, jax.core.ShapedArray)
            # or isinstance(item, str)
        ):
            return True
    return False


@jax.jit
def compute_major_minor_from_jacobian(jac):
    h1 = jnp.hypot(jac[0, 0] + jac[1, 1], jac[0, 1] - jac[1, 0])
    h2 = jnp.hypot(jac[0, 0] - jac[1, 1], jac[0, 1] + jac[1, 0])
    major = 0.5 * jnp.abs(h1 + h2)
    minor = 0.5 * jnp.abs(h1 - h2)
    return major, minor


def _cast_to_array_scalar(x, dtype=None):
    """Cast the input to an array scalar. Works on python scalars, iterables and jax arrays.
    For iterables it always takes the first element after a call to .ravel()"""
    if dtype is None:
        if hasattr(x, "dtype"):
            dtype = x.dtype
        else:
            dtype = float

    if isinstance(x, jax.Array):
        return jnp.atleast_1d(x).astype(dtype).ravel()[0]
    elif hasattr(x, "astype"):
        return x.astype(dtype).ravel()[0]
    else:
        return jnp.atleast_1d(jnp.array(x, dtype=dtype)).ravel()[0]


def cast_to_python_float(x):
    """Cast the input to a python float. Works on python floats and jax arrays.
    For jax arrays it always takes the first element after a call to .ravel()"""
    if isinstance(x, jax.Array):
        return _cast_to_array_scalar(x, dtype=float).item()
    else:
        try:
            return float(x)
        except TypeError:
            return x
        except ValueError as e:
            # we let NaNs through
            if " NaN " in str(e):
                return x
            else:
                raise e


def cast_to_python_int(x):
    """Cast the input to a python int. Works on python ints and jax arrays.
    For jax arrays it always takes the first element after a call to .ravel()"""
    if isinstance(x, jax.Array):
        return _cast_to_array_scalar(x, dtype=int).item()
    else:
        try:
            return int(x)
        except TypeError:
            return x
        except ValueError as e:
            # we let NaNs through
            if " NaN " in str(e):
                return x
            else:
                raise e


def cast_to_float(x):
    """Cast the input to a float. Works on python floats and jax arrays."""
    try:
        return float(x)
    except Exception:
        try:
            return jnp.asarray(x, dtype=float)
        except Exception:
            return x


def cast_to_int(x):
    """Cast the input to an int. Works on python floats/ints and jax arrays."""
    try:
        return int(x)
    except Exception:
        try:
            if not jnp.any(jnp.isnan(x)):
                return jnp.asarray(x, dtype=int)
            else:
                return x
        except Exception:
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
            for vx, vy in zip(sorted(x), sorted(y)):
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
        # this case covers comparing an array scalar to a python scalar or vice versa
        return jnp.array_equal(x, y)
    else:
        return x == y


def _convert_to_numpy_nan(x):
    """Convert input to numpy.nan if it is a NaN, otherwise return it unchanged
    so that we get consistent hashing."""
    try:
        if np.isnan(x):
            return np.nan
        else:
            return x
    except Exception:
        return x


def _recurse_list_to_tuple(x):
    if isinstance(x, list):
        return tuple(_recurse_list_to_tuple(v) for v in x)
    else:
        return _convert_to_numpy_nan(x)


def ensure_hashable(v):
    """Ensure that the input is hashable. If it is a jax array,
    convert it to a possibly nested tuple or python scalar.

    All NaNs are converted to numpy.nan to get consistent hashing.
    """
    if isinstance(v, jax.Array):
        try:
            if len(v.shape) > 0:
                return _recurse_list_to_tuple(v.tolist())
            else:
                return _convert_to_numpy_nan(v.item())
        except Exception:
            return _convert_to_numpy_nan(v)
    else:
        return _convert_to_numpy_nan(v)


@partial(jax.jit, static_argnames=("niter",))
def bisect_for_root(func, low, high, niter=75):
    def _func(i, args):
        func, low, flow, high, fhigh = args
        mid = (low + high) / 2.0
        fmid = func(mid)
        return jax.lax.cond(
            fmid * fhigh < 0,
            lambda func, low, flow, mid, fmid, high, fhigh: (
                func,
                mid,
                fmid,
                high,
                fhigh,
            ),
            lambda func, low, flow, mid, fmid, high, fhigh: (
                func,
                low,
                flow,
                mid,
                fmid,
            ),
            func,
            low,
            flow,
            mid,
            fmid,
            high,
            fhigh,
        )

    flow = func(low)
    fhigh = func(high)
    args = (func, low, flow, high, fhigh)
    return jax.lax.fori_loop(0, niter, _func, args)[-2]
