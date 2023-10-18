from functools import partial

import jax


def convert_to_float(x):
    if isinstance(x, jax.Array):
        if x.shape == ():
            return x.item()
        else:
            return x[0].astype(float).item()
    else:
        return float(x)


def cast_scalar_to_float(x):
    """Cast the input to a float. Works on python floats and jax arrays."""
    if isinstance(x, jax.Array):
        return x.astype(float)
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
