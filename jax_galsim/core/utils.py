import re
import textwrap
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten


def has_tracers(x):
    """Return True if the input item is a JAX tracer or object, False otherwise."""
    for item in tree_flatten(x)[0]:
        if isinstance(item, jax.core.Tracer) or type(item) is object:
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
            # this will return the same value for anything float-like that
            # cannot be cast to float
            # however, it will raise an error if something is not float-like
            return 1.0 * x
        except ValueError as e:
            # we let NaNs through
            if " NaN " in str(e):
                # this will return the same value for anything float-like that
                # cannot be cast to float
                # however, it will raise an error if something is not float-like
                return 1.0 * x
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
            # this will return the same value for anything int-like that
            # cannot be cast to int
            # however, it will raise an error if something is not int-like
            return 1 * x
        except ValueError as e:
            # we let NaNs through
            if " NaN " in str(e):
                # this will return the same value for anything int-like that
                # cannot be cast to int
                # however, it will raise an error if something is not int-like
                return 1 * x
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
            # this will return the same value for anything float-like that
            # cannot be cast to float
            # however, it will raise an error if something is not float-like
            # we exclude object types since they are used in JAX tracing
            if type(x) is object:
                return x
            else:
                return 1.0 * x


def cast_to_int(x):
    """Cast the input to an int. Works on python floats/ints and jax arrays."""
    try:
        return int(x)
    except Exception:
        try:
            if not jnp.any(jnp.isnan(x)):
                return jnp.asarray(x, dtype=int)
            else:
                # this will return the same value for anything int-like that
                # cannot be cast to int
                # however, it will raise an error if something is not int-like
                if type(x) is object:
                    return x
                else:
                    return 1 * x
        except Exception:
            # this will return the same value for anything int-like that
            # cannot be cast to int
            # however, it will raise an error if something is not int-like
            if type(x) is object:
                return x
            else:
                return 1 * x


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


# start of code from https://github.com/google/jax/blob/main/jax/_src/numpy/util.py #
# used with modifications for galsim under the following license:
# fmt: off
#
#    Copyright 2020 The JAX Authors.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# fmt: on

_galsim_signature_re = re.compile(r"^([\w., ]+=)?\s*[\w\.]+\([\w\W]*?\)$", re.MULTILINE)
_docreference = re.compile(r":doc:`(.*?)\s*<.*?>`")


class ParsedDoc(NamedTuple):
    """
    docstr: full docstring
    signature: signature from docstring.
    summary: summary from docstring.
    front_matter: front matter before sections.
    sections: dictionary of section titles to section content.
    """

    docstr: str = ""
    signature: str = ""
    summary: str = ""
    front_matter: str = ""
    sections: dict[str, str] = {}


def _break_off_body_section_by_newline(body):
    first_lines = []
    body_lines = []
    found_first_break = False
    for line in body.split("\n"):
        if not first_lines:
            first_lines.append(line)
            continue

        if not line.strip() and not found_first_break:
            found_first_break = True
            continue

        if found_first_break:
            body_lines.append(line)
        else:
            first_lines.append(line)

    firstline = "\n".join(first_lines)
    body = "\n".join(body_lines)
    body = textwrap.dedent(body.lstrip("\n"))

    return firstline, body


def _parse_galsimdoc(docstr):
    """Parse a standard galsim-style docstring.

    Args:
        docstr: the raw docstring from a function
    Returns:
        ParsedDoc: parsed version of the docstring
    """
    if docstr is None or not docstr.strip():
        return ParsedDoc(docstr)

    # Remove any :doc: directives in the docstring to avoid sphinx errors
    docstr = _docreference.sub(lambda match: f"{match.groups()[0]}", docstr)

    signature, body = "", docstr
    match = _galsim_signature_re.match(body)
    if match:
        signature = match.group()
        body = docstr[match.end() :]

    firstline, body = _break_off_body_section_by_newline(body)

    match = _galsim_signature_re.match(body)
    if match:
        signature = match.group()
        body = body[match.end() :]

    summary = firstline
    if not summary:
        summary, body = _break_off_body_section_by_newline(body)

    front_matter_lines = []
    body_lines = []
    found_params = False
    for line in body.split("\n"):
        if not found_params and line.lstrip().startswith("Parameters:"):
            found_params = True

        if found_params:
            body_lines.append(line)
        else:
            front_matter_lines.append(line)
    front_matter = "\n".join(front_matter_lines)
    body = "\n".join(body_lines)

    # we add back the body for now, but keep code above if we parse params in the future
    front_matter = front_matter + "\n" + body

    return ParsedDoc(
        docstr=docstr,
        signature=signature,
        summary=summary,
        front_matter=front_matter,
        sections={},
    )


def implements(
    original_fun,
    lax_description="",
    module=None,
):
    """Decorator for JAX functions which implement a specified GalSim function.

    This mainly contains logic to copy and modify the docstring of the original
    function. In particular, if `update_doc` is True, parameters listed in the
    original function that are not supported by the decorated function will
    be removed from the docstring. For this reason, it is important that parameter
    names match those in the original GalSim function.

    Parameters:
        original_fun:     The original function being implemented
        lax_description:  A string description that will be added to the beginning of
                          the docstring.
        module:           An optional string specifying the module from which the original function
                          is imported. This is useful for objects, where the module cannot
                          be determined from the original function itself.
    """

    def decorator(wrapped_fun):
        wrapped_fun.__galsim_wrapped__ = original_fun

        # Allows this pattern: @implements(getattr(np, 'new_function', None))
        if original_fun is None:
            if lax_description:
                wrapped_fun.__doc__ = lax_description
            return wrapped_fun

        docstr = getattr(original_fun, "__doc__", None)
        name = getattr(
            original_fun, "__name__", getattr(wrapped_fun, "__name__", str(wrapped_fun))
        )
        try:
            mod = module or original_fun.__module__
        except AttributeError:
            pass
        else:
            name = f"{mod}.{name}"

        if docstr:
            try:
                parsed = _parse_galsimdoc(docstr)

                docstr = parsed.summary.strip() + "\n" if parsed.summary else ""
                docstr += f"\nLAX-backend implementation of :func:`{name}`.\n"
                if lax_description:
                    docstr += "\n" + lax_description.strip() + "\n"
                docstr += "\n*Original docstring below.*\n"

                if parsed.front_matter:
                    docstr += "\n" + parsed.front_matter.strip() + "\n"
            except Exception:
                docstr = original_fun.__doc__

        wrapped_fun.__doc__ = docstr
        for attr in ["__name__", "__qualname__"]:
            try:
                value = getattr(original_fun, attr)
            except AttributeError:
                pass
            else:
                setattr(wrapped_fun, attr, value)
        return wrapped_fun

    return decorator


# end of code from https://github.com/google/jax/blob/main/jax/_src/numpy/util.py #
