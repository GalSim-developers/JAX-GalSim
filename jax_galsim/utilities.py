import functools

import galsim as _galsim
import jax
import jax.numpy as jnp

from jax_galsim.core.utils import has_tracers, implements
from jax_galsim.errors import GalSimIncompatibleValuesError, GalSimValueError
from jax_galsim.position import PositionD, PositionI

printoptions = _galsim.utilities.printoptions


@implements(
    _galsim.utilities.lazy_property,
    lax_description=(
        "The LAX version of this decorator uses an `_workspace` attribute "
        "attached to the object so that the cache can easily be discarded "
        "for certain operations. It also will not cache jax.core.Tracer objects "
        "in order to avoid side-effects in jit/grad/vmap transformations "
        "unless `cache_jax_tracers=True` is given."
    ),
)
def lazy_property(func_=None, cache_jax_tracers=False):
    # the extra layer of indirection here allows the decorator to
    # take keyword arguments and also be used without them.
    # see https://stackoverflow.com/a/57268935
    def _decorator(func):
        attname = func.__name__ + "_cached"

        @property
        @functools.wraps(func)
        def wrapper(self):
            if not hasattr(self, "_workspace"):
                self._workspace = {}
            if attname not in self._workspace:
                val = func(self)
                if cache_jax_tracers or (not has_tracers(val)):
                    self._workspace[attname] = val
            else:
                val = self._workspace[attname]
            return val

        return wrapper

    if callable(func_):
        return _decorator(func_)
    elif func_ is None:
        return _decorator
    else:
        raise RuntimeWarning("Positional arguments are not supported for the lazy_property decorator")


@implements(_galsim.utilities.parse_pos_args)
def parse_pos_args(args, kwargs, name1, name2, integer=False, others=[]):
    def canindex(arg):
        try:
            arg[0], arg[1]
        except (TypeError, IndexError):
            return False
        else:
            return True

    other_vals = []
    if len(args) == 0:
        # Then name1,name2 need to be kwargs
        try:
            x = kwargs.pop(name1)
            y = kwargs.pop(name2)
        except KeyError:
            raise TypeError("Expecting kwargs %s, %s.  Got %s" % (name1, name2, kwargs.keys()))
    elif (isinstance(args[0], PositionI) or (not integer and isinstance(args[0], PositionD))) and len(args) <= 1 + len(others):
        x = args[0].x
        y = args[0].y
        for arg in args[1:]:
            other_vals.append(arg)
            others.pop(0)
    elif canindex(args[0]) and len(args) <= 1 + len(others):
        x = args[0][0]
        y = args[0][1]
        for arg in args[1:]:
            other_vals.append(arg)
            others.pop(0)
    elif len(args) == 1:
        if integer:
            raise TypeError("Cannot parse argument %s as a PositionI" % (args[0]))
        else:
            raise TypeError("Cannot parse argument %s as a PositionD" % (args[0]))
    elif len(args) <= 2 + len(others):
        x = args[0]
        y = args[1]
        for arg in args[2:]:
            other_vals.append(arg)
            others.pop(0)
    else:
        raise TypeError("Too many arguments supplied")
    # Read any remaining other kwargs
    if others:
        for name in others:
            val = kwargs.pop(name)
            other_vals.append(val)
    if kwargs:
        raise TypeError("Received unexpected keyword arguments: %s", kwargs)

    if integer:
        pos = PositionI(x, y)
    else:
        pos = PositionD(x, y)
    if other_vals:
        return (pos,) + tuple(other_vals)
    else:
        return pos


@implements(_galsim.utilities.g1g2_to_e1e2)
def g1g2_to_e1e2(g1, g2):
    # Conversion:
    # e = (a^2-b^2) / (a^2+b^2)
    # g = (a-b) / (a+b)
    # b/a = (1-g)/(1+g)
    # e = (1-(b/a)^2) / (1+(b/a)^2)
    gsq = g1 * g1 + g2 * g2
    if gsq == 0.0:
        return 0.0, 0.0
    else:
        g = jnp.sqrt(gsq)
        boa = (1 - g) / (1 + g)
        e = (1 - boa * boa) / (1 + boa * boa)
        e1 = g1 * (e / g)
        e2 = g2 * (e / g)
        return e1, e2


@implements(_galsim.utilities.convert_interpolant)
def convert_interpolant(interpolant):
    from jax_galsim.interpolant import Interpolant

    if isinstance(interpolant, Interpolant):
        return interpolant
    else:
        # Will raise an appropriate exception if this is invalid.
        return Interpolant.from_name(interpolant)


@implements(_galsim.utilities.unweighted_moments)
def unweighted_moments(image, origin=None):
    from jax_galsim.position import PositionD

    if origin is None:
        origin = PositionD(0, 0)
    a = image.array.astype(float)
    offset = image.origin - origin
    xgrid, ygrid = jnp.meshgrid(
        jnp.arange(image.array.shape[1]) + offset.x,
        jnp.arange(image.array.shape[0]) + offset.y,
    )
    M0 = jnp.sum(a)
    Mx = jnp.sum(xgrid * a) / M0
    My = jnp.sum(ygrid * a) / M0
    Mxx = jnp.sum(((xgrid - Mx) ** 2) * a) / M0
    Myy = jnp.sum(((ygrid - My) ** 2) * a) / M0
    Mxy = jnp.sum((xgrid - Mx) * (ygrid - My) * a) / M0
    return dict(M0=M0, Mx=Mx, My=My, Mxx=Mxx, Myy=Myy, Mxy=Mxy)


@implements(_galsim.utilities.unweighted_shape)
def unweighted_shape(arg):
    from jax_galsim.image import Image

    if isinstance(arg, Image):
        arg = unweighted_moments(arg)
    rsqr = arg["Mxx"] + arg["Myy"]
    return dict(rsqr=rsqr, e1=(arg["Mxx"] - arg["Myy"]) / rsqr, e2=2 * arg["Mxy"] / rsqr)


@implements(_galsim.utilities.horner)
def horner(x, coef, dtype=None):
    x = jnp.array(x)
    coef = jnp.atleast_1d(coef)
    res = jnp.zeros_like(x, dtype=dtype)

    if len(coef.shape) != 1:
        raise GalSimValueError("coef must be 1-dimensional", coef)

    # TODO: we cannot trim zeros in jax galsim because then for all zeros
    # jax attempts to index an array with shape = (0,) which throws an
    # error.
    # coef = jnp.trim_zeros(coef, trim="b")  # trim only from the back

    return jax.lax.cond(
        coef.shape[0] == 0,
        lambda x, coef, res: res,
        lambda x, coef, res: jax.lax.fori_loop(
            0,
            coef.shape[0],
            lambda i, args: (args[0] * x + args[1][i], args[1]),
            (res, coef[::-1]),
        )[0],
        x,
        coef,
        res,
    )


@implements(_galsim.utilities.horner2d)
def horner2d(x, y, coefs, dtype=None, triangle=False):
    x = jnp.array(x)
    y = jnp.array(y)
    coefs = jnp.atleast_1d(coefs)
    res = jnp.zeros_like(x, dtype=dtype)

    if x.shape != y.shape:
        raise GalSimIncompatibleValuesError("x and y are not the same shape", x=x, y=y)

    if len(coefs.shape) != 2:
        raise GalSimValueError("coefs must be 2-dimensional", coefs)

    if triangle and coefs.shape[0] != coefs.shape[1]:
        raise GalSimIncompatibleValuesError("coefs must be square if triangle is True", coefs=coefs, triangle=triangle)

    if triangle:
        # this loop in python looks like
        # Note: for each power of x, it computes all powers of y
        #
        # result = np.zeros_like(x)
        # temp = np.zeros_like(x)
        #
        # for i, coef in enumerate(coefs[::-1]):
        #     result *= x
        #     _horner(y, coef[:i+1], temp)
        #     result += temp

        def _body(i, args):
            res, coeffs, y = args
            # only grab the triangular part
            tri_coeffs = jnp.where(
                jnp.arange(coeffs.shape[1]) < i + 1,
                coeffs[i, :],
                jnp.zeros_like(coeffs[i, :]),
            )
            res = res * x + horner(y, tri_coeffs, dtype=dtype)
            return res, coeffs, y

        res = jax.lax.fori_loop(
            0,
            coefs.shape[0],
            _body,
            (res, coefs[::-1, :], y),
        )[0]
    else:
        # this loop in python looks like
        # Note: for each power of x, it computes all powers of y
        #
        # result = np.zeros_like(x)
        # temp = np.zeros_like(x)
        #
        # for coef in coefs[::-1]:
        #     result *= x
        #     _horner(y, coef, temp)
        #     result += temp

        def _body(i, args):
            res, coeffs, y = args
            res = res * x + horner(y, coeffs[i], dtype=dtype)
            return res, coeffs, y

        res = jax.lax.fori_loop(
            0,
            coefs.shape[0],
            _body,
            (res, coefs[::-1, :], y),
        )[0]

    return res
