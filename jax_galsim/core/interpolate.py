import functools

import jax
import jax.numpy as jnp
import numpy as np


def akima_interp_coeffs(x, y, use_jax=True):
    """Compute the interpolation coefficients for an Akima cubic spline.

    An Akima cubic spline is a piecewise C(1) cubic polynomial that interpolates a set of
    points (x, y). Unlike a more traditional cubic spline, the Akima spline can be computed
    without solving a linear system of equations. However, the Akima spline does not have
    continuous second derivatives at the interpolation points.

    See https://en.wikipedia.org/wiki/Akima_spline and
    Akima (1970), "A new method of interpolation and smooth curve fitting based on local procedures",
    Journal of the ACM. 17: 589-602 for a description of the technique.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the data points. These must be sorted into increasing order
        and cannot contain any duplicates.
    y : array-like
        The y-coordinates of the data points.
    use_jax : bool, optional
        Whether to use JAX for computation. Default is True. If False, the
        coefficients are computed using NumPy on the host device. This can be
        useful when embded inside JAX code w/ JIT applied to pre-compute the
        coefficients.

    Returns
    -------
    tuple
        A tuple of arrays (a, b, c, d) where each array has shape (N-1,) and
        contains the coefficients for the cubic polynomial that interpolates
        the data points between x[i] and x[i+1].
    """
    if use_jax:
        return _akima_interp_coeffs_jax(x, y)
    else:
        return _akima_interp_coeffs_nojax(x, y)


def _akima_interp_coeffs_nojax(x, y):
    dx = x[1:] - x[:-1]
    mi = (y[1:] - y[:-1]) / dx

    # these values are imposed for points
    # at the ends
    s0 = mi[0:1]
    s1 = (mi[0:1] + mi[1:2]) / 2.0
    snm2 = (mi[-3:-2] + mi[-2:-1]) / 2.0
    snm1 = mi[-2:-1]

    wim1 = np.abs(mi[3:] - mi[2:-1])
    wi = np.abs(mi[1:-2] - mi[0:-3])
    denom = wim1 + wi
    numer = wim1 * mi[1:-2] + wi * mi[2:-1]

    msk_denom = np.abs(denom) >= 1e-12
    smid = np.zeros_like(denom)
    smid[msk_denom] = numer[msk_denom] / denom[msk_denom]
    smid[~msk_denom] = (mi[1:-2][~msk_denom] + mi[2:-1][~msk_denom]) / 2.0
    s = np.concatenate([s0, s1, smid, snm2, snm1])

    # these coeffs are for
    # P(x) = a + b * (x-xi) + c * (x-xi)**2 + d * (x-xi)**3
    # for a point x that falls in [xi, xip1]
    a = y[:-1]
    b = s[:-1]
    c = (3 * mi - 2 * s[:-1] - s[1:]) / dx
    d = (s[:-1] + s[1:] - 2 * mi) / dx / dx

    return (a, b, c, d)


@jax.jit
def _akima_interp_coeffs_jax(x, y):
    dx = x[1:] - x[:-1]
    mi = (y[1:] - y[:-1]) / dx

    # these values are imposed for points
    # at the ends
    s0 = mi[0:1]
    s1 = (mi[0:1] + mi[1:2]) / 2.0
    snm2 = (mi[-3:-2] + mi[-2:-1]) / 2.0
    snm1 = mi[-2:-1]

    wim1 = jnp.abs(mi[3:] - mi[2:-1])
    wi = jnp.abs(mi[1:-2] - mi[0:-3])
    denom = wim1 + wi
    numer = wim1 * mi[1:-2] + wi * mi[2:-1]

    smid = jnp.where(
        jnp.abs(denom) >= 1e-12,
        numer / denom,
        (mi[1:-2] + mi[2:-1]) / 2.0,
    )
    s = jnp.concatenate([s0, s1, smid, snm2, snm1])

    # these coeffs are for
    # P(x) = a + b * (x-xi) + c * (x-xi)**2 + d * (x-xi)**3
    # for a point x that falls in [xi, xip1]
    a = y[:-1]
    b = s[:-1]
    c = (3 * mi - 2 * s[:-1] - s[1:]) / dx
    d = (s[:-1] + s[1:] - 2 * mi) / dx / dx

    return (a, b, c, d)


@functools.partial(jax.jit, static_argnames=("fixed_spacing",))
def akima_interp(x, xp, yp, coeffs, fixed_spacing=False):
    """Conmpute the values of an Akima cubic spline at a set of points given the
    interpolation coefficients.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the points where the interpolation is computed.
    xp : array-like
        The x-coordinates of the data points. These must be sorted into increasing order
        and cannot contain any duplicates.
    yp : array-like
        The y-coordinates of the data points. Not used currently.
    coeffs : tuple
        The interpolation coefficients returned by `akima_interp_coeffs`.
    fixed_spacing : bool, optional
        Whether the data points are evenly spaced. Default is False. If True, the
        code uses a faster technique to compute the index of the data points x into
        the array xp such that xp[i] <= x < xp[i+1].

    Returns
    -------
    array-like
        The values of the Akima cubic spline at the points x.
    """
    xp = jnp.asarray(xp)
    # yp = jnp.array(yp)  # unused
    if fixed_spacing:
        dxp = xp[1] - xp[0]
        i = jnp.floor((x - xp[0]) / dxp).astype(jnp.int32)
        i = jnp.clip(i, 0, len(xp) - 2)
    else:
        i = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1) - 1

    # these coeffs are for
    # P(x) = a + b * (x-xi) + c * (x-xi)**2 + d * (x-xi)**3
    # for a point x that falls in [xi, xip1]
    a, b, c, d = coeffs
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    c = jnp.asarray(c)
    d = jnp.asarray(d)

    dx = x - xp[i]
    dx2 = dx * dx
    dx3 = dx2 * dx
    xval = a[i] + b[i] * dx + c[i] * dx2 + d[i] * dx3

    xval = jnp.where(x < xp[0], 0, xval)
    xval = jnp.where(x > xp[-1], 0, xval)
    return xval
