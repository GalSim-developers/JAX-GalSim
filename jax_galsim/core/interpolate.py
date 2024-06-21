import jax
import jax.numpy as jnp
import numpy as np


def akima_interp_coeffs_nojax(x, y):
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

    smid = np.where(
        np.abs(denom) >= 1e-12,
        numer / denom,
        (mi[1:-2] + mi[2:-1]) / 2.0,
    )
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
def akima_interp_coeffs(x, y):
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


@jax.jit
def akima_interp(x, xp, yp, coeffs):
    xp = jnp.asarray(xp)
    # yp = jnp.array(yp)  # unused
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
