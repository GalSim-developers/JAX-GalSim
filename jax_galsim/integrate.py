import jax
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from functools import partial
from typing import NamedTuple, Tuple

from jax import Array, jit, lax, vmap


# move pure function out of the class
def abs_weights(n: int):
    assert n > 1
    points = -jnp.cos(jnp.linspace(0, jnp.pi, n))

    if n == 2:
        weights = jnp.array([1.0, 1.0])
        return points, weights

    n -= 1
    N = jnp.arange(1, n, 2)
    length = len(N)
    m = n - length
    v0 = jnp.concatenate([2.0 / N / (N - 2), jnp.array([1.0 / N[-1]]), jnp.zeros(m)])
    v2 = -v0[:-1] - v0[:0:-1]
    g0 = -jnp.ones(n)
    g0 = g0.at[length].add(n)
    g0 = g0.at[m].add(n)
    g = g0 / (n**2 - 1 + (n % 2))

    w = jnp.fft.ihfft(v2 + g)
    # assert max(w.imag) < 1.0e-15
    w = w.real

    if n % 2 == 1:
        weights = jnp.concatenate([w, w[::-1]])
    else:
        weights = jnp.concatenate([w, w[len(w) - 2 :: -1]])

    return points, weights


class ClenshawCurtisQuad(NamedTuple):  # NamedTuple is already a pytree
    order: int
    absc: Array
    absw: Array
    errw: Array

    @classmethod  # alternative constructor, doesn't get in the way of flatten/unflatten
    def init(cls, order: int):
        order = 2 * order + 1
        absc, absw, errw = cls.compute_weights(order)
        absc, absw = cls.rescale_weights(absc, absw)
        return cls(order=order, absc=absc, absw=absw, errw=errw)

    @staticmethod  # staticmethod to ensure pureity
    def compute_weights(order: int):
        x, wx = abs_weights(order)
        nsub = (order + 1) // 2
        _, wsub = abs_weights(nsub)
        errw = wx.at[::2].add(-wsub)
        return x, wx, errw

    @staticmethod
    def rescale_weights(
        absc: Array,
        absw: Array,
        *,
        interval_in: Tuple[float, float] = (-1, 1),
        interval_out: Tuple[float, float] = (0, 1),
    ):
        (in_min, in_max), (out_min, out_max) = interval_in, interval_out
        delta_in, delta_out = in_max - in_min, out_max - out_min
        absc = ((absc - in_min) * out_max - (absc - in_max) * out_min) / delta_in
        absw = delta_out / delta_in * absw
        return absc, absw


@partial(jit, static_argnums=(0,))
def quad_integral(f, a, b, quad: ClenshawCurtisQuad):
    a = jnp.atleast_1d(a)
    b = jnp.atleast_1d(b)
    d = b - a
    xi = a[jnp.newaxis, :] + jnp.einsum("i...,k...->ik...", quad.absc, d)
    xi = xi.squeeze()
    fi = f(xi)
    S = d * jnp.einsum("i...,i...", quad.absw, fi)
    return S.squeeze()  # d * jnp.einsum('i...,i...', quad.errw, fi)


# Not yet expose but here in case one needs it (then add it in the __init__)
def incremental_int(fn, t, order=10):
    """Incremetal intergration using Clenshaw Curtis Quadrature

    Example::
        >>> f = lambda x: jnp.sqrt(x) * jnp.exp(-x) * jnp.sin(100.*x)
        >>> tinc = jnp.linspace(0.,1.,4096)
        >>> incremental_int(f5,tinc,150)

    It uses CC quad to compute int_t[i]^t[i+1] fn(x) dx

    Inputs:
        fn : function (vectorized of 1 variable)
        t : array
        order : order of the Clenshaw Curtis Quadrature
    Outputs:
        array of int_t[0]^t[i] fn(x) dx  i=0, len(t)-1

    """
    quad = ClenshawCurtisQuad.init(order)

    def integ(carry, t):
        y, t_prev = carry
        y = y + quadIntegral(fn, t_prev, t, quad)
        return (y, t), y

    (yf, _), y = jax.lax.scan(integ, (0.0, jnp.array(t[0])), t)
    return y
