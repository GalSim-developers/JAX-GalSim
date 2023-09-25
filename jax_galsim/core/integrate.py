from functools import partial
from typing import NamedTuple, Tuple

import jax.numpy as jnp
from jax import Array, jit


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
    w = w.real

    if n % 2 == 1:
        weights = jnp.concatenate([w, w[::-1]])
    else:
        weights = jnp.concatenate([w, w[len(w) - 2 :: -1]])

    return points, weights


class ClenshawCurtisQuad(NamedTuple):
    order: int
    absc: Array
    absw: Array
    errw: Array

    @classmethod
    def init(cls, order: int):
        order = 2 * order + 1
        absc, absw, errw = cls.compute_weights(order)
        absc, absw = cls.rescale_weights(absc, absw)
        return cls(order=order, absc=absc, absw=absw, errw=errw)

    @staticmethod
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
    return S.squeeze()
