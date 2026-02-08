import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core, dtypes
from jax._src.interpreters import ad, mlir
from jax._src.lax.lax import (
    _const,
    _float,
    broadcast_in_dim,
    broadcast_shapes,
    convert_element_type,
    standard_naryop,
)
from jax._src.lax.special import evaluate_chebyshev_polynomial

from jax_galsim.core.utils import implements


# the code here for Si, f, g and _si_small_pade is taken from galsim/src/math/Sinc.cpp
@jax.jit
def _f_pade(x, x2):
    # fmt: off
    y = 1. / x2
    f = (
            (1. +  # noqa: W504, E126, E226
             y*(7.44437068161936700618e2 +  # noqa: W504, E126, E226
                y*(1.96396372895146869801e5 +  # noqa: W504, E126, E226
                   y*(2.37750310125431834034e7 +  # noqa: W504, E126, E226
                      y*(1.43073403821274636888e9 +  # noqa: W504, E126, E226
                         y*(4.33736238870432522765e10 +  # noqa: W504, E126, E226
                            y*(6.40533830574022022911e11 +  # noqa: W504, E126, E226
                               y*(4.20968180571076940208e12 +  # noqa: W504, E126, E226
                                  y*(1.00795182980368574617e13 +  # noqa: W504, E126, E226
                                     y*(4.94816688199951963482e12 +  # noqa: W504, E126, E226
                                        y*(-4.94701168645415959931e11)))))))))))  # noqa: W504, E126, E226
            / (x*(1. +  # noqa: W504, E126, E226
                  y*(7.46437068161927678031e2 +  # noqa: W504, E126, E226
                     y*(1.97865247031583951450e5 +  # noqa: W504, E126, E226
                        y*(2.41535670165126845144e7 +  # noqa: W504, E126, E226
                           y*(1.47478952192985464958e9 +  # noqa: W504, E126, E226
                              y*(4.58595115847765779830e10 +  # noqa: W504, E126, E226
                                 y*(7.08501308149515401563e11 +  # noqa: W504, E126, E226
                                    y*(5.06084464593475076774e12 +  # noqa: W504, E126, E226
                                       y*(1.43468549171581016479e13 +  # noqa: W504, E126, E226
                                          y*(1.11535493509914254097e13)))))))))))  # noqa: W504, E126, E226
    )
    # fmt: on
    return f


@jax.jit
def _g_pade(x, x2):
    # fmt: off
    y = 1. / x2
    g = (
            y*(1. +  # noqa: W504, E126, E226
               y*(8.1359520115168615e2 +  # noqa: W504, E126, E226
                  y*(2.35239181626478200e5 +  # noqa: W504, E126, E226
                     y*(3.12557570795778731e7 +  # noqa: W504, E126, E226
                        y*(2.06297595146763354e9 +  # noqa: W504, E126, E226
                           y*(6.83052205423625007e10 +  # noqa: W504, E126, E226
                              y*(1.09049528450362786e12 +  # noqa: W504, E126, E226
                                 y*(7.57664583257834349e12 +  # noqa: W504, E126, E226
                                    y*(1.81004487464664575e13 +  # noqa: W504, E126, E226
                                       y*(6.43291613143049485e12 +  # noqa: W504, E126, E226
                                          y*(-1.36517137670871689e12)))))))))))  # noqa: W504, E126, E226
            / (1. +  # noqa: W504, E126, E226
               y*(8.19595201151451564e2 +  # noqa: W504, E126, E226
                  y*(2.40036752835578777e5 +  # noqa: W504, E126, E226
                     y*(3.26026661647090822e7 +  # noqa: W504, E126, E226
                        y*(2.23355543278099360e9 +  # noqa: W504, E126, E226
                           y*(7.87465017341829930e10 +  # noqa: W504, E126, E226
                              y*(1.39866710696414565e12 +  # noqa: W504, E126, E226
                                 y*(1.17164723371736605e13 +  # noqa: W504, E126, E226
                                    y*(4.01839087307656620e13 +  # noqa: W504, E126, E226
                                       y*(3.99653257887490811e13))))))))))  # noqa: W504, E126, E226
    )
    # fmt: on
    return g


@jax.jit
def _si_small_pade(x, x2):
    # fmt: off
    return (
                x*(1. +  # noqa: W504, E126, E226
                   x2*(-4.54393409816329991e-2 +  # noqa: W504, E126, E226
                       x2*(1.15457225751016682e-3 +  # noqa: W504, E126, E226
                           x2*(-1.41018536821330254e-5 +  # noqa: W504, E126, E226
                               x2*(9.43280809438713025e-8 +  # noqa: W504, E126, E226
                                   x2*(-3.53201978997168357e-10 +  # noqa: W504, E126, E226
                                       x2*(7.08240282274875911e-13 +  # noqa: W504, E126, E226
                                           x2*(-6.05338212010422477e-16))))))))  # noqa: W504, E126, E226
                / (1. +  # noqa: W504, E126, E226
                   x2*(1.01162145739225565e-2 +  # noqa: W504, E126, E226
                       x2*(4.99175116169755106e-5 +  # noqa: W504, E126, E226
                           x2*(1.55654986308745614e-7 +  # noqa: W504, E126, E226
                               x2*(3.28067571055789734e-10 +  # noqa: W504, E126, E226
                                   x2*(4.5049097575386581e-13 +  # noqa: W504, E126, E226
                                       x2*(3.21107051193712168e-16)))))))  # noqa: W504, E126, E226
    )
    # fmt: on


@implements(_galsim.bessel.si)
@jax.jit
def si(x):
    x2 = x * x
    return jnp.where(
        x2 > 16.0,
        jnp.sign(x) * (jnp.pi / 2)
        - _f_pade(x, x2) * jnp.cos(x)
        - _g_pade(x, x2) * jnp.sin(x),
        _si_small_pade(x, x2),
    )


# =====================================================================
# Modified Bessel K_v(x) — JAX primitive implementation
#
# Registered as a JAX primitive (bessel_kv_p) with JVP rules for
# automatic differentiation. Uses fori_loop with fixed iteration counts
# and operates element-wise (no vmap). Broadcasting is handled by
# _up_and_broadcast.
#
# The algorithms (Temme series, Steed's continued fraction, Olver
# uniform asymptotic expansion) are derived from TensorFlow
# Probability's bessel.py:
#   https://github.com/tensorflow/probability
#
# Original copyright and license:
#   Copyright 2020 The TensorFlow Probability Authors.
#   Licensed under the Apache License, Version 2.0.
# =====================================================================


def _up_and_broadcast(doit):
    """Broadcast args and upcast bf16/f16 to f32 before calling doit."""

    def up_and_broadcast(*args):
        broadcasted_shape = broadcast_shapes(*(a.shape for a in args))
        args = [
            broadcast_in_dim(a, broadcasted_shape, list(range(a.ndim))) for a in args
        ]
        a_dtype = args[0].dtype
        needs_upcast = a_dtype == dtypes.bfloat16 or a_dtype == np.float16
        if needs_upcast:
            args = [convert_element_type(a, np.float32) for a in args]
            a_x_type = np.float32
        else:
            a_x_type = a_dtype
        result = doit(*args, dtype=a_x_type)
        if needs_upcast:
            result = convert_element_type(result, a_dtype)
        return result

    return up_and_broadcast


# Olver expansion polynomial coefficients (10 terms, up to 31 coefficients each)
# fmt: off
_ASYMPTOTIC_OLVER_COEFFICIENTS = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     -0.20833333333333334, 0., 0.125, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.3342013888888889, 0.,
     -0.40104166666666669, 0., 0.0703125, 0., 0.0],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., -1.0258125964506173, 0., 1.8464626736111112,
     0., -0.89121093750000002, 0., 0.0732421875, 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 4.6695844234262474, 0., -11.207002616222995, 0.,
     8.78912353515625, 0., -2.3640869140624998, 0., 0.112152099609375,
     0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     -28.212072558200244, 0., 84.636217674600744, 0., -91.818241543240035,
     0., 42.534998745388457, 0., -7.3687943594796312, 0., 0.22710800170898438,
     0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 212.5701300392171, 0.,
     -765.25246814118157, 0., 1059.9904525279999, 0., -699.57962737613275,
     0., 218.19051174421159, 0., -26.491430486951554, 0., 0.57250142097473145,
     0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., -1919.4576623184068, 0.,
     8061.7221817373083, 0., -13586.550006434136, 0., 11655.393336864536,
     0., -5305.6469786134048, 0., 1200.9029132163525, 0.,
     -108.09091978839464, 0., 1.7277275025844574, 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 20204.291330966149, 0., -96980.598388637503, 0.,
     192547.0012325315, 0., -203400.17728041555, 0., 122200.46498301747,
     0., -41192.654968897557, 0., 7109.5143024893641, 0.,
     -493.915304773088, 0., 6.074042001273483, 0., 0., 0., 0., 0.,
     0., 0., 0.],
    [0., 0., 0., -242919.18790055133, 0., 1311763.6146629769, 0.,
     -2998015.9185381061, 0., 3763271.2976564039, 0., -2813563.2265865342, 0.,
     1268365.2733216248, 0., -331645.17248456361, 0., 45218.768981362737, 0.,
     -2499.8304818112092, 0., 24.380529699556064, 0., 0., 0., 0., 0.,
     0., 0., 0., 0.0],
    [3284469.8530720375, 0., -19706819.11843222, 0., 50952602.492664628,
     0., -74105148.211532637, 0., 66344512.274729028, 0., -37567176.660763353,
     0., 13288767.166421819, 0., -2785618.1280864552, 0., 308186.40461266245,
     0., -13886.089753717039, 0., 110.01714026924674, 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.]
]
# fmt: on


def _sqrt1px2(x):
    """Numerically stable sqrt(1 + x^2)."""
    eps = _const(x, jnp.finfo(jnp.float64).eps)
    return jnp.where(
        jnp.abs(x) * jnp.sqrt(eps) <= _const(x, 1.0),
        jnp.exp(_const(x, 0.5) * jnp.log1p(x * x)),
        jnp.abs(x),
    )


def _evaluate_temme_coeffs(v):
    """Chebyshev-based gamma coefficients for Temme's method.

    Returns (coeff1, coeff2, gamma1pv_inv, gamma1mv_inv).
    Uses JAX's evaluate_chebyshev_polynomial for Clenshaw recurrence.
    """
    coeff1_coeffs = [
        -1.142022680371168e0,
        6.5165112670737e-3,
        3.087090173086e-4,
        -3.4706269649e-6,
        6.9437664e-9,
        3.67795e-11,
        -1.356e-13,
    ]
    coeff2_coeffs = [
        1.843740587300905e0,
        -7.68528408447867e-2,
        1.2719271366546e-3,
        -4.9717367042e-6,
        -3.31261198e-8,
        2.423096e-10,
        -1.702e-13,
        -1.49e-15,
    ]
    y = _const(v, 2.0) * (_const(v, 8.0) * v * v - _const(v, 1.0))

    # evaluate_chebyshev_polynomial(x, coeffs) evaluates sum c_k T_k(x/2),
    # so passing y = 2w gives evaluation at w = 8v^2 - 1.
    coeff1 = evaluate_chebyshev_polynomial(y, list(reversed(coeff1_coeffs)))
    coeff2 = evaluate_chebyshev_polynomial(y, list(reversed(coeff2_coeffs)))

    gamma1pv = coeff2 - v * coeff1
    gamma1mv = coeff2 + v * coeff1
    return coeff1, coeff2, gamma1pv, gamma1mv


def _temme_series_kve(v, z):
    """Kve(v, z) and Kve(v+1, z) via Temme power series.

    Assumes |v| < 0.5 and |z| <= 2. Returns exponentially scaled values.
    Uses fori_loop with fixed 15 iterations (empirically, max needed is 12
    for f64 across the valid domain).
    """
    tol = _const(z, jnp.finfo(jnp.float64).eps)

    coeff1, coeff2, gamma1pv_inv, gamma1mv_inv = _evaluate_temme_coeffs(v)

    z_sq = z * z
    logzo2 = jnp.log(z / _const(z, 2.0))
    mu = -v * logzo2
    pi_v = _const(v, jnp.pi) * v
    sinc_v = jnp.where(v == _const(v, 0.0), _const(v, 1.0), jnp.sin(pi_v) / pi_v)
    sinhc_mu = jnp.where(mu == _const(mu, 0.0), _const(mu, 1.0), jnp.sinh(mu) / mu)

    initial_f = (coeff1 * jnp.cosh(mu) + coeff2 * (-logzo2) * sinhc_mu) / sinc_v
    initial_p = _const(v, 0.5) * jnp.exp(mu) / gamma1pv_inv
    initial_q = _const(v, 0.5) * jnp.exp(-mu) / gamma1mv_inv

    max_iters = 15

    def body_fn(i, carry):
        f, p, q, coeff, kv_sum, kvp1_sum, converged = carry
        index = i + _const(v, 1.0)

        new_f = (index * f + p + q) / (index * index - v * v)
        new_p = p / (index - v)
        new_q = q / (index + v)
        h = new_p - index * new_f
        new_coeff = coeff * z_sq / (_const(z, 4.0) * index)
        new_kv_sum = kv_sum + new_coeff * new_f
        new_kvp1_sum = kvp1_sum + new_coeff * h

        new_converged = converged | (
            jnp.abs(new_coeff * new_f) < jnp.abs(new_kv_sum) * tol
        )

        f = jnp.where(converged, f, new_f)
        p = jnp.where(converged, p, new_p)
        q = jnp.where(converged, q, new_q)
        coeff = jnp.where(converged, coeff, new_coeff)
        kv_sum = jnp.where(converged, kv_sum, new_kv_sum)
        kvp1_sum = jnp.where(converged, kvp1_sum, new_kvp1_sum)

        return (f, p, q, coeff, kv_sum, kvp1_sum, new_converged)

    init = (
        initial_f,
        initial_p,
        initial_q,
        jnp.ones_like(z),
        initial_f,
        initial_p,
        jnp.zeros_like(v, dtype=jnp.bool_),
    )
    _, _, _, _, kv_sum, kvp1_sum, _ = jax.lax.fori_loop(0, max_iters, body_fn, init)

    kve = kv_sum * jnp.exp(z)
    kvep1 = _const(z, 2.0) * kvp1_sum * jnp.exp(z) / z
    return kve, kvep1


def _continued_fraction_kve(v, z):
    """Kve(v, z) and Kve(v+1, z) via Steed's continued fraction.

    Assumes |v| < 0.5 and |z| > 2. Returns exponentially scaled values.
    Uses fori_loop with fixed 80 iterations (empirically, max needed is 77
    for f64 at z~2).
    """
    tol = _const(z, jnp.finfo(jnp.float64).eps)

    initial_numerator = v * v - _const(v, 0.25)
    initial_denominator = _const(z, 2.0) * (z + _const(z, 1.0))
    initial_ratio = _const(z, 1.0) / initial_denominator
    initial_seq = -initial_numerator

    max_iters = 80

    def body_fn(i, carry):
        (
            partial_numerator,
            partial_denominator,
            denominator_ratio,
            convergent_difference,
            hypergeometric_ratio,
            k_0,
            k_1,
            c,
            q,
            hypergeometric_sum,
            converged,
        ) = carry
        index = i + _const(z, 2.0)

        new_partial_numerator = partial_numerator - _const(z, 2.0) * (
            index - _const(z, 1.0)
        )
        new_c = -c * new_partial_numerator / index
        next_k = (k_0 - partial_denominator * k_1) / new_partial_numerator
        new_k_0 = k_1
        new_k_1 = next_k
        new_q = q + new_c * next_k
        new_partial_denominator = partial_denominator + _const(z, 2.0)
        new_denominator_ratio = _const(z, 1.0) / (
            new_partial_denominator + new_partial_numerator * denominator_ratio
        )
        new_convergent_difference = convergent_difference * (
            new_partial_denominator * new_denominator_ratio - _const(z, 1.0)
        )
        new_hypergeometric_ratio = hypergeometric_ratio + new_convergent_difference
        new_hypergeometric_sum = hypergeometric_sum + new_q * new_convergent_difference

        new_converged = converged | (
            jnp.abs(new_q * new_convergent_difference)
            < jnp.abs(new_hypergeometric_sum) * tol
        )

        partial_numerator = jnp.where(
            converged, partial_numerator, new_partial_numerator
        )
        c = jnp.where(converged, c, new_c)
        k_0 = jnp.where(converged, k_0, new_k_0)
        k_1 = jnp.where(converged, k_1, new_k_1)
        q = jnp.where(converged, q, new_q)
        partial_denominator = jnp.where(
            converged, partial_denominator, new_partial_denominator
        )
        denominator_ratio = jnp.where(
            converged, denominator_ratio, new_denominator_ratio
        )
        convergent_difference = jnp.where(
            converged, convergent_difference, new_convergent_difference
        )
        hypergeometric_ratio = jnp.where(
            converged, hypergeometric_ratio, new_hypergeometric_ratio
        )
        hypergeometric_sum = jnp.where(
            converged, hypergeometric_sum, new_hypergeometric_sum
        )

        return (
            partial_numerator,
            partial_denominator,
            denominator_ratio,
            convergent_difference,
            hypergeometric_ratio,
            k_0,
            k_1,
            c,
            q,
            hypergeometric_sum,
            new_converged,
        )

    init = (
        initial_numerator,
        initial_denominator,
        initial_ratio,
        initial_ratio,
        initial_ratio,
        jnp.zeros_like(z),
        jnp.ones_like(z),
        initial_seq,
        initial_seq,
        jnp.ones_like(z) - initial_numerator * initial_ratio,
        jnp.zeros_like(v, dtype=jnp.bool_),
    )
    result = jax.lax.fori_loop(0, max_iters, body_fn, init)
    hypergeometric_ratio = result[4]
    hypergeometric_sum = result[9]

    log_kve = _const(z, 0.5) * jnp.log(
        _const(z, jnp.pi) / (_const(z, 2.0) * z)
    ) - jnp.log(hypergeometric_sum)
    log_kvp1e = (
        log_kve
        + jnp.log1p(_const(z, 2.0) * (v + z + initial_numerator * hypergeometric_ratio))
        - jnp.log(z)
        - jnp.log(_const(z, 2.0))
    )
    return jnp.exp(log_kve), jnp.exp(log_kvp1e)


def _olver_kve(v, z):
    """Kve(v, z) using Olver's uniform asymptotic expansion.

    Valid for |v| >= 50. Returns Kv(v,z)*exp(z).
    """
    v_abs = jnp.abs(v)
    w = z / v_abs
    t = _const(z, 1.0) / _sqrt1px2(w)

    divisor = v_abs
    kve_sum = _const(z, 1.0)

    for i in range(len(_ASYMPTOTIC_OLVER_COEFFICIENTS)):
        coeff = _const(z, 0.0)
        for c in _ASYMPTOTIC_OLVER_COEFFICIENTS[i]:
            coeff = coeff * t + _const(z, c)
        term = coeff / divisor
        kve_sum = kve_sum + (term if i % 2 == 1 else -term)
        divisor = divisor * v_abs

    shared_prefactor = (
        _const(z, 1.0) / (_sqrt1px2(w) + w) + jnp.log(w) - jnp.log1p(_const(z, 1.0) / t)
    )
    log_k_prefactor = (
        _const(z, 0.5) * jnp.log(_const(z, jnp.pi) * t / (_const(z, 2.0) * v_abs))
        - v_abs * shared_prefactor
    )
    log_kve = log_k_prefactor + jnp.log(kve_sum)
    return jnp.exp(log_kve)


def _temme_kve(v, x):
    """Kve(v, x) using Temme's method for |v| < 50.

    Reduces to fractional order |u| <= 0.5, computes via series or CF,
    then forward recurrence to reach order v.
    Uses fori_loop with fixed 50 iterations for forward recurrence.
    """
    v = jnp.abs(v)
    n = jnp.round(v)
    u = v - n

    small_x = jnp.where(x <= _const(x, 2.0), x, _const(x, 0.1))
    large_x = jnp.where(x > _const(x, 2.0), x, _const(x, 1000.0))

    temme_kue, temme_kuep1 = _temme_series_kve(u, small_x)
    cf_kue, cf_kuep1 = _continued_fraction_kve(u, large_x)

    kue = jnp.where(x <= _const(x, 2.0), temme_kue, cf_kue)
    kuep1 = jnp.where(x <= _const(x, 2.0), temme_kuep1, cf_kuep1)

    max_recurrence = 50

    def recurrence_body(i, carry):
        kve, kvep1 = carry
        index = i + _const(x, 1.0)
        past_n = index > n
        next_kvep1 = _const(x, 2.0) * (u + index) * kvep1 / x + kve
        new_kve = jnp.where(past_n, kve, kvep1)
        new_kvep1 = jnp.where(past_n, kvep1, next_kvep1)
        return (new_kve, new_kvep1)

    kve, _ = jax.lax.fori_loop(0, max_recurrence, recurrence_body, (kue, kuep1))
    return kve


def _kve_core(nu, x):
    """Core dispatcher: computes Kve(nu, x) = Kv(nu, x) * exp(x).

    Branchless: computes both Olver and Temme, selects based on |nu| >= 50.
    """
    nu = jnp.abs(nu)

    small_nu = jnp.where(nu < _const(nu, 50.0), nu, _const(nu, 0.1))
    large_nu = jnp.where(nu >= _const(nu, 50.0), nu, _const(nu, 1000.0))

    olver_result = _olver_kve(large_nu, x)
    temme_result = _temme_kve(small_nu, x)

    return jnp.where(nu >= _const(nu, 50.0), olver_result, temme_result)


def _bessel_kv_impl(v, x, *, dtype):
    """Element-wise implementation of K_v(x).

    The dtype kwarg is required by the _up_and_broadcast pattern.
    """
    v = jnp.abs(v)  # K_{-v} = K_v

    safe_x = jnp.where(x > _const(x, 0.0), x, _const(x, 1.0))
    kve = _kve_core(v, safe_x)
    result = kve * jnp.exp(-safe_x)

    result = jnp.where(x == _const(x, 0.0), _const(x, jnp.inf), result)
    result = jnp.where(x < _const(x, 0.0), _const(x, jnp.nan), result)
    return result


# --- JAX primitive registration ---

_bessel_kv_p = standard_naryop([_float, _float], "bessel_kv")

mlir.register_lowering(
    _bessel_kv_p,
    mlir.lower_fun(_up_and_broadcast(_bessel_kv_impl), multiple_results=False),
)


def _bessel_kv_jvp_v(g, v, x):
    return jnp.zeros_like(v) * g


def _bessel_kv_jvp_x(g, v, x):
    return g * _const(x, -0.5) * (kv(v - _const(v, 1.0), x) + kv(v + _const(v, 1.0), x))


ad.defjvp(_bessel_kv_p, _bessel_kv_jvp_v, _bessel_kv_jvp_x)


@implements(_galsim.bessel.kv)
def kv(nu, x):
    """Modified Bessel function of the second kind K_v(x).

    Registered as a JAX primitive with JVP rules.
    Supports jit, vmap, and grad (w.r.t. x).
    Gradient w.r.t. v is not supported (returns zero).
    """
    nu = jnp.asarray(nu, dtype=float)
    x = jnp.asarray(x, dtype=float)
    nu, x = core.standard_insert_pvary(nu, x)
    return _bessel_kv_p.bind(nu, x)


@jax.jit
def _R(z, num, denom):
    return jnp.polyval(num, z) / jnp.polyval(denom, z)


@jax.jit
def _evaluate_rational(z, num, denom):
    return _R(z, num[::-1], denom[::-1])


# jitted & vectorized version
_v_rational = jax.jit(jax.vmap(_evaluate_rational, in_axes=(0, None, None)))


@implements(
    _galsim.bessel.j0,
    lax_description="""\
The JAX-GalSim implementation of ``j0`` is a vectorized version of the Boost C++
algorith for the Bessel function of the first kind J0(x).""",
)
@jax.jit
def j0(x):
    orig_shape = x.shape

    x = jnp.atleast_1d(x)

    P1 = jnp.array(
        [
            -4.1298668500990866786e11,
            2.7282507878605942706e10,
            -6.2140700423540120665e08,
            6.6302997904833794242e06,
            -3.6629814655107086448e04,
            1.0344222815443188943e02,
            -1.2117036164593528341e-01,
        ]
    )
    Q1 = jnp.array(
        [
            2.3883787996332290397e12,
            2.6328198300859648632e10,
            1.3985097372263433271e08,
            4.5612696224219938200e05,
            9.3614022392337710626e02,
            1.0,
            0.0,
        ]
    )

    P2 = jnp.array(
        [
            -1.8319397969392084011e03,
            -1.2254078161378989535e04,
            -7.2879702464464618998e03,
            1.0341910641583726701e04,
            1.1725046279757103576e04,
            4.4176707025325087628e03,
            7.4321196680624245801e02,
            4.8591703355916499363e01,
        ]
    )
    Q2 = jnp.array(
        [
            -3.5783478026152301072e05,
            2.4599102262586308984e05,
            -8.4055062591169562211e04,
            1.8680990008359188352e04,
            -2.9458766545509337327e03,
            3.3307310774649071172e02,
            -2.5258076240801555057e01,
            1.0,
        ]
    )

    PC = jnp.array(
        [
            2.2779090197304684302e04,
            4.1345386639580765797e04,
            2.1170523380864944322e04,
            3.4806486443249270347e03,
            1.5376201909008354296e02,
            8.8961548424210455236e-01,
        ]
    )
    QC = jnp.array(
        [
            2.2779090197304684318e04,
            4.1370412495510416640e04,
            2.1215350561880115730e04,
            3.5028735138235608207e03,
            1.5711159858080893649e02,
            1.0,
        ]
    )

    PS = jnp.array(
        [
            -8.9226600200800094098e01,
            -1.8591953644342993800e02,
            -1.1183429920482737611e02,
            -2.2300261666214198472e01,
            -1.2441026745835638459e00,
            -8.8033303048680751817e-03,
        ]
    )
    QS = jnp.array(
        [
            5.7105024128512061905e03,
            1.1951131543434613647e04,
            7.2642780169211018836e03,
            1.4887231232283756582e03,
            9.0593769594993125859e01,
            1.0,
        ]
    )

    x1 = 2.4048255576957727686e00
    x2 = 5.5200781102863106496e00
    x11 = 6.160e02
    x12 = -1.42444230422723137837e-03
    x21 = 1.4130e03
    x22 = 5.46860286310649596604e-04
    one_div_root_pi = 5.641895835477562869480794515607725858e-01

    def t1(x):  # x<=4
        y = x * x
        r = _v_rational(y, P1, Q1)
        factor = (x + x1) * ((x - x11 / 256) - x12)
        return factor * r

    def t2(x):  # x<=8
        y = 1 - (x * x) / 64
        r = _v_rational(y, P2, Q2)
        factor = (x + x2) * ((x - x21 / 256) - x22)
        return factor * r

    def t3(x):  # x>8
        y = 8 / x
        y2 = y * y
        rc = _v_rational(y2, PC, QC)
        rs = _v_rational(y2, PS, QS)
        factor = one_div_root_pi / jnp.sqrt(x)
        sx = jnp.sin(x)
        cx = jnp.cos(x)
        return factor * (rc * (cx + sx) - y * rs * (sx - cx))

    x = jnp.abs(x)
    return jnp.select(
        [x == 0, x <= 4, x <= 8, x > 8], [1, t1(x), t2(x), t3(x)], default=x
    ).reshape(orig_shape)
