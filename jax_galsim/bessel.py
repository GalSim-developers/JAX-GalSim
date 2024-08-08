import galsim as _galsim
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.math import bessel_kve as _tfp_bessel_kve

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
        jnp.sign(x) * (jnp.pi / 2) - _f_pade(x, x2) * jnp.cos(x) - _g_pade(x, x2) * jnp.sin(x),
        _si_small_pade(x, x2),
    )


@jax.jit
def kv(nu, x):
    """Modified Bessel 2nd kind"""
    nu = 1.0 * nu
    x = 1.0 * x
    return _tfp_bessel_kve(nu, x) / jnp.exp(jnp.abs(x))
