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
        jnp.sign(x) * (jnp.pi / 2)
        - _f_pade(x, x2) * jnp.cos(x)
        - _g_pade(x, x2) * jnp.sin(x),
        _si_small_pade(x, x2),
    )


@implements(_galsim.bessel.kv)
@jax.jit
def kv(nu, x):
    nu = 1.0 * nu
    x = 1.0 * x
    return _tfp_bessel_kve(nu, x) / jnp.exp(jnp.abs(x))


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
