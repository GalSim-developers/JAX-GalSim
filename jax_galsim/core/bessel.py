import jax
import jax.numpy as jnp


@jax.jit
def R(z, num,denom):
  return jnp.polyval(num,z)/jnp.polyval(denom,z)
@jax.jit
def _evaluate_rational(z, num, denom):
  return R(z,num[::-1],denom[::-1])

# jitted & vectorized version
v_rational = jax.jit(jax.vmap(_evaluate_rational, in_axes=(0, None, None)))


@jax.jit
def j0(x):
    """Bessel function of the first kind J0(x) (similar to scipy.special.j0)
    code from Boost C++ implementation
    boost/math/special_functions/detail/bessel_j0.hpp

    Examples::

        >>> x = jnp.linspace(0,300,10_000)
        >>> plt.plot(x,j0(x))
        >>> plt.plot(x,jax.vmap(jax.jacfwd(j0))(x))

    Inputs:
        x: scalar/array of real(s)

    Outputs:
        j0(x) with same shape as x

    """
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
    ##    assert len(P1) == len(Q1)

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
    ##      assert len(P2) == len(Q2)

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

    ##      assert len(PC) == len(QC)

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
    ##    assert len(PS) == len(QS)

    x1 = 2.4048255576957727686e00
    x2 = 5.5200781102863106496e00
    x11 = 6.160e02
    x12 = -1.42444230422723137837e-03
    x21 = 1.4130e03
    x22 = 5.46860286310649596604e-04
    one_div_root_pi = 5.641895835477562869480794515607725858e-01

    def t1(x):  # x<=4
        y = x * x
        r = v_rational(y, P1, Q1)
        factor = (x + x1) * ((x - x11 / 256) - x12)
        return factor * r

    def t2(x):  # x<=8
        y = 1 - (x * x) / 64
        r = v_rational(y, P2, Q2)
        factor = (x + x2) * ((x - x21 / 256) - x22)
        return factor * r

    def t3(x):  # x>8
        y = 8 / x
        y2 = y * y
        rc = v_rational(y2, PC, QC)
        rs = v_rational(y2, PS, QS)
        factor = one_div_root_pi / jnp.sqrt(x)
        sx = jnp.sin(x)
        cx = jnp.cos(x)
        return factor * (rc * (cx + sx) - y * rs * (sx - cx))

    x = jnp.abs(x)
    return jnp.select(
        [x == 0, x <= 4, x <= 8, x > 8], [1, t1(x), t2(x), t3(x)], default=x
    ).reshape(orig_shape)
