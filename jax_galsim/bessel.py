import galsim as _galsim
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from jax_galsim.core.utils import implements


# Chebyshev series evaluation
# Ported from SLATEC dcsevl function in BesselJ.cpp lines 1666-1676
@jax.jit
def _dcsevl(x, cs):
    """Evaluate Chebyshev series.

    Evaluates the N-term Chebyshev series cs at x using Clenshaw's
    recurrence algorithm. Only half the first coefficient is summed.

    Args:
        x: Value at which to evaluate series (should be in [-1, 1])
        cs: Array of Chebyshev series coefficients

    Returns:
        Evaluated series value
    """
    n = len(cs)
    b0 = jnp.array(0.0)
    b1 = jnp.array(0.0)
    b2 = jnp.array(0.0)
    twox = 2.0 * jnp.squeeze(x)

    def body_fn(i, carry):
        b0, b1, b2 = carry
        b2 = b1
        b1 = b0
        b0 = twox * b1 - b2 + cs[n - 1 - i]
        return (b0, b1, b2)

    b0, b1, b2 = jax.lax.fori_loop(0, n, body_fn, (b0, b1, b2))
    return 0.5 * (b0 - b2)


# the code here for Si, f, g and _si_small_pade is taken from galsim/src/math/Sinc.cpp
@jax.jit
def _f_pade(x, x2):
    # fmt: off
    y = 1. / x2
    return (
            (1. +  # noqa: E226
             y*(7.44437068161936700618e2 +  # noqa: E226
                y*(1.96396372895146869801e5 +  # noqa: E226
                   y*(2.37750310125431834034e7 +  # noqa: E226
                      y*(1.43073403821274636888e9 +  # noqa: E226
                         y*(4.33736238870432522765e10 +  # noqa: E226
                            y*(6.40533830574022022911e11 +  # noqa: E226
                               y*(4.20968180571076940208e12 +  # noqa: E226
                                  y*(1.00795182980368574617e13 +  # noqa: E226
                                     y*(4.94816688199951963482e12 +  # noqa: E226
                                        y*(-4.94701168645415959931e11)))))))))))  # noqa: E226
            / (x*(1. +  # noqa: E226
                  y*(7.46437068161927678031e2 +  # noqa: E226
                     y*(1.97865247031583951450e5 +  # noqa: E226
                        y*(2.41535670165126845144e7 +  # noqa: E226
                           y*(1.47478952192985464958e9 +  # noqa: E226
                              y*(4.58595115847765779830e10 +  # noqa: E226
                                 y*(7.08501308149515401563e11 +  # noqa: E226
                                    y*(5.06084464593475076774e12 +  # noqa: E226
                                       y*(1.43468549171581016479e13 +  # noqa: E226
                                          y*(1.11535493509914254097e13)))))))))))  # noqa: E226
    )
    # fmt: on


@jax.jit
def _g_pade(x, x2):
    # fmt: off
    y = 1. / x2
    return (
            y*(1. +  # noqa: E226
               y*(8.1359520115168615e2 +  # noqa: E226
                  y*(2.35239181626478200e5 +  # noqa: E226
                     y*(3.12557570795778731e7 +  # noqa: E226
                        y*(2.06297595146763354e9 +  # noqa: E226
                           y*(6.83052205423625007e10 +  # noqa: E226
                              y*(1.09049528450362786e12 +  # noqa: E226
                                 y*(7.57664583257834349e12 +  # noqa: E226
                                    y*(1.81004487464664575e13 +  # noqa: E226
                                       y*(6.43291613143049485e12 +  # noqa: E226
                                          y*(-1.36517137670871689e12)))))))))))  # noqa: E226
            / (1. +  # noqa: E226
               y*(8.19595201151451564e2 +  # noqa: E226
                  y*(2.40036752835578777e5 +  # noqa: E226
                     y*(3.26026661647090822e7 +  # noqa: E226
                        y*(2.23355543278099360e9 +  # noqa: E226
                           y*(7.87465017341829930e10 +  # noqa: E226
                              y*(1.39866710696414565e12 +  # noqa: E226
                                 y*(1.17164723371736605e13 +  # noqa: E226
                                    y*(4.01839087307656620e13 +  # noqa: E226
                                       y*(3.99653257887490811e13))))))))))  # noqa: E226
    )
    # fmt: on


@jax.jit
def _si_small_pade(x, x2):
    # fmt: off
    return (
                x*(1. +  # noqa: E226
                   x2*(-4.54393409816329991e-2 +  # noqa: E226
                       x2*(1.15457225751016682e-3 +  # noqa: E226
                           x2*(-1.41018536821330254e-5 +  # noqa: E226
                               x2*(9.43280809438713025e-8 +  # noqa: E226
                                   x2*(-3.53201978997168357e-10 +  # noqa: E226
                                       x2*(7.08240282274875911e-13 +  # noqa: E226
                                           x2*(-6.05338212010422477e-16))))))))  # noqa: E226
                / (1. +  # noqa: E226
                   x2*(1.01162145739225565e-2 +  # noqa: E226
                       x2*(4.99175116169755106e-5 +  # noqa: E226
                           x2*(1.55654986308745614e-7 +  # noqa: E226
                               x2*(3.28067571055789734e-10 +  # noqa: E226
                                   x2*(4.5049097575386581e-13 +  # noqa: E226
                                       x2*(3.21107051193712168e-16)))))))  # noqa: E226
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


# Modified Bessel K functions - ported from SLATEC in GalSim BesselK.cpp


@jax.jit
def _bessel_k0(x):
    """Modified Bessel function K_0(x) for x > 0.

    Implements SLATEC dbesk0 using Chebyshev series for x <= 2
    and asymptotic expansion for x > 2.

    Reference: BesselK.cpp lines 253-284, 286-442
    """
    # Chebyshev coefficients for K_0 (small x)
    # fmt: off
    bk0cs = jnp.array([
        -0.0353273932339027687201140060063153,
        0.344289899924628486886344927529213,
        0.0359799365153615016265721303687231,
        0.00126461541144692592338479508673447,
        2.28621210311945178608269830297585e-5,
        2.53479107902614945730790013428354e-7,
        1.90451637722020885897214059381366e-9,
        1.03496952576336245851008317853089e-11,
        4.25981614279108257652445327170133e-14,
        1.3744654358807508969423832544e-16,
        3.57089652850837359099688597333333e-19,
        7.63164366011643737667498666666666e-22,
        1.36542498844078185908053333333333e-24,
        2.07527526690666808319999999999999e-27,
        2.7128142180729856e-30,
        3.08259388791466666666666666666666e-33,
    ])

    # Asymptotic coefficients for 2 < x <= 8
    ak0cs = jnp.array([
        -0.07643947903327941424082978270088,
        -0.02235652605699819052023095550791,
        7.734181154693858235300618174047e-4,
        -4.281006688886099464452146435416e-5,
        3.08170017386297474365001482666e-6,
        -2.639367222009664974067448892723e-7,
        2.563713036403469206294088265742e-8,
        -2.742705549900201263857211915244e-9,
        3.169429658097499592080832873403e-10,
        -3.902353286962184141601065717962e-11,
        5.068040698188575402050092127286e-12,
        -6.889574741007870679541713557984e-13,
        9.744978497825917691388201336831e-14,
        -1.427332841884548505389855340122e-14,
        2.156412571021463039558062976527e-15,
        -3.34965425514956277218878205853e-16,
        5.335260216952911692145280392601e-17,
        -8.693669980890753807639622378837e-18,
        1.446404347862212227887763442346e-18,
    ])

    # Asymptotic coefficients for x > 8
    ak02cs = jnp.array([
        -0.01201869826307592239839346212452,
        -0.009174852691025695310652561075713,
        1.444550931775005821048843878057e-4,
        -4.013614175435709728671021077879e-6,
        1.567831810852310672590348990333e-7,
        -7.77011043852173771031579975446e-9,
        4.611182576179717882533130529586e-10,
        -3.158592997860565770526665803309e-11,
        2.435018039365041127835887814329e-12,
        -2.074331387398347897709853373506e-13,
        1.925787280589917084742736504693e-14,
        -1.927554805838956103600347182218e-15,
        2.062198029197818278285237869644e-16,
        -2.341685117579242402603640195071e-17,
        2.805902810643042246815178828458e-18,
    ])
    # fmt: on

    def k0_small(x):
        xsml = jnp.sqrt(4.0 * jnp.finfo(jnp.float64).eps)
        y = jnp.where(x > xsml, x * x, 0.0)
        return -jnp.log(0.5 * x) * jsp.i0(x) - 0.25 + _dcsevl(0.5 * y - 1.0, bk0cs)

    def k0_medium(x):
        return jnp.exp(-x) * (
            (_dcsevl((16.0 / x - 5.0) / 3.0, ak0cs) + 1.25) / jnp.sqrt(x)
        )

    def k0_large(x):
        return jnp.exp(-x) * ((_dcsevl(16.0 / x - 1.0, ak02cs) + 1.25) / jnp.sqrt(x))

    return jnp.where(
        x <= 2.0, k0_small(x), jnp.where(x <= 8.0, k0_medium(x), k0_large(x))
    )


@jax.jit
def _bessel_k1(x):
    """Modified Bessel function K_1(x) for x > 0.

    Implements SLATEC dbesk1 using Chebyshev series for x <= 2
    and asymptotic expansion for x > 2.

    Reference: BesselK.cpp lines 480-514, 516-655
    """
    # Chebyshev coefficients for K_1 (small x)
    # fmt: off
    bk1cs = jnp.array([
        0.025300227338947770532531120868533,
        -0.35315596077654487566723831691801,
        -0.12261118082265714823479067930042,
        -0.0069757238596398643501812920296083,
        -1.7302889575130520630176507368979e-4,
        -2.4334061415659682349600735030164e-6,
        -2.2133876307347258558315252545126e-8,
        -1.4114883926335277610958330212608e-10,
        -6.6669016941993290060853751264373e-13,
        -2.4274498505193659339263196864853e-15,
        -7.023863479386287597178379712e-18,
        -1.6543275155100994675491029333333e-20,
        -3.2338347459944491991893333333333e-23,
        -5.3312750529265274999466666666666e-26,
        -7.5130407162157226666666666666666e-29,
        -9.1550857176541866666666666666666e-32,
    ])

    # Asymptotic coefficients for 2 < x <= 8
    ak1cs = jnp.array([
        0.27443134069738829695257666227266,
        0.07571989953199367817089237814929,
        -0.0014410515564754061229853116175625,
        6.6501169551257479394251385477036e-5,
        -4.3699847095201407660580845089167e-6,
        3.5402774997630526799417139008534e-7,
        -3.3111637792932920208982688245704e-8,
        3.4459775819010534532311499770992e-9,
        -3.8989323474754271048981937492758e-10,
        4.7208197504658356400947449339005e-11,
        -6.047835662875356234537359156289e-12,
        8.1284948748658747888193837985663e-13,
        -1.1386945747147891428923915951042e-13,
        1.654035840846228232597294820509e-14,
        -2.4809025677068848221516010440533e-15,
        3.8292378907024096948429227299157e-16,
        -6.0647341040012418187768210377386e-17,
        9.8324256232648616038194004650666e-18,
        -1.6284168738284380035666620115626e-18,
    ])

    # Asymptotic coefficients for x > 8
    ak12cs = jnp.array([
        0.06379308343739001036600488534102,
        0.02832887813049720935835030284708,
        -2.475370673905250345414545566732e-4,
        5.771972451607248820470976625763e-6,
        -2.068939219536548302745533196552e-7,
        9.739983441381804180309213097887e-9,
        -5.585336140380624984688895511129e-10,
        3.732996634046185240221212854731e-11,
        -2.825051961023225445135065754928e-12,
        2.372019002484144173643496955486e-13,
        -2.176677387991753979268301667938e-14,
        2.157914161616032453939562689706e-15,
        -2.290196930718269275991551338154e-16,
        2.582885729823274961919939565226e-17,
        -3.07675264126846318762109817344e-18,
    ])
    # fmt: on

    def k1_small(x):
        xsml = 2.0 * jnp.sqrt(jnp.finfo(jnp.float64).eps)
        y = jnp.where(x > xsml, x * x, 0.0)
        return jnp.log(0.5 * x) * jsp.i1(x) + (_dcsevl(0.5 * y - 1.0, bk1cs) + 0.75) / x

    def k1_medium(x):
        return jnp.exp(-x) * (
            (_dcsevl((16.0 / x - 5.0) / 3.0, ak1cs) + 1.25) / jnp.sqrt(x)
        )

    def k1_large(x):
        return jnp.exp(-x) * ((_dcsevl(16.0 / x - 1.0, ak12cs) + 1.25) / jnp.sqrt(x))

    return jnp.where(
        x <= 2.0, k1_small(x), jnp.where(x <= 8.0, k1_medium(x), k1_large(x))
    )


@jax.jit
def _bessel_kv_asymptotic_large_nu(nu, x):
    """Compute K_ν(x) using uniform asymptotic expansion for large ν (ν ≥ 35).

    Implements the SLATEC dasyik algorithm for K Bessel functions.

    Reference: BesselI.cpp lines 844-953 (dasyik function)
    Algorithm: Olver, F.W.J. (1962), Tables of Bessel Functions of Moderate or Large Orders
    """
    # fmt: off
    sqrt_half_pi = 1.25331413731550025  # sqrt(pi/2)
    c = jnp.array([
        -0.208333333333333, 0.125, 0.334201388888889,
        -0.401041666666667, 0.0703125, -1.02581259645062, 1.84646267361111,
        -0.8912109375, 0.0732421875, 4.66958442342625, -11.207002616223,
        8.78912353515625, -2.3640869140625, 0.112152099609375,
        -28.2120725582002, 84.6362176746007, -91.81824154324,
        42.5349987453885, -7.36879435947963, 0.227108001708984,
        212.570130039217, -765.252468141182, 1059.990452528,
        -699.579627376133, 218.190511744212, -26.4914304869516,
        0.572501420974731, -1919.45766231841, 8061.72218173731,
        -13586.5500064341, 11655.3933368645, -5305.6469786134,
        1200.90291321635, -108.090919788395, 1.72772750258446,
        20204.2913309661, -96980.5983886375, 192547.001232532,
        -203400.177280416, 122200.464983017, -41192.6549688976,
        7109.51430248936, -493.915304773088, 6.07404200127348,
        -242919.187900551, 1311763.61466298, -2998015.91853811,
        3763271.2976564, -2813563.22658653, 1268365.27332162,
        -331645.172484564, 45218.7689813627, -2499.83048181121,
        24.3805296995561, 3284469.85307204, -19706819.1184322,
        50952602.4926646, -74105148.2115327, 66344512.274729,
        -37567176.6607634, 13288767.1664218, -2785618.12808645,
        308186.404612662, -13886.089753717, 110.017140269247
    ])
    # fmt: on

    fn = nu
    z = x / fn
    ra = jnp.sqrt(z * z + 1.0)
    gln = jnp.log((ra + 1.0) / z)
    coef = jnp.exp(-fn * (ra - gln))
    t = 1.0 / ra
    t2 = t * t
    t = -t / fn
    s2 = 1.0
    ap = 1.0
    coeff_idx = 0

    def body_fn(k, carry):
        s2, ap, coeff_idx = carry
        s1 = c[coeff_idx]
        new_idx = coeff_idx + 1

        def inner_body(j, inner_carry):
            s1_val, idx = inner_carry
            should_update = j < k + 1
            s1_new = jnp.where(should_update, s1_val * t2 + c[idx], s1_val)
            idx_new = jnp.where(should_update, idx + 1, idx)
            return (s1_new, idx_new)

        s1, new_idx = jax.lax.fori_loop(2, 12, inner_body, (s1, new_idx))

        ap_new = ap * t
        s2_new = s2 + ap_new * s1
        return (s2_new, ap_new, coeff_idx + k)

    s2, ap, _ = jax.lax.fori_loop(2, 12, body_fn, (s2, ap, coeff_idx))

    return s2 * coef * jnp.sqrt(jnp.abs(t)) * sqrt_half_pi


@jax.jit
def _bessel_kv_small_x(nu, x):
    """Compute K_ν(x) using power series for small x (x ≤ 2.0).

    Implements the SLATEC dbsknu series expansion.

    Reference: BesselK.cpp lines 880-965 (inside dbsknu)
    Algorithm: K_ν(x) = (π/2) * [I_{-ν}(x) - I_ν(x)] / sin(πν)
    """
    # fmt: off
    cc = jnp.array([
        0.577215664901533, -0.0420026350340952,
        -0.0421977345555443, 0.007218943246663, -2.152416741149e-4,
        -2.01348547807e-5, 1.133027232e-6, 6.116095e-9
    ])
    # fmt: on

    pi = jnp.pi
    tol = jnp.maximum(jnp.finfo(jnp.float64).eps, 1e-15)

    a1 = 1.0 - nu
    a2 = nu + 1.0
    t1 = 1.0 / jsp.gamma(a1)
    t2 = 1.0 / jsp.gamma(a2)

    dnu2 = jnp.where(jnp.abs(nu) >= tol, nu * nu, 0.0)

    def g1_small_nu():
        s = cc[0]
        ak = 1.0

        def body_fn(k, carry):
            s_val, ak_val = carry
            ak_new = ak_val * dnu2
            s_new = s_val + cc[k] * ak_new
            return (s_new, ak_new)

        s_final, _ = jax.lax.fori_loop(1, 8, body_fn, (s, ak))
        return -s_final

    def g1_large_nu():
        return (t1 - t2) / (nu + nu)

    g1 = jnp.where(jnp.abs(nu) > 0.1, g1_large_nu(), g1_small_nu())
    g2 = (t1 + t2) * 0.5

    rx = 2.0 / x
    flrx = jnp.log(rx)
    fmu = nu * flrx

    smu = jnp.where(nu != 0.0, jnp.sinh(fmu) / fmu, 1.0)
    fc = jnp.where(nu != 0.0, nu * pi / jnp.sin(nu * pi), 1.0)

    f = fc * (g1 * jnp.cosh(fmu) + g2 * flrx * smu)
    fc_exp = jnp.exp(fmu)
    p = fc_exp * 0.5 / t2
    q = 0.5 / (fc_exp * t1)

    ak = 1.0
    ck = 1.0
    bk = 1.0
    s1 = f
    s2 = p

    # Power series iteration (up to 17 terms)
    cx = x * x * 0.25

    def series_body(i, carry):
        f_val, p_val, q_val, ak_val, ck_val, bk_val, s1_val, s2_val = carry

        f_new = (ak_val * f_val + p_val + q_val) / (bk_val - dnu2)
        p_new = p_val / (ak_val - nu)
        q_new = q_val / (ak_val + nu)
        ck_new = ck_val * cx / ak_val
        s1_new = s1_val + ck_new * f_new
        s2_new = s2_val + ck_new * (p_new - ak_val * f_new)
        bk_new = bk_val + ak_val + ak_val + 1.0
        ak_new = ak_val + 1.0

        return (f_new, p_new, q_new, ak_new, ck_new, bk_new, s1_new, s2_new)

    init = (f, p, q, ak, ck, bk, s1, s2)
    _, _, _, _, _, _, s1_final, _ = jax.lax.cond(
        x >= tol,
        lambda: jax.lax.fori_loop(0, 17, series_body, init),
        lambda: init,
    )

    return s1_final


@jax.jit
def _bessel_kv_miller(nu, x):
    """Compute K_ν(x) using Miller's algorithm for moderate x (2.0 < x ≤ 17.0).

    Implements the SLATEC Miller algorithm using continued fractions,
    then uses forward recursion to get from dnu to nu.

    Reference: BesselK.cpp lines 677-1036 (dbsknu function)
    """
    pi = jnp.pi
    sqrt_half_pi = 1.2533141373155  # sqrt(pi/2)
    tol = jnp.maximum(jnp.finfo(jnp.float64).eps, 1e-15)

    rx = 2.0 / x
    inu = jnp.floor(nu + 0.5).astype(int)
    dnu = nu - inu
    dnu2 = jnp.where(jnp.abs(dnu) >= tol, dnu * dnu, 0.0)

    coef = sqrt_half_pi / jnp.sqrt(x) * jnp.exp(-x)

    etest = jnp.cos(pi * dnu) / (pi * x * tol)
    fks = 1.0
    fhs = 0.25
    fk = 0.0
    ck = x + x + 2.0
    p1 = 0.0
    p2 = 1.0

    a_arr = jnp.zeros(160)
    b_arr = jnp.zeros(160)

    def forward_body(i, carry):
        (
            fk_val,
            fks_val,
            fhs_val,
            ck_val,
            p1_val,
            p2_val,
            a_arr_val,
            b_arr_val,
            k,
            converged,
        ) = carry

        fk_new = fk_val + 1.0
        ak = (fhs_val - dnu2) / (fks_val + fk_new)
        bk = ck_val / (fk_new + 1.0)
        pt = p2_val
        p2_new = bk * p2_val - ak * p1_val
        p1_new = pt

        a_arr_new = jnp.where(converged, a_arr_val, a_arr_val.at[k].set(ak))
        b_arr_new = jnp.where(converged, b_arr_val, b_arr_val.at[k].set(bk))

        ck_new = ck_val + 2.0
        fks_new = fks_val + fk_new + fk_new + 1.0
        fhs_new = fhs_val + fk_new + fk_new

        has_converged = (etest <= fk_new * jnp.abs(p1_new)) | converged
        k_new = jnp.where(has_converged, k, k + 1)

        return (
            fk_new,
            fks_new,
            fhs_new,
            ck_new,
            p1_new,
            p2_new,
            a_arr_new,
            b_arr_new,
            k_new,
            has_converged,
        )

    _, _, _, _, _, _, a_final, b_final, k_final, _ = jax.lax.fori_loop(
        0, 160, forward_body, (fk, fks, fhs, ck, p1, p2, a_arr, b_arr, 0, False)
    )

    s = 1.0
    p1_back = 0.0
    p2_back = 1.0

    def backward_body(i, carry):
        s_val, p1_val, p2_val = carry
        should_update = i < k_final
        idx = k_final - 1 - i
        pt = p2_val
        p2_new = jnp.where(
            should_update, (b_final[idx] * p2_val - p1_val) / a_final[idx], p2_val
        )
        p1_new = jnp.where(should_update, pt, p1_val)
        s_new = jnp.where(should_update, s_val + p2_new, s_val)

        return (s_new, p1_new, p2_new)

    s_final, p1_back_final, p2_back_final = jax.lax.fori_loop(
        0, 160, backward_body, (s, p1_back, p2_back)
    )

    s1 = coef * (p2_back_final / s_final)

    def no_recursion():
        return s1

    def with_recursion():
        s2 = s1 * (x + dnu + 0.5 - p1_back_final / p2_back_final) / x
        ck_rec = (dnu + dnu + 2.0) / x
        inu_adjusted = inu - 1

        def recursion_body(i, carry):
            s1_val, s2_val, ck_val = carry
            should_update = i < inu_adjusted
            st = s2_val
            s2_new = jnp.where(should_update, ck_val * s2_val + s1_val, s2_val)
            s1_new = jnp.where(should_update, st, s1_val)
            ck_new = jnp.where(should_update, ck_val + rx, ck_val)
            return (s1_new, s2_new, ck_new)

        _, s2_final, _ = jax.lax.fori_loop(0, 50, recursion_body, (s1, s2, ck_rec))
        return s2_final

    return jax.lax.cond(inu == 0, no_recursion, with_recursion)


@jax.jit
def _bessel_kv_asymptotic(nu, x):
    """Compute K_ν(x) using asymptotic expansion for large x (x > 17.0).

    Enhanced version with 30 terms for better accuracy.

    Reference: BesselK.cpp lines 799-832 (inside dbsknu)
    """
    sqrt_pi_2x = jnp.sqrt(jnp.pi / (2.0 * x))
    exp_neg_x = jnp.exp(-x)

    two_nu = nu + nu
    fmu = two_nu * two_nu
    ex = x * 8.0

    s = 1.0
    ck = 1.0
    sqk = 1.0
    ak = 0.0
    dk = ex

    def body_fn(_, carry):
        s_val, ck_val, dk_val, ak_val, sqk_val = carry
        ck_new = ck_val * (fmu - sqk_val) / dk_val
        s_new = s_val + ck_new
        dk_new = dk_val + ex
        ak_new = ak_val + 8.0
        sqk_new = sqk_val + ak_new
        return (s_new, ck_new, dk_new, ak_new, sqk_new)

    s_final, _, _, _, _ = jax.lax.fori_loop(0, 30, body_fn, (s, ck, dk, ak, sqk))

    return sqrt_pi_2x * exp_neg_x * s_final


@jax.jit
def _bessel_kv_fractional(nu, x):
    """Compute K_ν(x) for fractional ν using SLATEC algorithms.

    Decision tree based on the C++ dbesk/dbsknu implementation:
    - ν ≥ 35: Uniform asymptotic expansion (large ν)
    - x ≤ 2: Power series expansion
    - 2 < x ≤ 17: Miller's algorithm
    - x > 17: Asymptotic expansion for large x

    Reference: BesselK.cpp lines 62-215, 677-1036
    """
    return jnp.where(
        nu >= 35.0,
        _bessel_kv_asymptotic_large_nu(nu, x),
        jnp.where(
            x <= 2.0,
            _bessel_kv_small_x(nu, x),
            jnp.where(
                x <= 17.0, _bessel_kv_miller(nu, x), _bessel_kv_asymptotic(nu, x)
            ),
        ),
    )


@jax.jit
def _bessel_kn_recurrence(n, x, k0_val, k1_val):
    """Compute K_n(x) for integer n >= 2 using forward recurrence.

    Uses the recurrence relation:
        K_{n+1}(x) = K_{n-1}(x) + (2*n/x) * K_n(x)

    Args:
        n: Integer order (n >= 2)
        x: Argument value
        k0_val: Pre-computed K_0(x)
        k1_val: Pre-computed K_1(x)

    Returns:
        K_n(x)
    """

    def body_fn(i, carry):
        k_prev, k_curr = carry
        should_update = i < n
        k_next = jnp.where(should_update, k_prev + (2.0 * i / x) * k_curr, k_curr)
        k_prev_new = jnp.where(should_update, k_curr, k_prev)
        return (k_prev_new, k_next)

    _, k_n = jax.lax.fori_loop(1, 400, body_fn, (k0_val, k1_val))
    return k_n


@implements(_galsim.bessel.kn)
@jax.jit
def kn(n, x):
    """Modified Bessel function of the second kind K_n(x) for integer n.

    This is a convenience wrapper that uses the integer-order implementations
    for K_0, K_1, and recurrence for higher orders.

    Args:
        n: Integer order (can be negative, K_{-n} = K_n)
        x: Argument (must be positive)

    Returns:
        K_n(x)
    """
    n = jnp.abs(jnp.asarray(n, dtype=int))  # K_{-n} = K_n
    x = 1.0 * x  # promote to float

    k0 = _bessel_k0(x)
    k1 = _bessel_k1(x)

    return jnp.where(
        n == 0, k0, jnp.where(n == 1, k1, _bessel_kn_recurrence(n, x, k0, k1))
    )


@implements(_galsim.bessel.kv)
@jax.jit
def _kv_impl(nu, x):
    """Modified Bessel function of the second kind K_ν(x) - internal implementation.

    Implementation strategy:
    - Integer orders (ν = 0, 1, 2, ...): Pure JAX using Chebyshev series and recurrence
    - Half-integer orders (ν = 0.5, 1.5, ...): Pure JAX using closed-form expressions
    - Arbitrary fractional orders: Pure JAX using SLATEC algorithms

    All implementations are fully JIT-compatible and ported from the NETLIB SLATEC
    library via the C++ GalSim reference implementation.

    Args:
        nu: Order (can be negative, integer, or fractional)
        x: Argument (must be positive)

    Returns:
        K_ν(x)

    Reference: BesselK.cpp in GalSim C++ source
    """
    nu = 1.0 * nu  # promote to float
    x = 1.0 * x

    # K_{-nu}(x) = K_nu(x)
    nu = jnp.abs(nu)

    nu_int = jnp.floor(nu).astype(int)
    nu_frac = nu - nu_int

    is_half_integer = jnp.abs(nu_frac - 0.5) < 1e-10
    is_integer = nu_frac < 1e-10

    def integer_order(nu_int, x):
        k0 = _bessel_k0(x)
        k1 = _bessel_k1(x)
        return jnp.where(
            nu_int == 0,
            k0,
            jnp.where(nu_int == 1, k1, _bessel_kn_recurrence(nu_int, x, k0, k1)),
        )

    def half_integer_order(nu_int, x):
        sqrt_pi_2x = jnp.sqrt(jnp.pi / (2.0 * x))
        exp_neg_x = jnp.exp(-x)
        inv_x = 1.0 / x

        p0 = 1.0
        p1 = 1.0 + inv_x
        p2 = 1.0 + 3.0 * inv_x + 3.0 * inv_x**2
        p3 = 1.0 + 6.0 * inv_x + 15.0 * inv_x**2 + 15.0 * inv_x**3
        p4 = 1.0 + 10.0 * inv_x + 45.0 * inv_x**2 + 105.0 * inv_x**3 + 105.0 * inv_x**4

        poly = jnp.select(
            [nu_int == 0, nu_int == 1, nu_int == 2, nu_int == 3],
            [p0, p1, p2, p3],
            default=p4,
        )

        return sqrt_pi_2x * exp_neg_x * poly

    def very_small_x_limit(nu, x):
        return jnp.power(2.0, nu - 1.0) * jsp.gamma(nu) / jnp.power(x, nu)

    result_integer = integer_order(nu_int, x)
    result_half_integer = half_integer_order(nu_int, x)
    result_fractional = jnp.where(
        x < 1e-10, very_small_x_limit(nu, x), _bessel_kv_fractional(nu, x)
    )

    return jnp.where(
        is_integer,
        result_integer,
        jnp.where(is_half_integer, result_half_integer, result_fractional),
    )


@jax.custom_vjp
def kv(nu, x):
    """Modified Bessel function of the second kind K_ν(x) with custom gradients.

    Implementation strategy:
    - Integer orders (ν = 0, 1, 2, ...): Pure JAX using Chebyshev series and recurrence
    - Half-integer orders (ν = 0.5, 1.5, ...): Pure JAX using closed-form expressions
    - Arbitrary fractional orders: Pure JAX using SLATEC algorithms

    All implementations are fully JIT-compatible and ported from the NETLIB SLATEC
    library via the C++ GalSim reference implementation.

    Custom gradients are implemented using analytical derivative formulas based on
    Bessel function recurrence relations:
        ∂K_ν/∂x = -1/2 * (K_{ν-1}(x) + K_{ν+1}(x))

    This is derived from the modified Bessel recurrence relation:
        K_{ν-1}(x) + K_{ν+1}(x) = -2 * K'_ν(x)

    Note: Gradients with respect to the order parameter ν are not supported and will
    return zero. This follows the approach used in TensorFlow Probability, as gradients
    with respect to the order are rarely needed in practice.

    Args:
        nu: Order (can be negative, integer, or fractional)
        x: Argument (must be positive)

    Returns:
        K_ν(x)

    Reference:
        - BesselK.cpp in GalSim C++ source
        - TensorFlow Probability bessel.py for custom gradient approach
        - Abramowitz & Stegun 9.6.26 for derivative recurrence relations
    """
    return _kv_impl(nu, x)


def _kv_fwd(nu, x):
    """Forward pass for kv with custom gradients.

    Computes K_ν(x) and saves K_{ν-1}(x) and K_{ν+1}(x) for use in the backward pass.
    """
    kv_val = _kv_impl(nu, x)
    kv_prev = _kv_impl(nu - 1.0, x)
    kv_next = _kv_impl(nu + 1.0, x)
    return kv_val, (nu, x, kv_prev, kv_next)


def _kv_bwd(residuals, g):
    """Backward pass for kv with custom gradients.

    Uses the analytical derivative formula from Bessel recurrence relations:
        ∂K_ν/∂x = -1/2 * (K_{ν-1}(x) + K_{ν+1}(x))

    This formula comes from the modified Bessel function recurrence relation:
        K_{ν-1}(x) + K_{ν+1}(x) = -2 * K'_ν(x)

    Gradient with respect to ν is not supported and returns zero.
    """
    nu, x, kv_prev, kv_next = residuals
    # Analytical gradient: ∂K_ν/∂x = -1/2 * (K_{ν-1}(x) + K_{ν+1}(x))
    grad_x = -0.5 * (kv_prev + kv_next) * g
    # Gradient w.r.t. ν not supported (following TFP approach)
    grad_nu = jnp.zeros_like(nu)
    return (grad_nu, grad_x)


kv.defvjp(_kv_fwd, _kv_bwd)


@jax.jit
def _evaluate_rational(z, num, denom):
    return jnp.polyval(num[::-1], z) / jnp.polyval(denom[::-1], z)


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
