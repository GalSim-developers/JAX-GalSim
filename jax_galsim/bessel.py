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

    idx = jnp.where(x <= 2.0, 0, jnp.where(x <= 8.0, 1, 2))
    return jax.lax.switch(idx, [k0_small, k0_medium, k0_large], x)


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

    idx = jnp.where(x <= 2.0, 0, jnp.where(x <= 8.0, 1, 2))
    return jax.lax.switch(idx, [k1_small, k1_medium, k1_large], x)


# =====================================================================
# Modified Bessel K_v(x) for fractional order
# Ported from TensorFlow Probability's bessel.py (Apache 2.0 License)
# Uses Temme's method (|v| < 50) + Olver's uniform asymptotic (|v| >= 50)
# =====================================================================

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
    """Numerically stable computation of sqrt(1 + x^2)."""
    eps = jnp.finfo(jnp.float64).eps
    return jnp.where(
        jnp.abs(x) * jnp.sqrt(eps) <= 1.0,
        jnp.exp(0.5 * jnp.log1p(x * x)),
        jnp.abs(x),
    )


def _evaluate_temme_coeffs(v):
    """Numerically stable computation of gamma-related coefficients for Temme's method.

    Computes:
      coeff1 = (1/Gamma(1-v) - 1/Gamma(1+v)) / (2v)
      coeff2 = (1/Gamma(1-v) + 1/Gamma(1+v)) / 2
      gamma1pv = 1/Gamma(1+v)
      gamma1mv = 1/Gamma(1-v)

    Uses Chebyshev expansions for numerical stability (avoids catastrophic cancellation).
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
    w = 8.0 * v * v - 1.0
    y = 2.0 * w

    # Clenshaw's recurrence for coeff1
    prev = 0.0
    current = 0.0
    for i in reversed(range(1, len(coeff1_coeffs))):
        temp = current
        current = y * current - prev + coeff1_coeffs[i]
        prev = temp
    coeff1 = w * current - prev + 0.5 * coeff1_coeffs[0]

    # Clenshaw's recurrence for coeff2
    prev = 0.0
    current = 0.0
    for i in reversed(range(1, len(coeff2_coeffs))):
        temp = current
        current = y * current - prev + coeff2_coeffs[i]
        prev = temp
    coeff2 = w * current - prev + 0.5 * coeff2_coeffs[0]

    gamma1pv = coeff2 - v * coeff1
    gamma1mv = coeff2 + v * coeff1
    return coeff1, coeff2, gamma1pv, gamma1mv


def _temme_series_kve(v, z):
    """Compute Kve(v, z) and Kve(v+1, z) via Temme power series.

    Assumes |v| < 0.5 and |z| <= 2 for fast convergence.
    Returns exponentially scaled values: Kv(v,z)*exp(z).
    """
    tol = jnp.finfo(jnp.float64).eps

    coeff1, coeff2, gamma1pv_inv, gamma1mv_inv = _evaluate_temme_coeffs(v)

    z_sq = z * z
    logzo2 = jnp.log(z / 2.0)
    mu = -v * logzo2
    sinc_v = jnp.where(v == 0.0, 1.0, jnp.sin(jnp.pi * v) / (jnp.pi * v))
    sinhc_mu = jnp.where(mu == 0.0, 1.0, jnp.sinh(mu) / mu)

    initial_f = (coeff1 * jnp.cosh(mu) + coeff2 * (-logzo2) * sinhc_mu) / sinc_v
    initial_p = 0.5 * jnp.exp(mu) / gamma1pv_inv
    initial_q = 0.5 * jnp.exp(-mu) / gamma1mv_inv

    max_iterations = 1000

    def body_fn(carry):
        should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum = carry
        f = jnp.where(
            should_stop,
            f,
            (index * f + p + q) / (index * index - v * v),
        )
        p = jnp.where(should_stop, p, p / (index - v))
        q = jnp.where(should_stop, q, q / (index + v))
        h = p - index * f
        coeff = jnp.where(should_stop, coeff, coeff * z_sq / (4.0 * index))
        kv_sum = jnp.where(should_stop, kv_sum, kv_sum + coeff * f)
        kvp1_sum = jnp.where(should_stop, kvp1_sum, kvp1_sum + coeff * h)
        index = index + 1.0
        should_stop = (jnp.abs(coeff * f) < jnp.abs(kv_sum) * tol) | (
            index > max_iterations
        )
        return (should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum)

    def cond_fn(carry):
        should_stop = carry[0]
        return ~should_stop

    init = (
        jnp.array(False),
        1.0,
        initial_f,
        initial_p,
        initial_q,
        1.0,
        initial_f,
        initial_p,
    )
    _, _, _, _, _, _, kv_sum, kvp1_sum = jax.lax.while_loop(cond_fn, body_fn, init)

    # Convert to exponentially scaled: kve = kv * exp(z)
    kve = kv_sum * jnp.exp(z)
    kvep1 = 2.0 * kvp1_sum * jnp.exp(z) / z

    return kve, kvep1


def _continued_fraction_kve(v, z):
    """Compute Kve(v, z) and Kve(v+1, z) via Steed's continued fraction.

    Assumes |v| < 0.5 and |z| > 2.
    Returns exponentially scaled values: Kv(v,z)*exp(z).
    """
    tol = jnp.finfo(jnp.float64).eps
    max_iterations = 1000

    initial_numerator = v * v - 0.25
    initial_denominator = 2.0 * (z + 1.0)
    initial_ratio = 1.0 / initial_denominator
    initial_seq = -initial_numerator

    def steeds_body(carry):
        (
            should_stop,
            index,
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
        ) = carry

        partial_numerator = partial_numerator - 2.0 * (index - 1.0)
        c = jnp.where(should_stop, c, -c * partial_numerator / index)
        next_k = (k_0 - partial_denominator * k_1) / partial_numerator
        k_0 = jnp.where(should_stop, k_0, k_1)
        k_1 = jnp.where(should_stop, k_1, next_k)
        q = jnp.where(should_stop, q, q + c * next_k)
        partial_denominator = partial_denominator + 2.0
        denominator_ratio = 1.0 / (
            partial_denominator + partial_numerator * denominator_ratio
        )
        convergent_difference = jnp.where(
            should_stop,
            convergent_difference,
            convergent_difference * (partial_denominator * denominator_ratio - 1.0),
        )
        hypergeometric_ratio = jnp.where(
            should_stop,
            hypergeometric_ratio,
            hypergeometric_ratio + convergent_difference,
        )
        hypergeometric_sum = jnp.where(
            should_stop,
            hypergeometric_sum,
            hypergeometric_sum + q * convergent_difference,
        )
        index = index + 1.0
        should_stop = (
            jnp.abs(q * convergent_difference) < jnp.abs(hypergeometric_sum) * tol
        ) | (index > max_iterations)
        return (
            should_stop,
            index,
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
        )

    def cond_fn(carry):
        return ~carry[0]

    init = (
        jnp.array(False),
        2.0,
        initial_numerator,
        initial_denominator,
        initial_ratio,
        initial_ratio,
        initial_ratio,
        0.0,
        1.0,
        initial_seq,
        initial_seq,
        1.0 - initial_numerator * initial_ratio,
    )
    result = jax.lax.while_loop(cond_fn, steeds_body, init)
    hypergeometric_ratio = result[6]
    hypergeometric_sum = result[11]

    # log(kve) = 0.5*log(pi/(2z)) - log(hypergeometric_sum)
    log_kve = 0.5 * jnp.log(jnp.pi / (2.0 * z)) - jnp.log(hypergeometric_sum)
    log_kvp1e = (
        log_kve
        + jnp.log1p(2.0 * (v + z + initial_numerator * hypergeometric_ratio))
        - jnp.log(z)
        - jnp.log(2.0)
    )
    return jnp.exp(log_kve), jnp.exp(log_kvp1e)


def _olver_kve(v, z):
    """Compute Kve(v, z) using Olver's uniform asymptotic expansion.

    Valid for |v| >= 50. Returns exponentially scaled value: Kv(v,z)*exp(z).
    """
    v_abs = jnp.abs(v)
    w = z / v_abs
    t = 1.0 / _sqrt1px2(w)

    divisor = v_abs
    kve_sum = 1.0

    # Evaluate the Olver polynomial terms using Horner's method
    for i in range(len(_ASYMPTOTIC_OLVER_COEFFICIENTS)):
        coeff = 0.0
        for c in _ASYMPTOTIC_OLVER_COEFFICIENTS[i]:
            coeff = coeff * t + c
        term = coeff / divisor
        # For K_v, signs alternate: (-1)^i
        kve_sum = kve_sum + (term if i % 2 == 1 else -term)
        divisor = divisor * v_abs

    # log(kve) = 0.5*log(pi*t/(2*v_abs)) - v_abs*shared_prefactor
    shared_prefactor = 1.0 / (_sqrt1px2(w) + w) + jnp.log(w) - jnp.log1p(1.0 / t)
    log_k_prefactor = (
        0.5 * jnp.log(jnp.pi * t / (2.0 * v_abs)) - v_abs * shared_prefactor
    )

    log_kve = log_k_prefactor + jnp.log(kve_sum)
    return jnp.exp(log_kve)


def _temme_kve(v, x):
    """Compute Kve(v, x) using Temme's method for |v| < 50.

    Reduces to fractional order |u| <= 0.5, computes Kve(u, x) and Kve(u+1, x),
    then uses forward recurrence to reach order v.
    Returns exponentially scaled value: Kv(v,x)*exp(x).
    """
    v = jnp.abs(v)
    n = jnp.round(v)
    u = v - n

    # Branchless: compute both methods with safe inputs, select with jnp.where
    small_x = jnp.where(x <= 2.0, x, 0.1)
    large_x = jnp.where(x > 2.0, x, 1000.0)

    temme_kue, temme_kuep1 = _temme_series_kve(u, small_x)
    cf_kue, cf_kuep1 = _continued_fraction_kve(u, large_x)

    kue = jnp.where(x <= 2.0, temme_kue, cf_kue)
    kuep1 = jnp.where(x <= 2.0, temme_kuep1, cf_kuep1)

    # Forward recurrence: K_{v+1}(z) = (2v/z)*K_v(z) + K_{v-1}(z)
    # This recurrence is also satisfied by Kv*exp(z) (the exponentially scaled form).
    def bessel_recurrence(carry):
        index, kve, kvep1 = carry
        next_kvep1 = 2.0 * (u + index) * kvep1 / x + kve
        kve = jnp.where(index > n, kve, kvep1)
        kvep1 = jnp.where(index > n, kvep1, next_kvep1)
        return (index + 1.0, kve, kvep1)

    def recurrence_cond(carry):
        index = carry[0]
        return index <= n

    _, kve, _ = jax.lax.while_loop(
        recurrence_cond, bessel_recurrence, (1.0, kue, kuep1)
    )
    return kve


def _kve_core(nu, x):
    """Core dispatcher for Kve(nu, x) = Kv(nu, x) * exp(x).

    Branchless: computes both Olver and Temme with safe dummy inputs,
    selects based on |nu| >= 50.
    """
    nu = jnp.abs(nu)

    # Safe inputs: avoid invalid regions for each method
    small_nu = jnp.where(nu < 50.0, nu, 0.1)
    large_nu = jnp.where(nu >= 50.0, nu, 1000.0)

    olver_result = _olver_kve(large_nu, x)
    temme_result = _temme_kve(small_nu, x)

    return jnp.where(nu >= 50.0, olver_result, temme_result)


@jax.jit
def _bessel_kn_recurrence(n, x, k0_val, k1_val):
    """Compute K_n(x) for integer n >= 2 using forward recurrence.

    Uses the recurrence relation:
        K_{n+1}(x) = K_{n-1}(x) + (2*n/x) * K_n(x)

    For n <= 5, uses direct computation without a loop.
    For n > 5, uses fori_loop with 399 iterations.

    Args:
        n: Integer order (n >= 2)
        x: Argument value
        k0_val: Pre-computed K_0(x)
        k1_val: Pre-computed K_1(x)

    Returns:
        K_n(x)
    """

    def small_n():
        k2 = k0_val + (2.0 / x) * k1_val
        k3 = k1_val + (4.0 / x) * k2
        k4 = k2 + (6.0 / x) * k3
        k5 = k3 + (8.0 / x) * k4
        return jnp.select(
            [n == 2, n == 3, n == 4, n == 5], [k2, k3, k4, k5], default=k5
        )

    def large_n():
        def body_fn(i, carry):
            k_prev, k_curr = carry
            should_update = i < n
            k_next = jnp.where(should_update, k_prev + (2.0 * i / x) * k_curr, k_curr)
            k_prev_new = jnp.where(should_update, k_curr, k_prev)
            return (k_prev_new, k_next)

        _, k_n = jax.lax.fori_loop(1, 400, body_fn, (k0_val, k1_val))
        return k_n

    return jax.lax.cond(n <= 5, small_n, large_n)


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

    idx = jnp.where(n == 0, 0, jnp.where(n == 1, 1, 2))
    return jax.lax.switch(
        idx,
        [
            lambda n, x, k0, k1: k0,
            lambda n, x, k0, k1: k1,
            lambda n, x, k0, k1: _bessel_kn_recurrence(n, x, k0, k1),
        ],
        n,
        x,
        k0,
        k1,
    )


def _kv_scalar(nu, x):
    """Scalar implementation of K_ν(x) using TFP-ported Temme + Olver algorithms."""
    nu = 1.0 * nu
    x = 1.0 * x
    nu = jnp.abs(nu)  # K_{-v} = K_v

    # Compute via exponentially scaled form for numerical stability
    # Use a safe x for the core computation (avoid x=0 which causes issues)
    safe_x = jnp.where(x > 0.0, x, 1.0)
    kve = _kve_core(nu, safe_x)
    result = kve * jnp.exp(-safe_x)

    # Edge cases
    result = jnp.where(x == 0.0, jnp.inf, result)
    result = jnp.where(x < 0.0, jnp.nan, result)
    return result


@implements(_galsim.bessel.kv)
@jax.jit
def _kv_impl(nu, x):
    """Modified Bessel function of the second kind K_ν(x) - internal implementation.

    Uses TFP-ported Temme + Olver algorithms for all orders.
    Handles both scalar and array inputs via jax.vmap.
    """
    nu = jnp.asarray(1.0 * nu)
    x = jnp.asarray(1.0 * x)
    out_shape = jnp.broadcast_shapes(jnp.shape(nu), jnp.shape(x))
    if out_shape == ():
        return _kv_scalar(nu, x)
    nu_bc, x_bc = jnp.broadcast_arrays(nu, x)
    flat_nu = nu_bc.ravel()
    flat_x = x_bc.ravel()
    return jax.vmap(_kv_scalar)(flat_nu, flat_x).reshape(out_shape)


@jax.custom_vjp
def kv(nu, x):
    """Modified Bessel function of the second kind K_ν(x) with custom gradients.

    Uses TFP-ported Temme + Olver algorithms. Custom gradients via:
        ∂K_ν/∂x = -1/2 * (K_{ν-1}(x) + K_{ν+1}(x))

    Gradient w.r.t. ν is not supported (returns zero).
    """
    return _kv_impl(nu, x)


def _kv_fwd(nu, x):
    kv_val = _kv_impl(nu, x)
    kv_prev = _kv_impl(nu - 1.0, x)
    kv_next = _kv_impl(nu + 1.0, x)
    return kv_val, (nu, x, kv_prev, kv_next)


def _kv_bwd(residuals, g):
    nu, x, kv_prev, kv_next = residuals
    grad_x = -0.5 * (kv_prev + kv_next) * g
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
