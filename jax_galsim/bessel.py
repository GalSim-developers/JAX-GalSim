import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _evaluate_rational(z, num, denom):
    """Evaluate polynomial ratio P(z)/Q(z)
    

       Inputs:
           z: the argument (real scalar)
           num: array of coefficients of P(z) = sum_i^N num[i] z^i
           denom : array of coefficients of Q(z) = sum_i^N denom[i] z^i

       Returns:
           the  estimate of P(z)/Q(z) 

       Esamples::
           >>> num = jnp.array([1.,2.,4.,-5.])
           >>> denom = jnp.array([-0.5,2,-3,-10])
           >>> _evaluate_rational(1.,num,denom) # DeviceArray(-0.17391304, dtype=float64)
           >>> v_ratio = jax.vmap(_evaluate_rational, in_axes=(0,None, None))
           >>> x=jnp.linspace(0,100,10000)
           >>> plt.plot(x,v_ratio(x, num,denom))
           >>> plt.xscale("log");
           >>> vgrad_ratio = jax.vmap(jax.jacfwd(_evaluate_rational), in_axes=(0,None, None))
           >>> plt.plot(x,vgrad_ratio(x, num,denom))

       Requirements: len(num) = len(denom) = N
    """
    assert len(num) == len(denom), "Num and Denom polynomial arrays must have same length"
    count = len(num)

    def true_fn_update(z):

      def body_true(val1):
        # decode val1
        s1, s2, i = val1 
        s1 *= z; s2 *= z; s1 += num[i]; s2 += denom[i];
        return s1, s2, i-1

      def cond_true(val1):
        s1, s2, i = val1 
        return i>=0

      val1_init = (num[-1], denom[-1], count-2)
      s1, s2, _ = jax.lax.while_loop(cond_true, body_true, val1_init)

      return s1/s2

    def false_fn_update(z):
      def body_false(val1):
        # decode val1
        s1, s2, i = val1 
        s1 *= z; s2 *= z; s1 += num[i]; s2 += denom[i];
        return s1, s2, i+1

      def cond_false(val1):
        s1, s2, i = val1 
        return i<count

      val1_init = (num[0], denom[0],1)
      s1, s2, _ = jax.lax.while_loop(cond_false, body_false, val1_init)

      return s1/s2


    return jnp.where(z<=1, true_fn_update(z), false_fn_update(1/z))  

#jitted & vectorized version 
v_rational = jax.jit(jax.vmap(_evaluate_rational, in_axes=(0,None, None)))



@jax.jit
def J0(x):
    """ Bessel function of the first kind J0(x)
        code from Boost C++ implementation
        boost/math/special_functions/detail/bessel_j0.hpp

        Examples::

            >>> x = jnp.linspace(0,300,10_000)
            >>> plt.plot(x,J0(x))
            >>> plt.plot(x,jax.vmap(jax.jacfwd(J0))(x))

        Inputs:
            x: scalar/array of real(s)

        Outputs:
            J0(x) with same shape as x
    
    """
    orig_shape = x.shape
    
    x = jnp.atleast_1d(x)

    P1 = jnp.array([-4.1298668500990866786e+11,
                      2.7282507878605942706e+10,
                      -6.2140700423540120665e+08,
                      6.6302997904833794242e+06,
                      -3.6629814655107086448e+04,
                      1.0344222815443188943e+02,
                      -1.2117036164593528341e-01]) 
    Q1 = jnp.array([2.3883787996332290397e+12,
                      2.6328198300859648632e+10,
                      1.3985097372263433271e+08,
                      4.5612696224219938200e+05,
                      9.3614022392337710626e+02,
                      1.0,
                      0.0])
##    assert len(P1) == len(Q1)

    P2 = jnp.array([-1.8319397969392084011e+03,
                    -1.2254078161378989535e+04,
                      -7.2879702464464618998e+03,
                      1.0341910641583726701e+04,
                      1.1725046279757103576e+04,
                      4.4176707025325087628e+03,
                      7.4321196680624245801e+02,
                      4.8591703355916499363e+01])
    Q2 = jnp.array([-3.5783478026152301072e+05,
                      2.4599102262586308984e+05,
                      -8.4055062591169562211e+04,
                      1.8680990008359188352e+04,
                      -2.9458766545509337327e+03,
                      3.3307310774649071172e+02,
                      -2.5258076240801555057e+01,
                      1.0])
##      assert len(P2) == len(Q2)


    PC = jnp.array([2.2779090197304684302e+04,
                      4.1345386639580765797e+04,
                      2.1170523380864944322e+04,
                      3.4806486443249270347e+03,
                      1.5376201909008354296e+02,
                      8.8961548424210455236e-01])
    QC = jnp.array([2.2779090197304684318e+04,
                      4.1370412495510416640e+04,
                      2.1215350561880115730e+04,
                      3.5028735138235608207e+03,
                      1.5711159858080893649e+02,
                      1.0])

##      assert len(PC) == len(QC)


    PS = jnp.array([-8.9226600200800094098e+01,
                    -1.8591953644342993800e+02,
                      -1.1183429920482737611e+02,
                      -2.2300261666214198472e+01,
                      -1.2441026745835638459e+00,
                      -8.8033303048680751817e-03])
    QS = jnp.array([5.7105024128512061905e+03,
                      1.1951131543434613647e+04,
                      7.2642780169211018836e+03,
                      1.4887231232283756582e+03,
                      9.0593769594993125859e+01,
                      1.0])
##    assert len(PS) == len(QS)


    x1 = 2.4048255576957727686e+00
    x2 = 5.5200781102863106496e+00
    x11 = 6.160e+02
    x12 = -1.42444230422723137837e-03
    x21 = 1.4130e+03
    x22 = 5.46860286310649596604e-04
    one_div_root_pi =  5.641895835477562869480794515607725858e-01

    def t1(x):  # x<=4
        y = x * x
        r = v_rational(y, P1, Q1)
        factor = (x + x1) * ((x - x11/256) - x12);
        return factor * r

    def t2(x): # x<=8
        y = 1 - (x * x)/64
        r = v_rational(y, P2, Q2)
        factor = (x + x2) * ((x - x21/256) - x22)
        return factor * r

    def t3(x): #x>8
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
        [x == 0, x <= 4, x <= 8, x>8],
        [1, t1(x), t2(x), t3(x)],
        default = x).reshape(orig_shape)
