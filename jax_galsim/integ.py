from functools import partial

import galsim as _galsim
import jax.lax
import jax.numpy as jnp
from quadax import quadgk

from jax_galsim.core.utils import implements


@implements(
    _galsim.integ.int1d,
    lax_description=(
        """\
The JAX-GalSim package uses the adaptive Gauss-Kronrod-Patterson
method implemented in the ``quadax`` package. Some import caveats are: "

- This implementation is different than the one in GalSim and lacks some features that
  greatly enhance galsim's accuracy.
- The JAX-GalSim implementation returns NaN on error/non-convergence instead of
  rasing an exception.
"""
    ),
)
@partial(jax.jit, static_argnames=("func", "_wrap_as_callback"))
def int1d(
    func,
    min,
    max,
    rel_err=1.0e-6,
    abs_err=1.0e-12,
    _wrap_as_callback=False,
    _inf_cutoff=1e4,
):
    # the hidden _wrap_as_callback keyword is used for testing against galsim
    # if true, we assume the input function is pure python and wrap it so it
    # can be used with jax
    if _wrap_as_callback:

        @jax.jit
        def _func(x):
            rdt = jax.ShapeDtypeStruct(x.shape, x.dtype)
            return jax.pure_callback(func, rdt, x)
    else:
        _func = func

    _min = jax.lax.cond(
        jnp.abs(min) > _inf_cutoff,
        lambda: jnp.sign(min) * jnp.inf,
        lambda: jnp.float_(min),
    )
    _max = jax.lax.cond(
        jnp.abs(max) > _inf_cutoff,
        lambda: jnp.sign(max) * jnp.inf,
        lambda: jnp.float_(max),
    )

    def _split_inf_integration():
        # Split the integration into two parts
        val1, info1 = quadgk(_func, [_min, 0.0], epsabs=abs_err, epsrel=rel_err)
        val2, info2 = quadgk(_func, [0.0, _max], epsabs=abs_err, epsrel=rel_err)
        status = info1.status | info2.status
        return val1 + val2, status

    def _base_integration():
        val, info = quadgk(_func, [_min, _max], epsabs=abs_err, epsrel=rel_err)
        return val, info.status

    val, status = jax.lax.cond(
        jnp.isinf(_min) & jnp.isinf(_max),
        _split_inf_integration,
        _base_integration,
    )

    return jax.lax.cond(
        status == 0,
        lambda: val,
        lambda: jnp.nan,
    )
