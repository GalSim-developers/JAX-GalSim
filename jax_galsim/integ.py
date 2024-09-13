import galsim as _galsim
import jax.lax
import jax.numpy as jnp
from quadax import quadgk

from jax_galsim.core.uitls import implements


@implements(
    _galsim.integ.int1d,
    lax_description=(
        "The JAX-GalSim package uses the adaptive Gauss-Kronrod-Patterson "
        "method implemented in the ``quadax`` package. That implementation "
        "is slightly different than the one in GalSim. Further, this function "
        "returns NaN on error instead of rasing an exception."
    ),
)
def int1d(func, min, max, rel_err=1.0e-6, abs_err=1.0e-12):
    val, info = quadgk(func, [min, max], epsabs=abs_err, epsrel=rel_err)

    return jax.lax.cond(
        info.status == 0,
        lambda: val,
        lambda: jnp.nan,
    )
