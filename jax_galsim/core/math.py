import jax
import jax.numpy as jnp


@jax.jit
def safe_sqrt(x):
    """Numerically safe sqrt operation with zero derivative as zero."""
    msk = x > 0
    x_msk = jnp.where(msk, x, 1.0)
    return jnp.where(msk, jnp.sqrt(x_msk), 0.0)
