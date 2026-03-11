from functools import partial

import jax

import jax_galsim as jgs


@partial(jax.jit, static_argnames=("n_obj"))
def _generate_sim(rng_key, n_obj):
    pass
