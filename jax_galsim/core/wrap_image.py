# from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def wrap_nonhermition(im, xmin, ymin, nxwrap, nywrap):
    def _body_j(j, vals):
        i, im = vals

        ii = (i - ymin) % nywrap + ymin
        jj = (j - xmin) % nxwrap + xmin

        im = jax.lax.cond(
            # weird way to say if ii != i and jj != j
            # I tried other ways and got test failures
            jnp.abs(ii - i) + jnp.abs(jj - j) != 0,
            lambda im, i, j, ii, jj: im.at[ii, jj].add(im[i, j]),
            lambda im, i, j, ii, jj: im,
            im,
            i,
            j,
            ii,
            jj,
        )

        return [i, im]

    def _body_i(i, vals):
        im = vals
        _, im = jax.lax.fori_loop(0, im.shape[1], _body_j, [i, im])
        return im

    im = jax.lax.fori_loop(0, im.shape[0], _body_i, im)
    return im
