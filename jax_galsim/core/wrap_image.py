# from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def wrap_nonhermitian(im, xmin, ymin, nxwrap, nywrap):
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


@jax.jit
def expand_hermitian_x(im):
    return jnp.concatenate([im[:, 1:][::-1, ::-1].conjugate(), im], axis=1)


@jax.jit
def contract_hermitian_x(im):
    return im[:, im.shape[1] // 2 :]


@jax.jit
def wrap_hermitian_x(im, im_xmin, im_ymin, wrap_xmin, wrap_ymin, wrap_nx, wrap_ny):
    im_exp = expand_hermitian_x(im)
    im_exp = wrap_nonhermitian(
        im_exp, wrap_xmin - im_xmin, wrap_ymin - im_ymin, wrap_nx, wrap_ny
    )
    return contract_hermitian_x(im_exp)


@jax.jit
def expand_hermitian_y(im):
    return jnp.concatenate([im[1:, :][::-1, ::-1].conjugate(), im], axis=0)


@jax.jit
def contract_hermitian_y(im):
    return im[im.shape[0] // 2 :, :]


@jax.jit
def wrap_hermitian_y(im, im_xmin, im_ymin, wrap_xmin, wrap_ymin, wrap_nx, wrap_ny):
    im_exp = expand_hermitian_y(im)
    im_exp = wrap_nonhermitian(
        im_exp, wrap_xmin - im_xmin, wrap_ymin - im_ymin, wrap_nx, wrap_ny
    )
    return contract_hermitian_y(im_exp)
