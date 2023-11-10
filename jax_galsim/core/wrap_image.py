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


# I am leaving this code here for posterity. It has a bug that I cannot find.
# It tries to be more clever instead of simply expanding the hermitian image to
# it's full shape, wrapping everything, and then contracting. -MRB
# @jax.jit
# def wrap_hermitian_x(im, im_xmin, im_ymin, wrap_xmin, wrap_ymin, wrap_nx, wrap_ny):
#     def _body_j(j, vals):
#         i, im = vals

#         # first do zero or positive x freq
#         im_y = i + im_ymin
#         im_x = j + im_xmin
#         wrap_y = (im_y - wrap_ymin) % wrap_ny + wrap_ymin
#         wrap_x = (im_x - wrap_xmin) % wrap_nx + wrap_xmin
#         wrap_yind = wrap_y - im_ymin
#         wrap_xind = wrap_x - im_xmin
#         im = jax.lax.cond(
#             wrap_xind >= 0,
#             lambda wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind: jax.lax.cond(
#                 jnp.abs(wrap_x - im_x) + jnp.abs(wrap_y - im_y) != 0,
#                 lambda im, wrap_yind, wrap_xind: im.at[wrap_yind, wrap_xind].add(im[i, j]),
#                 lambda im, wrap_yind, wrap_xind: im,
#                 im,
#                 wrap_yind,
#                 wrap_xind,
#             ),
#             lambda wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind: im,
#             wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind,
#         )

#         # now do neg x freq
#         im_y = -im_y
#         im_x = -im_x
#         wrap_y = (im_y - wrap_ymin) % wrap_ny + wrap_ymin
#         wrap_x = (im_x - wrap_xmin) % wrap_nx + wrap_xmin
#         wrap_yind = wrap_y - im_ymin
#         wrap_xind = wrap_x - im_xmin
#         im = jax.lax.cond(
#             im_x != 0,
#             lambda wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind: jax.lax.cond(
#                 wrap_xind >= 0,
#                 lambda wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind: jax.lax.cond(
#                     (jnp.abs(wrap_x - im_x) + jnp.abs(wrap_y - im_y)) != 0,
#                     lambda im, wrap_yind, wrap_xind: im.at[wrap_yind, wrap_xind].add(im[i, j].conjugate()),
#                     lambda im, wrap_yind, wrap_xind: im,
#                     im,
#                     wrap_yind,
#                     wrap_xind,
#                 ),
#                 lambda wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind: im,
#                 wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind,
#             ),
#             lambda wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind: im,
#             wrap_x, im_x, wrap_y, im_y, im, wrap_yind, wrap_xind,
#         )

#         return [i, im]

#     def _body_i(i, vals):
#         im = vals
#         _, im = jax.lax.fori_loop(0, im.shape[1], _body_j, [i, im])
#         return im

#     im = jax.lax.fori_loop(0, im.shape[0], _body_i, im)
#     return im
