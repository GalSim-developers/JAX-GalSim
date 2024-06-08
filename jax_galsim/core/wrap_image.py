from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("xmin", "ymin", "nxwrap", "nywrap"))
def wrap_nonhermitian(im, xmin, ymin, nxwrap, nywrap):
    # these bits compute how many total blocks we need to cover the image
    nx = im.shape[1] // nxwrap
    if im.shape[1] % nxwrap:
        nx += 1

    ny = im.shape[0] // nywrap
    if im.shape[0] % nywrap:
        ny += 1

    # we have to pad to the correct size to get a whole number of blocks
    sim = jnp.pad(im, ((0, ny * nywrap - im.shape[0]), (0, nx * nxwrap - im.shape[1])))

    # then we roll the array so that the blocks start at the (0, 0) corner
    # this makes the indexing below super clean and fast
    sim = jnp.roll(sim, (-ymin, -xmin), axis=(0, 1))

    # we do the reduction into another array and then set the final result in im at the end
    fim = jnp.zeros((nywrap, nxwrap), dtype=im.dtype)

    # this set of loops will be explicitly unrolled by JAX when it compiles the function
    # via JIT
    # the number of iterations (i.e., blocks) should be small enough that this is OK
    yl = 0
    for _ in range(ny):
        yh = yl + nywrap

        xl = 0
        for _ in range(nx):
            xh = xl + nxwrap
            fim = fim + sim[yl:yh, xl:xh]
            xl = xh

        yl = yh

    im = im.at[ymin : ymin + nywrap, xmin : xmin + nxwrap].set(fim)
    return im


@jax.jit
def expand_hermitian_x(im):
    return jnp.concatenate([im[:, 1:][::-1, ::-1].conjugate(), im], axis=1)


@jax.jit
def contract_hermitian_x(im):
    return im[:, im.shape[1] // 2 :]


@partial(
    jax.jit,
    static_argnames=[
        "im_xmin",
        "im_ymin",
        "wrap_xmin",
        "wrap_ymin",
        "wrap_nx",
        "wrap_ny",
    ],
)
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


@partial(
    jax.jit,
    static_argnames=[
        "im_xmin",
        "im_ymin",
        "wrap_xmin",
        "wrap_ymin",
        "wrap_nx",
        "wrap_ny",
    ],
)
def wrap_hermitian_y(im, im_xmin, im_ymin, wrap_xmin, wrap_ymin, wrap_nx, wrap_ny):
    im_exp = expand_hermitian_y(im)
    im_exp = wrap_nonhermitian(
        im_exp, wrap_xmin - im_xmin, wrap_ymin - im_ymin, wrap_nx, wrap_ny
    )
    return contract_hermitian_y(im_exp)
