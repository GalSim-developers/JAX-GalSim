from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("nxwrap", "nywrap"))
def _block_reduce_index(sim, nxwrap, nywrap):
    # this routine was written by Jean-Eric Campagne w/ edits and comments by Matthew Becker
    def rolling_window_i(arr, wind):
        idx = (
            jnp.arange(arr.shape[0] - wind + 1)[::wind, None]
            + jnp.arange(wind)[None, :]
        )
        return arr[idx]

    # the starting array shape is (Ny, Nx) with Nx = sim.shape[1] and Ny = sim.shape[0]
    # and below nx = Nx // nxwrap and ny = Ny // nywrap
    y = rolling_window_i(sim, nywrap)
    # now the array shape is (ny, nywrap, Nx)
    y = jnp.moveaxis(y, -1, -2)
    # now the array shape is (ny, Nx, nywrap)
    # this vampped function acts on the second to first axis since
    # the original function acts on the first axis and vmap adds an axis at the end
    y = jax.vmap(partial(rolling_window_i, wind=nxwrap))(y)
    # now the array shape is (ny, nx, nxwrap, nywrap)
    y = y.reshape(-1, nxwrap, nywrap)
    # now the array shape is (ny * nx, nxwrap, nywrap)
    y = jnp.moveaxis(y, -1, -2)
    # now the array shape is (ny * nx, nywrap, nxwrap)
    # then we sum on axis 0 to get the final shape (nywrap, nxwrap)
    return y.sum(axis=0)


@partial(jax.jit, static_argnames=("nx", "ny", "nxwrap", "nywrap"))
def _block_reduce_loop(sim, nx, ny, nxwrap, nywrap):
    # we do the reduction into another array and then set the final result in im at the end
    fim = jnp.zeros((nywrap, nxwrap), dtype=sim.dtype)

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

    return fim


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
    # the compile time can be a lot for big values of nbx * ny, so we branch here to a
    # index-based algorithm if needed
    if nx * ny > 50:
        fim = _block_reduce_index(sim, nxwrap, nywrap)
    else:
        fim = _block_reduce_loop(sim, nx, ny, nxwrap, nywrap)

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
