import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim as jgs


@pytest.mark.parametrize(
    "params,gsobj,args",
    [
        (["scale_radius", "half_light_radius"], jgs.Spergel, [1.0]),
        (["scale_radius", "half_light_radius"], jgs.Exponential, []),
        (["sigma", "fwhm", "half_light_radius"], jgs.Gaussian, []),
        (["scale_radius", "half_light_radius", "fwhm"], jgs.Moffat, [2.0]),
    ],
)
def test_deriv_params_gsobject(params, gsobj, args):
    val = 2.0
    eps = 1e-5

    for param in params:
        print("\nparam:", param, flush=True)

        def _run(val_):
            kwargs = {param: val_}
            return jnp.max(
                gsobj(
                    *args,
                    **kwargs,
                    gsparams=jgs.GSParams(minimum_fft_size=64, maximum_fft_size=64),
                )
                .drawImage(nx=48, ny=48, scale=0.2, method="fft")
                .array[24, 24]
                ** 2
            )

        gfunc = jax.jit(jax.grad(_run))
        gval = gfunc(val)

        gfdiff = (_run(val + eps) - _run(val - eps)) / 2.0 / eps

        np.testing.assert_allclose(gval, gfdiff, rtol=0, atol=1e-6)


def test_deriv_params_moffat_with_trunc():
    val = 2.0
    eps = 1e-5

    def _run(val_):
        return jnp.max(
            jgs.Moffat(
                2.5,
                half_light_radius=val_,
                trunc=20.0,
                gsparams=jgs.GSParams(minimum_fft_size=64, maximum_fft_size=64),
            )
            .drawImage(nx=48, ny=48, scale=0.2)
            .array[24, 24]
            ** 2
        )

    gfunc = jax.jit(jax.grad(_run))
    with jax.disable_jit(), jax.debug_nans():
        gval = gfunc(val)

    gfdiff = (_run(val + eps) - _run(val - eps)) / 2.0 / eps

    np.testing.assert_allclose(gval, gfdiff, rtol=0, atol=1e-6)


def test_deriv_params_moffat_with_respect_to_trunc():
    val = 20.0
    eps = 1e-5

    def _run(val_):
        return jnp.max(
            jgs.Moffat(
                2.5,
                half_light_radius=2.0,
                trunc=val_,
                gsparams=jgs.GSParams(minimum_fft_size=64, maximum_fft_size=64),
            )
            .drawImage(nx=48, ny=48, scale=0.2)
            .array[24, 24]
            ** 2
        )

    gfunc = jax.jit(jax.grad(_run))
    with jax.disable_jit(), jax.debug_nans():
        gval = gfunc(val)

    gfdiff = (_run(val + eps) - _run(val - eps)) / 2.0 / eps

    np.testing.assert_allclose(gval, gfdiff, rtol=0, atol=1e-6)
