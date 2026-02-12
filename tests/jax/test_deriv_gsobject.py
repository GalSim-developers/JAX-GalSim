import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim as jgs


@pytest.mark.parametrize(
    "gsobj,params,args, kwargs",
    [
        (jgs.Spergel, ["scale_radius", "half_light_radius"], [1.0], {}),
        (jgs.Exponential, ["scale_radius", "half_light_radius"], [], {}),
        (jgs.Gaussian, ["sigma", "fwhm", "half_light_radius"], [], {}),
        (jgs.Moffat, ["scale_radius", "half_light_radius", "fwhm"], [2.0], {}),
        (
            jgs.Moffat,
            ["scale_radius", "half_light_radius", "fwhm"],
            [2.0],
            {"trunc": 20.0},
        ),
    ],
)
def test_deriv_gsobject_radii(params, gsobj, args, kwargs):
    val = 2.0
    eps = 1e-5

    for param in params:
        print("\nparam:", param, flush=True)

        def _run(val_):
            kwargs_ = {param: val_}
            kwargs_.update(kwargs)
            return jnp.max(
                gsobj(
                    *args,
                    **kwargs_,
                    gsparams=jgs.GSParams(minimum_fft_size=8, maximum_fft_size=8),
                )
                .drawImage(nx=5, ny=5, scale=0.2, method="fft")
                .array[2, 2]
                ** 2
            )

        gfunc = jax.jit(jax.grad(_run))
        gval = gfunc(val)

        gfdiff = (_run(val + eps) - _run(val - eps)) / 2.0 / eps

        atol = 1e-5

        np.testing.assert_allclose(gval, gfdiff, rtol=0, atol=atol)


def test_deriv_gsobject_moffat_trunc():
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
            .drawImage(nx=48, ny=48, scale=0.2, method="fft")
            .array[24, 24]
            ** 2
        )

    gfunc = jax.jit(jax.grad(_run))
    gval = gfunc(val)

    gfdiff = (_run(val + eps) - _run(val - eps)) / 2.0 / eps

    np.testing.assert_allclose(gval, gfdiff, rtol=0, atol=1e-6)
