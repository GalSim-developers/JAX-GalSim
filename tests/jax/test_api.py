import inspect
import os
import pickle

import galsim as _galsim
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim


def test_api_same():
    galsim_api = set(dir(_galsim))
    # we do not have the _galsim C++ layer in jax_galsim
    galsim_api.remove("_galsim")
    jax_galsim_api = set(dir(jax_galsim))
    # the jax_galsim.core module is specific to jax_galsim
    jax_galsim_api.remove("core")
    assert jax_galsim_api.issubset(
        galsim_api
    ), "jax_galsim API is not a subset of galsim API: %r" % (
        jax_galsim_api - galsim_api
    )


OK_ERRORS = [
    "got an unexpected keyword argument",
    "At least one GSObject must be provided",
    "Single input argument must be a GSObject or list of them",
    "__init__() missing 1 required positional argument",
    "__init__() missing 2 required positional arguments",
    "Arguments to Convolution must be GSObjects",
    "Convolution constructor got unexpected keyword argument(s)",
    "Either scale_radius or half_light_radius must be specified",
    "One of sigma, fwhm, and half_light_radius must be specified",
    "One of scale_radius, half_light_radius, or fwhm must be specified",
    "One of scale_radius, half_light_radius must be specified"
    "Arguments to Sum must be GSObjects",
    "'ArrayImpl' object has no attribute 'gsparams'",
    "Supplied image must be an Image or file name",
    "Argument to Deconvolution must be a GSObject.",
]


def _attempt_init(cls, kwargs):
    try:
        return cls(**kwargs)
    except Exception as e:
        if any(estr in repr(e) for estr in OK_ERRORS):
            pass
        else:
            raise e

    try:
        return cls(jnp.array(2.0), **kwargs)
    except Exception as e:
        if any(estr in repr(e) for estr in OK_ERRORS):
            pass
        else:
            raise e

    try:
        return cls(jnp.array(2.0), jnp.array(4.0), **kwargs)
    except Exception as e:
        if any(estr in repr(e) for estr in OK_ERRORS):
            pass
        else:
            raise e

    if cls in [jax_galsim.Convolution, jax_galsim.Deconvolution]:
        try:
            return cls(jax_galsim.Gaussian(**kwargs))
        except Exception as e:
            if any(estr in repr(e) for estr in OK_ERRORS):
                pass
            else:
                raise e

    if cls in [jax_galsim.InterpolatedImage]:
        try:
            return cls(
                jax_galsim.ImageD(jnp.arange(100).reshape((10, 10))),
                scale=1.3,
                **kwargs
            )
        except Exception as e:
            if any(estr in repr(e) for estr in OK_ERRORS):
                pass
            else:
                raise e

    return None


@jax.jit
def _xfun(x, prof):
    return prof.xValue(x=x, y=-0.3)


@jax.jit
def _kfun(x, prof):
    return prof.kValue(kx=x, ky=-0.3).real


_xgradfun = jax.jit(jax.grad(_xfun))
_kgradfun = jax.jit(jax.grad(_kfun))

_xfun_vmap = jax.jit(jax.vmap(_xfun, in_axes=(0, None)))
_kfun_vmap = jax.jit(jax.vmap(_kfun, in_axes=(0, None)))

_xgradfun_vmap = jax.jit(jax.vmap(_xgradfun, in_axes=(0, None)))
_kgradfun_vmap = jax.jit(jax.vmap(_kgradfun, in_axes=(0, None)))


def _run_object_checks(obj, cls, kind):
    if kind == "pickle-eval-repr":
        from numpy import array  # noqa: F401

        # eval repr is identity mapping
        assert eval(repr(obj)) == obj

        # pickle is identity mapping
        assert pickle.loads(pickle.dumps(obj)) == obj

        # check that we can hash the object
        hash(obj)
    elif kind == "pickle-eval-repr-img" or kind == "pickle-eval-repr-nohash":
        from numpy import array  # noqa: F401

        # eval repr is identity mapping
        assert eval(repr(obj)) == obj

        # pickle is identity mapping
        assert pickle.loads(pickle.dumps(obj)) == obj

        # check that we cannot hash the object
        assert obj.__hash__ is None
    elif kind == "pickle-eval-repr-wcs":
        import jax_galsim as galsim  # noqa: F401

        # eval repr is identity mapping
        assert eval(repr(obj)) == obj

        # pickle is identity mapping
        assert pickle.loads(pickle.dumps(obj)) == obj

        # check that we cannot hash the object
        hash(obj)
    elif kind == "jax-compatible":
        # JAX tracing should be an identity
        assert cls.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj
    elif kind == "vmap-jit-grad":
        # JAX tracing should be an identity
        assert cls.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

        eps = 1e-6
        x = jnp.linspace(-1, 1, 10)

        if cls not in [jax_galsim.Convolution, jax_galsim.Deconvolution]:
            # we can jit the object
            np.testing.assert_allclose(_xfun(0.3, obj), obj.xValue(x=0.3, y=-0.3))

            # check derivs
            grad = _xgradfun(0.3, obj)
            finite_diff = (
                obj.xValue(x=0.3 + eps, y=-0.3) - obj.xValue(x=0.3 - eps, y=-0.3)
            ) / (2 * eps)
            np.testing.assert_allclose(grad, finite_diff)

            # check vmap
            np.testing.assert_allclose(
                _xfun_vmap(x, obj), [obj.xValue(x=_x, y=-0.3) for _x in x]
            )
            # check vmap grad
            np.testing.assert_allclose(
                _xgradfun_vmap(x, obj), [_xgradfun(_x, obj) for _x in x]
            )

        np.testing.assert_allclose(_kfun(0.3, obj), obj.kValue(kx=0.3, ky=-0.3).real)

        grad = _kgradfun(0.3, obj)
        finite_diff = (
            obj.kValue(kx=0.3 + eps, ky=-0.3).real
            - obj.kValue(kx=0.3 - eps, ky=-0.3).real
        ) / (2 * eps)
        np.testing.assert_allclose(grad, finite_diff)

        np.testing.assert_allclose(
            _kfun_vmap(x, obj), [obj.kValue(kx=_x, ky=-0.3).real for _x in x]
        )
        np.testing.assert_allclose(
            _kgradfun_vmap(x, obj), [_kgradfun(_x, obj) for _x in x]
        )
    elif kind == "vmap-jit-grad-wcs":
        assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

        def _reg_fun(x):
            return obj.toWorld(jax_galsim.PositionD(x, -0.3)).x

        _fun = jax.jit(_reg_fun)
        _gradfun = jax.jit(jax.grad(_fun))
        _fun_vmap = jax.jit(jax.vmap(_fun))
        _gradfun_vmap = jax.jit(jax.vmap(_gradfun))

        # we can jit the object
        np.testing.assert_allclose(_fun(0.3), _reg_fun(0.3))

        # check derivs
        eps = 1e-6
        grad = _gradfun(0.3)
        finite_diff = (_reg_fun(0.3 + eps) - _reg_fun(0.3 - eps)) / (2 * eps)
        np.testing.assert_allclose(grad, finite_diff)

        # check vmap
        x = jnp.linspace(-0.9, 0.9, 10)
        np.testing.assert_allclose(_fun_vmap(x), [_reg_fun(_x) for _x in x])

        # check vmap grad
        np.testing.assert_allclose(_gradfun_vmap(x), [_gradfun(_x) for _x in x])
    elif kind == "vmap-jit-grad-celestialwcs":
        assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

        def _reg_fun(x):
            return obj.toWorld(jax_galsim.PositionD(x, -0.3)).ra.rad

        _fun = jax.jit(_reg_fun)
        _gradfun = jax.jit(jax.grad(_fun))
        _fun_vmap = jax.jit(jax.vmap(_fun))
        _gradfun_vmap = jax.jit(jax.vmap(_gradfun))

        # we can jit the object
        np.testing.assert_allclose(_fun(0.3), _reg_fun(0.3))

        # check derivs
        eps = 1e-2
        grad = _gradfun(0.3)
        finite_diff = (_reg_fun(0.3 + eps) - _reg_fun(0.3 - eps)) / (2 * eps)
        np.testing.assert_allclose(grad, finite_diff)

        # check vmap
        x = jnp.linspace(-0.9, 0.9, 10)
        np.testing.assert_allclose(_fun_vmap(x), [_reg_fun(_x) for _x in x])

        # check vmap grad
        np.testing.assert_allclose(_gradfun_vmap(x), [_gradfun(_x) for _x in x])

        # go the other way
        def _reg_fun(x):
            return obj.toImage(
                jax_galsim.CelestialCoord(
                    x * jax_galsim.degrees, -56.51006288339 * jax_galsim.degrees
                )
            ).x

        _fun = jax.jit(_reg_fun)
        _gradfun = jax.jit(jax.grad(_fun))
        _fun_vmap = jax.jit(jax.vmap(_fun))
        _gradfun_vmap = jax.jit(jax.vmap(_gradfun))

        # we can jit the object
        farg = 66.03
        np.testing.assert_allclose(_fun(farg), _reg_fun(farg))

        # check derivs
        eps = 1e-5
        grad = _gradfun(farg)
        finite_diff = (_reg_fun(farg + eps) - _reg_fun(farg - eps)) / (2 * eps)
        np.testing.assert_allclose(grad, finite_diff)

        # check vmap
        x = jnp.linspace(-0.9, 0.9, 10) + farg
        np.testing.assert_allclose(_fun_vmap(x), [_reg_fun(_x) for _x in x])

        # check vmap grad
        np.testing.assert_allclose(_gradfun_vmap(x), [_gradfun(_x) for _x in x])
    elif kind == "vmap-jit-grad-random":
        assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

        for key in obj._params:
            if key in ["N", "n"]:
                continue

            if key == "p":
                cen = 0.6
                x = jnp.linspace(0.1, 0.9, 10)
            else:
                cen = 2.0
                x = jnp.arange(10) + 2.0

            if key == "k":
                rtol = 2e-2
            else:
                rtol = 1e-7

            def _reg_fun(p):
                kwargs = {key: p}
                arr = jnp.zeros(100)
                return jnp.sum(cls(seed=10, **kwargs).generate(arr).astype(float))

            _fun = jax.jit(_reg_fun)
            _gradfun = jax.jit(jax.grad(_fun))
            _fun_vmap = jax.jit(jax.vmap(_fun))
            _gradfun_vmap = jax.jit(jax.vmap(_gradfun))

            # we can jit the object
            np.testing.assert_allclose(_fun(cen), _reg_fun(cen))

            # check derivs
            eps = 1e-6
            grad = _gradfun(cen)
            finite_diff = (_reg_fun(cen + eps) - _reg_fun(cen - eps)) / (2 * eps)
            np.testing.assert_allclose(grad, finite_diff, rtol=rtol)

            # check vmap
            np.testing.assert_allclose(_fun_vmap(x), [_reg_fun(_x) for _x in x])

            # check vmap grad
            np.testing.assert_allclose(
                _gradfun_vmap(x), [_gradfun(_x) for _x in x], rtol=rtol
            )
    elif kind == "docs-methods":
        # always has gsparams
        if isinstance(obj, jax_galsim.GSObject):
            assert obj.gsparams is not None
            assert obj.gsparams == jax_galsim.GSParams.default

        # check docs
        gscls = getattr(_galsim, cls.__name__)
        assert all(
            line.strip() in cls.__doc__
            for line in gscls.__doc__.splitlines()
            if line.strip()
        )

        # check methods except the special JAX ones which should be exclusive to JAX
        for method in dir(cls):
            if not method.startswith("_"):
                if method not in [
                    "params",
                    "tree_flatten",
                    "tree_unflatten",
                    "from_galsim",
                ]:
                    # this deprecated method doesn't have consistent doc strings in galsim
                    if (
                        issubclass(cls, jax_galsim.wcs.BaseWCS)
                        and method == "withOrigin"
                    ):
                        continue

                    assert method in dir(gscls), (
                        cls.__name__ + "." + method + " not in galsim." + gscls.__name__
                    )

                    # check doc strings
                    if getattr(gscls, method).__doc__ is not None:
                        assert getattr(cls, method).__doc__ is not None, (
                            cls.__name__ + "." + method + " does not have a doc string"
                        )
                        for line in getattr(gscls, method).__doc__.splitlines():
                            # we skip the lazy_property decorator doc string since this is not always
                            # used in jax_galsim
                            if (
                                line.strip()
                                and line not in _galsim.utilities.lazy_property.__doc__
                            ):
                                assert line.strip() in getattr(cls, method).__doc__, (
                                    cls.__name__
                                    + "."
                                    + method
                                    + " doc string does not match galsim."
                                    + gscls.__name__
                                    + "."
                                    + method
                                )
                else:
                    assert method not in dir(gscls), cls.__name__ + "." + method
    else:
        raise RuntimeError("Unknown test: %r" % kind)


@pytest.mark.parametrize(
    "kind",
    [
        "docs-methods",
        "pickle-eval-repr",
        "vmap-jit-grad",
    ],
)
def test_api_gsobject(kind):
    jax_galsim_api = set(dir(jax_galsim))
    classes = []
    for api in sorted(jax_galsim_api):
        if not api.startswith("__"):
            _attr = getattr(jax_galsim, api)
            if inspect.isclass(_attr) and issubclass(_attr, jax_galsim.GSObject):
                classes.append(_attr)
    cls_tested = set()
    for cls in classes:        
        for scale_type in [
            None,
            "fwhm",
            "sigma",
            "half_light_radius",
            "scale_radius",
            "flux",
        ]:
            if scale_type is not None:
                kwargs = {scale_type: jnp.array(1.5)}
            else:
                kwargs = {}
            obj = _attempt_init(cls, kwargs)

            if obj is not None and obj.__class__ is not jax_galsim.GSObject:
                cls_tested.add(cls.__name__)
                print(obj)

                _run_object_checks(obj, cls, kind)

                if cls.__name__ == "Gaussian":
                    _obj = obj + obj
                    print(_obj)
                    _run_object_checks(_obj, _obj.__class__, kind)

                    _obj = 2.0 * obj
                    print(_obj)
                    _run_object_checks(_obj, _obj.__class__, kind)

                    _obj = obj.shear(g1=0.1, g2=0.2)
                    print(_obj)
                    _run_object_checks(_obj, _obj.__class__, kind)

    assert "Exponential" in cls_tested
    assert "Gaussian" in cls_tested
    assert "Moffat" in cls_tested
    assert "Spergel" in cls_tested
    assert "Box" in cls_tested
    assert "Pixel" in cls_tested
    assert "InterpolatedImage" in cls_tested


@pytest.mark.parametrize(
    "obj",
    [
        jax_galsim.Shear(g1=jnp.array(0.1), g2=jnp.array(0.2)),
        jax_galsim.Shear(e1=jnp.array(0.1), e2=jnp.array(0.2)),
        jax_galsim.Shear(eta1=jnp.array(0.1), eta2=jnp.array(0.2)),
        jax_galsim.Shear(jnp.array(0.1) + 1j * jnp.array(0.2)),
    ],
)
def test_api_shear(obj):
    _run_object_checks(obj, jax_galsim.Shear, "docs-methods")
    _run_object_checks(obj, jax_galsim.Shear, "pickle-eval-repr")

    def _reg_sfun(g1):
        return (jax_galsim.Shear(g1=g1, g2=0.2) + jax_galsim.Shear(g1=g1, g2=-0.1)).eta1

    _sfun = jax.jit(_reg_sfun)

    _sgradfun = jax.jit(jax.grad(_sfun))
    _sfun_vmap = jax.jit(jax.vmap(_sfun))
    _sgradfun_vmap = jax.jit(jax.vmap(_sgradfun))

    # JAX tracing should be an identity
    assert jax_galsim.Shear.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

    # we can jit the object
    np.testing.assert_allclose(_sfun(0.3), _reg_sfun(0.3))

    # check derivs
    eps = 1e-6
    grad = _sgradfun(0.3)
    finite_diff = (_reg_sfun(0.3 + eps) - _reg_sfun(0.3 - eps)) / (2 * eps)
    np.testing.assert_allclose(grad, finite_diff)

    # check vmap
    x = jnp.linspace(-0.9, 0.9, 10)
    np.testing.assert_allclose(_sfun_vmap(x), [_reg_sfun(_x) for _x in x])

    # check vmap grad
    np.testing.assert_allclose(_sgradfun_vmap(x), [_sgradfun(_x) for _x in x])


@pytest.mark.parametrize(
    "obj",
    [
        jax_galsim.BoundsD(),
        jax_galsim.BoundsI(),
        jax_galsim.BoundsD(
            jnp.array(0.2), jnp.array(4.0), jnp.array(-0.5), jnp.array(4.7)
        ),
        jax_galsim.BoundsI(jnp.array(-10), jnp.array(5), jnp.array(0), jnp.array(7)),
    ],
)
def test_api_bounds(obj):
    _run_object_checks(obj, obj.__class__, "docs-methods")
    _run_object_checks(obj, obj.__class__, "pickle-eval-repr")

    # JAX tracing should be an identity
    assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

    if isinstance(obj, jax_galsim.BoundsD):

        def _reg_sfun(g1):
            return (
                (
                    obj.__class__(g1, g1 + 0.5, 2 * g1, 2 * g1 + 0.5).expand(0.5)
                    + obj.__class__(-g1, -g1 + 0.5, -2 * g1, -2 * g1 + 0.5)
                )
                .expand(4)
                .area()
            )

        _sfun = jax.jit(_reg_sfun)

        _sgradfun = jax.jit(jax.grad(_sfun))
        _sfun_vmap = jax.jit(jax.vmap(_sfun))
        _sgradfun_vmap = jax.jit(jax.vmap(_sgradfun))

        # we can jit the object
        np.testing.assert_allclose(_sfun(0.3), _reg_sfun(0.3))

        # check derivs
        eps = 1e-6
        grad = _sgradfun(0.3)
        finite_diff = (_reg_sfun(0.3 + eps) - _reg_sfun(0.3 - eps)) / (2 * eps)
        np.testing.assert_allclose(grad, finite_diff)

        # check vmap
        x = jnp.linspace(-0.9, 0.9, 10)
        np.testing.assert_allclose(_sfun_vmap(x), [_reg_sfun(_x) for _x in x])

        # check vmap grad
        np.testing.assert_allclose(_sgradfun_vmap(x), [_sgradfun(_x) for _x in x])


@pytest.mark.parametrize(
    "obj",
    [
        jax_galsim.PositionD(),
        jax_galsim.PositionI(),
        jax_galsim.PositionD(jnp.array(0.1), jnp.array(-0.2)),
        jax_galsim.PositionD(x=jnp.array(0.1), y=jnp.array(-0.2)),
        jax_galsim.PositionI(jnp.array(1), jnp.array(-2)),
        jax_galsim.PositionI(x=jnp.array(1), y=jnp.array(-2)),
    ],
)
def test_api_position(obj):
    _run_object_checks(obj, obj.__class__, "docs-methods")
    _run_object_checks(obj, obj.__class__, "pickle-eval-repr")

    # JAX tracing should be an identity
    assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

    if isinstance(obj, jax_galsim.PositionD):

        def _reg_sfun(g1):
            return (
                (obj.__class__(g1, 0.5) + obj.__class__(-g1, -2))
                .shear(jax_galsim.Shear(g1=0.1, g2=0.2))
                .x
            )

        _sfun = jax.jit(_reg_sfun)

        _sgradfun = jax.jit(jax.grad(_sfun))
        _sfun_vmap = jax.jit(jax.vmap(_sfun))
        _sgradfun_vmap = jax.jit(jax.vmap(_sgradfun))

        # we can jit the object
        np.testing.assert_allclose(_sfun(0.3), _reg_sfun(0.3))

        # check derivs
        eps = 1e-6
        grad = _sgradfun(0.3)
        finite_diff = (_reg_sfun(0.3 + eps) - _reg_sfun(0.3 - eps)) / (2 * eps)
        np.testing.assert_allclose(grad, finite_diff)

        # check vmap
        x = jnp.linspace(-0.9, 0.9, 10)
        np.testing.assert_allclose(_sfun_vmap(x), [_reg_sfun(_x) for _x in x])

        # check vmap grad
        np.testing.assert_allclose(_sgradfun_vmap(x), [_sgradfun(_x) for _x in x])


@pytest.mark.parametrize(
    "obj",
    [
        jax_galsim.ImageD(jnp.ones((10, 10))),
        jax_galsim.ImageD(jnp.ones((10, 10)), scale=jnp.array(0.5)),
    ],
)
def test_api_image(obj):
    _run_object_checks(obj, obj.__class__, "docs-methods")
    _run_object_checks(obj, obj.__class__, "pickle-eval-repr-img")

    # JAX tracing should be an identity
    assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

    def _reg_sfun(g1):
        return (obj / g1)(2, 2)

    _sfun = jax.jit(_reg_sfun)

    _sgradfun = jax.jit(jax.grad(_sfun))
    _sfun_vmap = jax.jit(jax.vmap(_sfun))
    _sgradfun_vmap = jax.jit(jax.vmap(_sgradfun))

    # we can jit the object
    np.testing.assert_allclose(_sfun(0.3), _reg_sfun(0.3))

    # check derivs
    eps = 1e-6
    grad = _sgradfun(0.3)
    finite_diff = (_reg_sfun(0.3 + eps) - _reg_sfun(0.3 - eps)) / (2 * eps)
    np.testing.assert_allclose(grad, finite_diff)

    # check vmap
    x = jnp.linspace(-0.9, 0.9, 10)
    np.testing.assert_allclose(_sfun_vmap(x), [_reg_sfun(_x) for _x in x])

    # check vmap grad
    np.testing.assert_allclose(_sgradfun_vmap(x), [_sgradfun(_x) for _x in x])


OK_ERRORS_WCS = [
    "__init__() missing 3 required positional arguments",
    "__init__() missing 2 required positional arguments",
    "__init__() missing 1 required positional argument",
    "origin must be a PositionD or PositionI argument",
    "__init__() takes from 2 to 4 positional arguments but 5 were given",
    "__init__() takes 2 positional arguments but 3 were given",
    "__init__() takes 2 positional arguments but 5 were given",
    "__init__() takes 3 positional arguments but 5 were given",
    "'ArrayImpl' object has no attribute 'lower'",
    "expected str, bytes or os.PathLike object, not",
    "__init__() got an unexpected keyword argument 'dir'",
]


def _attempt_init_wcs(cls):
    obj = None

    try:
        obj = cls(jnp.array(0.4))
    except Exception as e:
        if any(estr in repr(e) for estr in OK_ERRORS_WCS):
            pass
        else:
            raise e

    try:
        obj = cls(
            jnp.array(0.4), jax_galsim.Shear(g1=jnp.array(0.1), g2=jnp.array(0.2))
        )
    except Exception as e:
        if any(estr in repr(e) for estr in OK_ERRORS_WCS):
            pass
        else:
            raise e

    try:
        obj = cls(jnp.array(0.45), jnp.array(-0.02), jnp.array(0.04), jnp.array(-0.35))
    except Exception as e:
        if any(estr in repr(e) for estr in OK_ERRORS_WCS):
            pass
        else:
            raise e

    try:
        dr = os.path.join(
            os.path.dirname(__file__), "..", "GalSim", "tests", "des_data"
        )
        file_name = "DECam_00158414_01.fits.fz"
        obj = cls(file_name, dir=dr)
    except Exception as e:
        if any(estr in repr(e) for estr in OK_ERRORS_WCS):
            pass
        else:
            raise e

    return obj


def test_api_wcs():
    classes = []
    for item in sorted(dir(jax_galsim.wcs)):
        cls = getattr(jax_galsim.wcs, item)
        if (
            inspect.isclass(cls)
            and issubclass(cls, jax_galsim.wcs.BaseWCS)
            and cls
            not in (
                jax_galsim.wcs.BaseWCS,
                jax_galsim.wcs.EuclideanWCS,
                jax_galsim.wcs.LocalWCS,
                jax_galsim.wcs.UniformWCS,
                jax_galsim.wcs.CelestialWCS,
            )
        ):
            classes.append(getattr(jax_galsim.wcs, item))

    for item in sorted(dir(jax_galsim.fitswcs)):
        cls = getattr(jax_galsim.fitswcs, item)
        if (
            inspect.isclass(cls)
            and issubclass(cls, jax_galsim.wcs.BaseWCS)
            and cls
            not in (
                jax_galsim.wcs.BaseWCS,
                jax_galsim.wcs.EuclideanWCS,
                jax_galsim.wcs.LocalWCS,
                jax_galsim.wcs.UniformWCS,
                jax_galsim.wcs.CelestialWCS,
            )
        ):
            classes.append(getattr(jax_galsim.fitswcs, item))

    tested = set()
    for cls in classes:
        obj = _attempt_init_wcs(cls)
        if obj is not None:
            print(obj)
            tested.add(cls.__name__)
            _run_object_checks(obj, cls, "docs-methods")
            _run_object_checks(obj, cls, "pickle-eval-repr-wcs")
            if isinstance(obj, jax_galsim.wcs.CelestialWCS):
                _run_object_checks(obj, cls, "vmap-jit-grad-celestialwcs")
            else:
                _run_object_checks(obj, cls, "vmap-jit-grad-wcs")

    assert {
        "AffineTransform",
        "JacobianWCS",
        "ShearWCS",
        "PixelScale",
        "OffsetShearWCS",
        "OffsetWCS",
        "GSFitsWCS",
    } <= tested


def test_api_angleunit():
    obj = jax_galsim.AngleUnit(jnp.array(0.1))
    _run_object_checks(obj, obj.__class__, "docs-methods")
    _run_object_checks(obj, obj.__class__, "pickle-eval-repr")

    # JAX tracing should be an identity
    assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

    def _reg_sfun(g1):
        return jax_galsim.AngleUnit(g1) / jax_galsim.AngleUnit(0.05)

    _sfun = jax.jit(_reg_sfun)

    _sgradfun = jax.jit(jax.grad(_sfun))
    _sfun_vmap = jax.jit(jax.vmap(_sfun))
    _sgradfun_vmap = jax.jit(jax.vmap(_sgradfun))

    # we can jit the object
    np.testing.assert_allclose(_sfun(0.3), _reg_sfun(0.3))

    # check derivs
    eps = 1e-6
    grad = _sgradfun(0.3)
    finite_diff = (_reg_sfun(0.3 + eps) - _reg_sfun(0.3 - eps)) / (2 * eps)
    np.testing.assert_allclose(grad, finite_diff)

    # check vmap
    x = jnp.linspace(-0.9, 0.9, 10)
    np.testing.assert_allclose(_sfun_vmap(x), [_reg_sfun(_x) for _x in x])

    # check vmap grad
    np.testing.assert_allclose(_sgradfun_vmap(x), [_sgradfun(_x) for _x in x])


def test_api_angle():
    obj = jax_galsim.Angle(jnp.array(0.1) * jax_galsim.degrees)
    _run_object_checks(obj, obj.__class__, "docs-methods")
    _run_object_checks(obj, obj.__class__, "pickle-eval-repr")

    # JAX tracing should be an identity
    assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

    def _reg_sfun(g1):
        return (
            jax_galsim.Angle(g1 * jax_galsim.degrees)
            + jax_galsim.Angle(g1**2 * jax_galsim.degrees)
        ) / jax_galsim.degrees

    _sfun = jax.jit(_reg_sfun)

    _sgradfun = jax.jit(jax.grad(_sfun))
    _sfun_vmap = jax.jit(jax.vmap(_sfun))
    _sgradfun_vmap = jax.jit(jax.vmap(_sgradfun))

    # we can jit the object
    np.testing.assert_allclose(_sfun(0.3), _reg_sfun(0.3))

    # check derivs
    eps = 1e-6
    grad = _sgradfun(0.3)
    finite_diff = (_reg_sfun(0.3 + eps) - _reg_sfun(0.3 - eps)) / (2 * eps)
    np.testing.assert_allclose(grad, finite_diff)

    # check vmap
    x = jnp.linspace(-0.9, 0.9, 10)
    np.testing.assert_allclose(_sfun_vmap(x), [_reg_sfun(_x) for _x in x])

    # check vmap grad
    np.testing.assert_allclose(_sgradfun_vmap(x), [_sgradfun(_x) for _x in x])


def test_api_celestial_coord():
    obj = jax_galsim.CelestialCoord(45 * jax_galsim.degrees, -30 * jax_galsim.degrees)
    _run_object_checks(obj, obj.__class__, "docs-methods")
    _run_object_checks(obj, obj.__class__, "pickle-eval-repr")

    # JAX tracing should be an identity
    assert obj.__class__.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

    def _reg_sfun(g1):
        return obj.distanceTo(
            jax_galsim.CelestialCoord(g1 * jax_galsim.degrees, 20 * jax_galsim.degrees)
        ).rad

    _sfun = jax.jit(_reg_sfun)

    _sgradfun = jax.jit(jax.grad(_sfun))
    _sfun_vmap = jax.jit(jax.vmap(_sfun))
    _sgradfun_vmap = jax.jit(jax.vmap(_sgradfun))

    # we can jit the object
    np.testing.assert_allclose(_sfun(0.3), _reg_sfun(0.3))

    # check derivs
    eps = 1e-6
    grad = _sgradfun(0.3)
    finite_diff = (_reg_sfun(0.3 + eps) - _reg_sfun(0.3 - eps)) / (2 * eps)
    np.testing.assert_allclose(grad, finite_diff)

    # check vmap
    x = jnp.linspace(-0.9, 0.9, 10)
    np.testing.assert_allclose(_sfun_vmap(x), [_reg_sfun(_x) for _x in x])

    # check vmap grad
    np.testing.assert_allclose(_sgradfun_vmap(x), [_sgradfun(_x) for _x in x])


def test_api_random():
    classes = []
    for item in sorted(dir(jax_galsim.random)):
        cls = getattr(jax_galsim.random, item)
        if inspect.isclass(cls) and issubclass(cls, jax_galsim.random.BaseDeviate):
            classes.append(getattr(jax_galsim.random, item))

    tested = set()
    for cls in classes:
        obj = cls(seed=42)
        print(obj)
        tested.add(cls.__name__)
        _run_object_checks(obj, cls, "docs-methods")
        _run_object_checks(obj, cls, "pickle-eval-repr-img")
        _run_object_checks(obj, cls, "vmap-jit-grad-random")

    assert {
        "UniformDeviate",
        "GaussianDeviate",
        "BinomialDeviate",
        "PoissonDeviate",
        "WeibullDeviate",
        "GammaDeviate",
        "Chi2Deviate",
    } <= tested


def _init_noise(cls):
    try:
        obj = cls(jax_galsim.random.GaussianDeviate(seed=42))
    except Exception as e:
        if "__init__() missing 1 required positional argument: 'var_image'" in str(e):
            pass
        else:
            raise e
    else:
        return obj

    try:
        obj = cls(
            jax_galsim.random.GaussianDeviate(seed=42),
            jax_galsim.ImageD(jnp.ones((10, 10)) * 2.0),
        )
    except Exception as e:
        raise e
    else:
        return obj


def test_api_noise():
    classes = []
    for item in sorted(dir(jax_galsim.noise)):
        cls = getattr(jax_galsim.noise, item)
        if (
            inspect.isclass(cls)
            and issubclass(cls, jax_galsim.noise.BaseNoise)
            and cls is not jax_galsim.noise.BaseNoise
        ):
            classes.append(getattr(jax_galsim.noise, item))

    tested = set()
    for cls in classes:
        obj = _init_noise(cls)
        print(obj)
        tested.add(cls.__name__)
        _run_object_checks(obj, cls, "docs-methods")
        _run_object_checks(obj, cls, "pickle-eval-repr-img")
        # _run_object_checks(obj, cls, "vmap-jit-grad-random")

    assert {
        "GaussianNoise",
        "PoissonNoise",
        "DeviateNoise",
        "VariableGaussianNoise",
        "CCDNoise",
    } <= tested


@pytest.mark.parametrize(
    "obj1",
    [
        jax_galsim.Gaussian(fwhm=1.0),
        jax_galsim.Pixel(scale=1.0),
        jax_galsim.Exponential(scale_radius=1.0),
        jax_galsim.Exponential(half_light_radius=1.0),
        jax_galsim.Moffat(fwhm=1.0, beta=3),
        jax_galsim.Moffat(scale_radius=1.0, beta=3),
        jax_galsim.Spergel(nu=0.0,scale_radius=1.0),
        jax_galsim.Spergel(nu=0.0,half_light_radius=1.0),
        jax_galsim.Shear(g1=0.1, g2=0.2),
        jax_galsim.PositionD(x=0.1, y=0.2),
        jax_galsim.BoundsI(xmin=0, xmax=1, ymin=0, ymax=1),
        jax_galsim.BoundsD(xmin=0, xmax=1, ymin=0, ymax=1),
        jax_galsim.ShearWCS(0.2, jax_galsim.Shear(g1=0.1, g2=0.2)),
        jax_galsim.Delta(),
        jax_galsim.Nearest(),
        jax_galsim.Lanczos(3),
        jax_galsim.Lanczos(3, conserve_dc=False),
        jax_galsim.Quintic(),
        jax_galsim.Linear(),
        jax_galsim.Cubic(),
        jax_galsim.SincInterpolant(),
    ],
)
def test_api_pickling_eval_repr_basic(obj1):
    # test copied from galsim
    import copy
    import pickle
    from collections.abc import Hashable
    from numbers import Complex, Integral, Real  # noqa: F401

    # In case the repr uses these:
    from numpy import (  # noqa: F401
        array,
        complex64,
        complex128,
        float32,
        float64,
        int16,
        int32,
        ndarray,
        uint16,
        uint32,
    )

    def func(x):
        return x

    print("Try pickling ", str(obj1))

    # print('pickled obj1 = ',pickle.dumps(obj1))
    obj2 = pickle.loads(pickle.dumps(obj1))
    assert obj2 is not obj1
    # print('obj1 = ',repr(obj1))
    # print('obj2 = ',repr(obj2))
    f1 = func(obj1)
    f2 = func(obj2)
    # print('func(obj1) = ',repr(f1))
    # print('func(obj2) = ',repr(f2))
    assert f1 == f2

    # Check that == works properly if the other thing isn't the same type.
    assert f1 != object()
    assert object() != f1

    # Test the hash values are equal for two equivalent objects.
    if isinstance(obj1, Hashable):
        # print('hash = ',hash(obj1),hash(obj2))
        assert hash(obj1) == hash(obj2)

    obj3 = copy.copy(obj1)
    assert obj3 is not obj1
    random = hasattr(obj1, "rng") or "rng" in repr(obj1)
    if not random:  # Things with an rng attribute won't be identical on copy.
        f3 = func(obj3)
        assert f3 == f1

    obj4 = copy.deepcopy(obj1)
    assert obj4 is not obj1
    f4 = func(obj4)
    if random:
        f1 = func(obj1)
    # print('func(obj1) = ',repr(f1))
    # print('func(obj4) = ',repr(f4))
    assert f4 == f1  # But everything should be identical with deepcopy.

    # Also test that the repr is an accurate representation of the object.
    # The gold standard is that eval(repr(obj)) == obj.  So check that here as well.
    # A few objects we don't expect to work this way in GalSim; when testing these, we set the
    # `irreprable` kwarg to true.  Also, we skip anything with random deviates since these don't
    # respect the eval/repr roundtrip.

    if not random:
        # A further complication is that the default numpy print options do not lead to sufficient
        # precision for the eval string to exactly reproduce the original object, and start
        # truncating the output for relatively small size arrays.  So we temporarily bump up the
        # precision and truncation threshold for testing.
        # print(repr(obj1))
        with _galsim.utilities.printoptions(precision=20, threshold=np.inf):
            obj5 = eval(repr(obj1))
        # print('obj1 = ',repr(obj1))
        # print('obj5 = ',repr(obj5))
        f5 = func(obj5)
        # print('f1 = ',f1)
        # print('f5 = ',f5)
        assert f5 == f1, "func(obj1) = %r\nfunc(obj5) = %r" % (f1, f5)
    else:
        # Even if we're not actually doing the test, still make the repr to check for syntax errors.
        repr(obj1)


def test_api_photon_array():
    pa = jax_galsim.PhotonArray(101)

    _run_object_checks(pa, pa.__class__, "docs-methods")
    _run_object_checks(pa, pa.__class__, "pickle-eval-repr-nohash")
    _run_object_checks(pa, pa.__class__, "jax-compatible")


def test_api_sensor():
    classes = []
    for item in sorted(dir(jax_galsim)):
        cls = getattr(jax_galsim, item)
        if inspect.isclass(cls) and issubclass(cls, jax_galsim.sensor.Sensor):
            classes.append(getattr(jax_galsim.sensor, item))

    tested = set()
    for cls in classes:
        obj = cls()
        print(obj)
        tested.add(cls.__name__)
        _run_object_checks(obj, obj.__class__, "docs-methods")
        _run_object_checks(obj, obj.__class__, "pickle-eval-repr")
        _run_object_checks(obj, obj.__class__, "jax-compatible")

    assert {
        "Sensor",
    } <= tested
