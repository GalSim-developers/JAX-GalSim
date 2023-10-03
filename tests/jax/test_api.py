import inspect
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
    "Arguments to Sum must be GSObjects",
    "'ArrayImpl' object has no attribute 'gsparams'",
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
        # eval repr is identity mapping
        assert eval(repr(obj)) == obj

        # pickle is identity mapping
        assert pickle.loads(pickle.dumps(obj)) == obj

        # check that we can hash the object
        hash(obj)
    elif kind == "pickle-eval-repr-img":
        from numpy import array  # noqa: F401

        # eval repr is identity mapping
        assert eval(repr(obj)) == obj

        # pickle is identity mapping
        assert pickle.loads(pickle.dumps(obj)) == obj

        # check that we cannot hash the object
        assert obj.__hash__ is None
    elif kind == "vmap-jit-grad":
        # JAX tracing should be an identity
        assert cls.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

        # we can jit the object
        np.testing.assert_allclose(_xfun(0.3, obj), obj.xValue(x=0.3, y=-0.3))
        np.testing.assert_allclose(_kfun(0.3, obj), obj.kValue(kx=0.3, ky=-0.3).real)

        # check derivs
        eps = 1e-6
        grad = _xgradfun(0.3, obj)
        finite_diff = (
            obj.xValue(x=0.3 + eps, y=-0.3) - obj.xValue(x=0.3 - eps, y=-0.3)
        ) / (2 * eps)
        np.testing.assert_allclose(grad, finite_diff)

        grad = _kgradfun(0.3, obj)
        finite_diff = (
            obj.kValue(kx=0.3 + eps, ky=-0.3).real
            - obj.kValue(kx=0.3 - eps, ky=-0.3).real
        ) / (2 * eps)
        np.testing.assert_allclose(grad, finite_diff)

        # check vmap
        x = jnp.linspace(-1, 1, 10)
        np.testing.assert_allclose(
            _xfun_vmap(x, obj), [obj.xValue(x=_x, y=-0.3) for _x in x]
        )
        np.testing.assert_allclose(
            _kfun_vmap(x, obj), [obj.kValue(kx=_x, ky=-0.3).real for _x in x]
        )

        # check vmap grad
        np.testing.assert_allclose(
            _xgradfun_vmap(x, obj), [_xgradfun(_x, obj) for _x in x]
        )
        np.testing.assert_allclose(
            _kgradfun_vmap(x, obj), [_kgradfun(_x, obj) for _x in x]
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
                                assert line.strip() in getattr(cls, method).__doc__
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
    assert "Box" in cls_tested
    assert "Pixel" in cls_tested


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
        jax_galsim.ImageD(jnp.ones((10, 10)), scale=0.5),
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


def test_api_wcs():
    assert False