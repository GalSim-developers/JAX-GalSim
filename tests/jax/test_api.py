import inspect
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import pytest

import jax_galsim
import galsim as _galsim


def test_api_same():
    galsim_api = set(dir(_galsim))
    # we do not have the _galsim C++ layer in jax_galsim
    galsim_api.remove("_galsim")
    jax_galsim_api = set(dir(jax_galsim))
    # the jax_galsim.core module is specific to jax_galsim
    jax_galsim_api.remove("core")
    assert jax_galsim_api.issubset(galsim_api), (
        "jax_galsim API is not a subset of galsim API: %r" % (jax_galsim_api - galsim_api)
    )


OK_ERRORS = [
    "got an unexpected keyword argument",
    "At least one GSObject must be provided",
    "Single input argument must be a GSObject or list of them",
    "__init__() missing 1 required positional argument",
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
        return cls(2.0, **kwargs)
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
    elif kind == "vmap-jit-grad":
        # JAX tracing should be an identity
        assert cls.tree_unflatten(*((obj.tree_flatten())[::-1])) == obj

        # we can jit the object
        np.testing.assert_allclose(_xfun(0.3, obj), obj.xValue(x=0.3, y=-0.3))
        np.testing.assert_allclose(_kfun(0.3, obj), obj.kValue(kx=0.3, ky=-0.3).real)

        # check derivs
        eps = 1e-6
        grad = _xgradfun(0.3, obj)
        finite_diff = (obj.xValue(x=0.3 + eps, y=-0.3) - obj.xValue(x=0.3 - eps, y=-0.3)) / (
            2 * eps
        )
        np.testing.assert_allclose(grad, finite_diff)

        grad = _kgradfun(0.3, obj)
        finite_diff = (obj.kValue(kx=0.3 + eps, ky=-0.3).real - obj.kValue(kx=0.3 - eps, ky=-0.3).real) / (
            2 * eps
        )
        np.testing.assert_allclose(grad, finite_diff)

        # check vmap
        x = jnp.linspace(-1, 1, 10)
        np.testing.assert_allclose(_xfun_vmap(x, obj), [obj.xValue(x=_x, y=-0.3) for _x in x])
        np.testing.assert_allclose(_kfun_vmap(x, obj), [obj.kValue(kx=_x, ky=-0.3).real for _x in x])

        # check vmap grad
        np.testing.assert_allclose(
            _xgradfun_vmap(x, obj), [_xgradfun(_x, obj) for _x in x]
        )
        np.testing.assert_allclose(
            _kgradfun_vmap(x, obj), [_kgradfun(_x, obj) for _x in x]
        )
    elif kind == "docs-methods":
        # always has gsparams
        assert obj.gsparams is not None
        assert obj.gsparams == jax_galsim.GSParams.default

        # check docs
        gscls = getattr(_galsim, cls.__name__)
        assert all(line.strip() in cls.__doc__ for line in gscls.__doc__.splitlines() if line.strip())

        # check methods except the special JAX ones which should be exclusive to JAX
        for method in dir(cls):
            if not method.startswith("_"):
                if method not in ["params", "tree_flatten", "tree_unflatten"]:
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
                            if line.strip() and line not in _galsim.utilities.lazy_property.__doc__:
                                assert line.strip() in getattr(cls, method).__doc__
                else:
                    assert method not in dir(gscls), cls.__name__ + "." + method


@pytest.mark.parametrize("kind", [
    "docs-methods",
    "pickle-eval-repr",
    "vmap-jit-grad",
])
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
        for scale_type in ["fwhm", "sigma", "half_light_radius", "scale_radius"]:
            kwargs = {scale_type: 1.5}
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
