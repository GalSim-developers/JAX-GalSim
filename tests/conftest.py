# Define the accuracy for running the tests
import jax

jax.config.update("jax_enable_x64", True)

import inspect  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from functools import lru_cache, partial  # noqa: E402
from unittest.mock import patch  # noqa: E402

import galsim  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

import jax_galsim  # noqa: E402

# Identify the path to this current file
test_directory = os.path.dirname(os.path.abspath(__file__))

# Loading which tests to run
with open(os.path.join(test_directory, "galsim_tests_config.yaml"), "r") as f:
    test_config = yaml.safe_load(f)

# we need to patch the galsim utilities check_pickle function
# to use jax_galsim. it has an import inside a function so
# we patch sys.modules.
# see https://stackoverflow.com/questions/34213088/mocking-a-module-imported-inside-of-a-function
orig_check_pickle = galsim.utilities.check_pickle
orig_check_pickle.__globals__["BaseDeviate"] = jax_galsim.BaseDeviate


def _check_pickle(*args, **kwargs):
    with patch.dict(sys.modules, {"galsim": jax_galsim}):
        return orig_check_pickle(*args, **kwargs)


galsim.utilities.check_pickle = _check_pickle


def pytest_ignore_collect(collection_path, path, config):
    """This hook will skip collecting tests that are not
    enabled in the enabled_tests.yaml file.

    These somtimes fail to import and cause pytest to fail.
    """
    if "tests/GalSim/tests" in str(collection_path):
        if (
            not any(
                [
                    t in str(collection_path)
                    for t in test_config["enabled_tests"]["galsim"]
                ]
            )
        ) and "*" not in test_config["enabled_tests"]["galsim"]:
            return True

    if "tests/Coord/tests" in str(collection_path):
        if (
            not any(
                [
                    t in str(collection_path)
                    for t in test_config["enabled_tests"]["coord"]
                ]
            )
        ) and "*" not in test_config["enabled_tests"]["coord"]:
            return True


def pytest_collection_modifyitems(config, items):
    """This hook will automatically skip tests that are not enabled in the
    enabled_tests.yaml file.
    """
    skip = pytest.mark.skip(
        reason="Skipping this because functionalities are not implemented yet"
    )
    for item in items:
        # if this is a jax test we execute it
        if "jax" in item.nodeid:
            continue

        # if this is a galsim test we check if it is requested or not
        if (
            (
                not any(
                    [t in item.nodeid for t in test_config["enabled_tests"]["galsim"]]
                )
            )
            and "*" not in test_config["enabled_tests"]["galsim"]
        ) and (
            (not any([t in item.nodeid for t in test_config["enabled_tests"]["coord"]]))
            and "*" not in test_config["enabled_tests"]["coord"]
        ):
            item.add_marker(skip)


@lru_cache(maxsize=128)
def _infile(val, fname):
    with open(fname, "r") as f:
        for line in f:
            if val in line:
                return True

    return False


def _convert_galsim_to_jax_galsim(obj):
    import galsim as _galsim  # noqa: F401
    from numpy import array  # noqa: F401

    import jax_galsim as galsim  # noqa: F401

    if isinstance(obj, _galsim.GSObject):
        ret_obj = eval(repr(obj))
        return ret_obj
    else:
        return obj


def pytest_pycollect_makemodule(module_path, path, parent):
    """This hook is tasked with overriding the galsim import
    at the top of each test file. Replaces it by jax-galsim.
    """
    # Load the module
    module = pytest.Module.from_parent(parent, path=module_path)

    # Overwrites the galsim module
    module.obj.galsim = __import__("jax_galsim")
    if hasattr(module.obj, "coord"):
        module.obj.coord = __import__("jax_galsim")
    if hasattr(module.obj, "radians"):
        module.obj.radians = __import__("jax_galsim").radians
        module.obj.degrees = __import__("jax_galsim").degrees
        module.obj.hours = __import__("jax_galsim").hours
        module.obj.arcmin = __import__("jax_galsim").arcmin
        module.obj.arcsec = __import__("jax_galsim").arcsec

    # ensure we can run on numpy functions when testing integration in galsim
    if str(module_path).endswith("tests/GalSim/tests/test_integ.py"):
        module.obj.galsim.integ.int1d = partial(
            jax_galsim.integ.int1d, _wrap_as_callback=True
        )
        # make things easier for us, is 7 in galsim
        module.obj.test_decimal = 4

    if str(module_path).endswith(
        "tests/GalSim/tests/test_interpolatedimage.py"
    ) and hasattr(module.obj, "setup"):
        module.obj.setup()

    # Overwrites galsim in the galsim_test_helpers module
    for k, v in module.obj.__dict__.items():
        if (
            callable(v)
            and hasattr(v, "__globals__")
            and (
                inspect.getsourcefile(v).endswith("galsim_test_helpers.py")
                or inspect.getsourcefile(v).endswith("galsim/utilities.py")
            )
            and _infile("def " + k, inspect.getsourcefile(v))
            and "galsim" in v.__globals__
        ):
            v.__globals__["galsim"] = __import__("jax_galsim")

        if k == "default_params" and isinstance(v, __import__("galsim").GSParams):
            module.obj.default_params = __import__("jax_galsim").GSParams.from_galsim(v)

    # override coord in its helper_util.py
    for k, v in module.obj.__dict__.items():
        if (
            callable(v)
            and hasattr(v, "__globals__")
            and inspect.getsourcefile(v).endswith("helper_util.py")
            and _infile("def " + k, inspect.getsourcefile(v))
        ):
            v.__globals__["coord"] = __import__("jax_galsim")
            v.__globals__["galsim"] = __import__("jax_galsim")

    # the galsim WCS tests have some items that are galsim objects that need conversions
    # to jax_galsim objects
    if str(module_path).endswith("tests/GalSim/tests/test_wcs.py"):
        for k, v in module.obj.__dict__.items():
            if isinstance(v, __import__("galsim").GSObject):
                module.obj.__dict__[k] = _convert_galsim_to_jax_galsim(v)
            elif isinstance(v, list):
                module.obj.__dict__[k] = [
                    _convert_galsim_to_jax_galsim(obj) for obj in v
                ]

        module.obj._convert_galsim_to_jax_galsim = _convert_galsim_to_jax_galsim

    return module


def pytest_report_teststatus(report, config):
    """This hook will allow tests to be skipped if they
    fail for a reason authorized in the config file.
    """
    if report.when == "call" and report.outcome == "allowed failure":
        return report.outcome, "-", "ALLOWED FAILURE"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_logreport(report):
    """Alters the report to allow not-implemented tests
    to fail
    """
    if report.when == "call" and report.failed:
        # Ok, so we have a failure, let's see if it a failure we expect
        message = report.longrepr.reprcrash.message
        if any([t in message for t in test_config["allowed_failures"]]):
            report.wasxfail = "allowed failure - %s" % (" ".join(message.splitlines()))
            report.outcome = "allowed failure"

    yield report
