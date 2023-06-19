import os
import pytest
import yaml

# Define the accuracy for running the tests
from jax.config import config

config.update("jax_enable_x64", True)

# Identify the path to this current file
test_directory = os.path.dirname(os.path.abspath(__file__))


# Loading which tests to run
with open(os.path.join(test_directory, "galsim_tests_config.yaml"), "r") as f:
    test_config = yaml.safe_load(f)


def pytest_collection_modifyitems(config, items):
    """This hook will automatically skip tests that are not enabled in the
    enabled_tests.yaml file.
    """
    skip = pytest.mark.skip(reason="Skipping this because functionalities are not implemented yet")
    for item in items:
        # if this is a jax test we execute it
        if "jax" in item.nodeid:
            continue

        # if this is a galsim test we check if it is requested or not
        if not any([t in item.nodeid for t in test_config["enabled_tests"]]):
            item.add_marker(skip)


def pytest_pycollect_makemodule(module_path, path, parent):
    """This hook is tasked with overriding the galsim import
    at the top of each test file. Replaces it by jax-galsim.
    """
    # Load the module
    module = pytest.Module.from_parent(parent, path=module_path)
    # Overwrites the galsim module
    module.obj.galsim = __import__("jax_galsim")
    return module


def pytest_report_teststatus(report, config):
    """This hook will allow tests to be skipped if they
    fail for a reason authorized in the config file.
    """
    if report.when == "call" and report.outcome == "allowed failure":
        # It's an allowed failure, so let's mark it as such
        report.outcome = "allowed failure"
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
            report.wasxfail = True
            report.outcome = "allowed failure"

    yield report
