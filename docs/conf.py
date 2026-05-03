# Configuration file for the Sphinx documentation builder.
#
# For the full list of configuration options, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable without installing it.
sys.path.insert(0, os.path.abspath(".."))
# Make the custom extension importable.
sys.path.insert(0, os.path.abspath("_ext"))

###########################################
# custom code to update the API coverage
from update_api_coverage import update_api_coverage

update_api_coverage()
###########################################

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "JAX-GalSim"
author = "GalSim Developers"
copyright = "2026, GalSim Developers"

try:
    from jax_galsim._version import version as release
except ImportError:
    release = "0.0.1.dev0"

version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

# Extension load order matters:
#   1. sphinx.ext.autodoc   – must be first; it defines autodoc-process-docstring
#   2. galsim_docstring     – our handler runs before Napoleon sees the lines
#   3. sphinx.ext.napoleon  – converts the cleaned-up Parameters: block to RST
extensions = [
    "sphinx.ext.autodoc",  # API docs from docstrings (defines the event)
    "galsim_docstring",  # custom – splits implements() docstrings
    "sphinx.ext.napoleon",  # Google/NumPy-style docstring parsing
    "sphinx.ext.viewcode",  # "View source" links
    "sphinx.ext.intersphinx",  # cross-links to external docs
    "sphinx_design",  # dropdown / collapsible directives
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# Napoleon (docstring parsing)
# ---------------------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autoclass_content = "class"  # use only the class docstring (not __init__)

# Packages that are imported by jax_galsim but may not be present at
# documentation build time.
autodoc_mock_imports = []

# ---------------------------------------------------------------------------
# Intersphinx mappings
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "galsim": ("https://galsim-developers.github.io/GalSim/_build/html", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_title = f"{project} {version}"
