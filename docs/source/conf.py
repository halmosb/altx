"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "altx"
copyright = "2026, Marcell Tamás Kurbucz"
author = "Marcell Tamás Kurbucz"

try:
    from setuptools_scm import get_version as _get_scm_version

    release = _get_scm_version(root="../..", relative_to=__file__)
except Exception:
    try:
        release = _pkg_version("altx")
    except PackageNotFoundError:
        release = "0.0.0"

version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

myst_enable_extensions = ["dollarmath"]


templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Numpy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 2,
    "show_toc_level": 2,
    "navbar_align": "left",
}

# Strip >>> and ... prompts when the copy button is clicked
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
