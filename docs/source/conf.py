# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EddyFlux'
copyright = '2025, RajarshiB'
author = 'RajarshiB'
release = '2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'myst_parser',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode'
]

# Optional: If you're using .md or .rst alongside
source_suffix = ['.rst', '.md', '.ipynb']  # <-- This is enough

# Optional: set notebook prolog/epilog if needed
nbsphinx_execute = 'never'  # disables execution during build
# html_theme = 'alabaster' or 'sphinx_rtd_theme'


html_theme = 'sphinx_rtd_theme'  # or any other



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'  # or 'sphinx_rtd_theme' if installed

html_static_path = ['_static']


