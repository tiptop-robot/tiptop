# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TiPToP"
copyright = "2025, TiPToP Contributors"
author = "TiPToP Contributors"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.video",
]

# MyST-Parser configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Logo configuration
html_favicon = "_static/logo-light.png"

html_title = "TiPToP"

# Furo theme options - Black & White styling
html_theme_options = {
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    # Light mode colors
    "light_css_variables": {
        "color-brand-primary": "#000000",
        "color-brand-content": "#000000",
        "color-admonition-background": "#f8f8f8",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f8f8f8",
        "color-foreground-primary": "#000000",
        "color-foreground-secondary": "#333333",
        "color-link": "#000000",
        "color-link--hover": "#555555",
    },
    # Dark mode colors
    "dark_css_variables": {
        "color-brand-primary": "#ffffff",
        "color-brand-content": "#ffffff",
        "color-admonition-background": "#1a1a1a",
        "color-background-primary": "#0d0d0d",
        "color-background-secondary": "#1a1a1a",
        "color-foreground-primary": "#ffffff",
        "color-foreground-secondary": "#cccccc",
        "color-link": "#ffffff",
        "color-link--hover": "#aaaaaa",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/NishanthJKumar/TiPToP-robot",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Intersphinx mapping for cross-referencing external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
