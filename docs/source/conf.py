project = "TARDIS"
copyright = "2023, Julius von Rohrscheidt and Bastian Rieck"
author = "Julius von Rohrscheidt and Bastian Rieck"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
]

# Ensure that member functions are documented. These are sane defaults.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

templates_path = ["_templates"]
exclude_patterns = []

# Tries to assign some semantic meaning to arguments provided with
# single backtics, such as `x`. This way, we can ignore `func` and
# `class` targets etc. (They still work, though!)
default_role = "obj"

html_theme = "alabaster"
html_static_path = ["_static"]

# Ensures that modules are sorted correctly. Since they all pertain to
# the same package, the prefix itself can be ignored.
modindex_common_prefix = ["tardis."]
