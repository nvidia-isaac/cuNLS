import nvidia_sphinx_theme

project = "cuNLS"
author = "NVIDIA CORPORATION & AFFILIATES"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
html_show_sphinx = False

default_role = "code"
highlight_language = "c++"
primary_domain = "cpp"

autosectionlabel_prefix_document = True
