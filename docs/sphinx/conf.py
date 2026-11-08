# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvidia_sphinx_theme  # noqa: F401  (fails early if the theme is not installed)

project = "cuNLS"
author = "NVIDIA CORPORATION & AFFILIATES"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "myst_parser",  # the agent guide is Markdown, shipped in the wheel
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
html_show_sphinx = False
html_favicon = "_static/cuNLS_icon.svg"

default_role = "code"
highlight_language = "c++"
primary_domain = "cpp"

autosectionlabel_prefix_document = True


# ---------------------------------------------------------------------------
# llms.txt: an index of the documentation for LLM agents (https://llmstxt.org),
# written into the HTML output at build time from the page titles, so it cannot
# drift from the docs.
# ---------------------------------------------------------------------------


def _write_llms_txt(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    import pathlib

    env = app.builder.env
    head = [
        "# cuNLS",
        "",
        "> GPU-accelerated nonlinear least squares (CUDA/C++, Python package pycunls): "
        "model fitting, pose graphs, bundle adjustment, visual-inertial estimation, RANSAC, "
        "constrained optimization and MPC.",
        "",
        "Start with the agent guide (also `python -m pycunls guide`). Check an install with "
        "`python -m pycunls doctor`.",
        "",
        "## Pages",
        "",
    ]
    pages = []
    for doc in sorted(env.found_docs):
        title = env.titles.get(doc)
        text = title.astext() if title is not None else doc
        pages.append(f"- [{text}]({app.builder.get_target_uri(doc)})")
    (pathlib.Path(app.outdir) / "llms.txt").write_text("\n".join(head + pages) + "\n")


def setup(app):
    app.connect("build-finished", _write_llms_txt)
