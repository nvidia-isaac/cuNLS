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

"""The docs agree with the code: the files they cite exist, the pycunls names they use exist,
and every public pycunls name is documented somewhere. Needs the source tree (the CI test step
mounts it); skipped for an installed wheel alone."""

import importlib
import pathlib
import re

import pytest

import pycunls

REPO = pathlib.Path(__file__).resolve().parents[2]
if not (REPO / "docs" / "sphinx").is_dir():
    pytest.skip("source tree not available", allow_module_level=True)


def _doc_files():
    files = [REPO / "README.md", REPO / "AGENTS.md", REPO / "CONTRIBUTING.md",
             REPO / "examples" / "README.md"]
    files += sorted(REPO.glob("cunls/**/AGENTS.md"))
    files += sorted(REPO.glob("python/**/*.md"))
    files += sorted(REPO.glob("docs/sphinx/**/*.rst"))
    # Skip build outputs inside the checkout (not parent directories of the checkout).
    return [f for f in files if f.exists()
            and not {"_build", "build"} & set(f.relative_to(REPO).parts)]


# A file path in backticks: at least one slash and a file extension (prose such as `factor/state`
# and Sphinx :doc: targets have none); ./paths are generated outputs.
_PATH = re.compile(r"`{1,2}([\w\-][\w.\-]*(?:/[\w.\-]+)+\.[A-Za-z]{1,4})`{1,2}")


def test_cited_files_exist():
    missing = []
    for f in _doc_files():
        for path in _PATH.findall(f.read_text()):
            roots = (REPO, f.parent, REPO / "cunls", REPO / "python",
                     REPO / "python" / "examples")
            if not any((root / path).exists() for root in roots):
                missing.append(f"{f.relative_to(REPO)}: {path}")
    assert not missing, "cited files that do not exist:\n" + "\n".join(missing)


def test_named_pycunls_api_exists():
    missing = set()
    for f in _doc_files():
        for name in re.findall(r"\bpycunls\.([A-Za-z_]\w*)", f.read_text()):
            if name == "X" or hasattr(pycunls, name):  # X: the placeholder in help(pycunls.X)
                continue
            try:
                importlib.import_module(f"pycunls.{name}")
            except ImportError:
                missing.add(f"{f.relative_to(REPO)}: pycunls.{name}")
    assert not missing, "docs name API that does not exist:\n" + "\n".join(sorted(missing))


def test_every_public_name_is_documented():
    text = " ".join(f.read_text() for f in _doc_files()
                    if f.suffix == ".rst" or f.name == "AGENT_GUIDE.md")
    missing = []
    for name in pycunls.__all__:
        if re.search(rf"\b{name}\b", text):
            continue
        # Families are documented once, e.g. BoundFactorBatchN or BoundFactorBatch{1,2,...}.
        stem = re.sub(r"\d+$", "", name)
        if stem != name and re.search(rf"\b{stem}(N|\{{|\d)", text):
            continue
        missing.append(name)
    assert not missing, f"public names no docs page mentions: {missing}"
