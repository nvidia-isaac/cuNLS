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

"""The Python API documents itself: every public name, method and property has a docstring,
every constructor argument is explained, and the type stub lists every native name."""

import enum
import pathlib
import re

import pycunls
from pycunls import _pycunls_core as core

# A nanobind auto-docstring is only the signature, e.g. "(self) -> float" or "f(x: int) -> None".
_SIGNATURE = re.compile(r"^\s*(\w+)?\(.*\)\s*(->\s*.*)?$")


def _described(doc) -> bool:
    return any(not _SIGNATURE.match(line) for line in (doc or "").splitlines() if line.strip())


def _public_classes():
    for name in pycunls.__all__:
        obj = getattr(pycunls, name)
        if isinstance(obj, type) and not issubclass(obj, enum.Enum):
            yield name, obj


def _is_member(value) -> bool:
    return callable(value) or isinstance(value, property) or type(value).__name__ in (
        "nb_method", "nb_func", "property")


def test_every_public_name_has_a_docstring():
    missing = [n for n in pycunls.__all__ if not _described(getattr(pycunls, n).__doc__)]
    assert not missing, f"undocumented: {missing}"


def test_every_method_and_property_has_a_docstring():
    missing = []
    for name, cls in _public_classes():
        for attr, value in vars(cls).items():
            if attr.startswith("_") or not _is_member(value):
                continue
            if not _described(getattr(value, "__doc__", "")):
                missing.append(f"{name}.{attr}")
    assert not missing, f"undocumented: {missing}"


def test_every_constructor_argument_is_explained():
    missing = []
    for name, cls in _public_classes():
        init = vars(cls).get("__init__")
        if init is None or not init.__doc__:
            continue
        text = (cls.__doc__ or "") + (init.__doc__ if _described(init.__doc__) else "")
        for line in init.__doc__.splitlines():
            for arg in re.findall(r"(\w+):", line.split("->")[0]):
                if arg != "self" and arg not in text:
                    missing.append(f"{name}({arg})")
    assert not missing, f"constructor arguments not mentioned in the class docstring: {missing}"


def test_stub_lists_every_native_name():
    stub = (pathlib.Path(core.__file__).parent / "_pycunls_core.pyi")
    if not stub.exists():  # an installed wheel without the source stub
        stub = pathlib.Path(__file__).resolve().parents[1] / "pycunls" / "_pycunls_core.pyi"
    text = stub.read_text()
    names = [n for n in dir(core) if not n.startswith("_")]
    missing = [n for n in names if not re.search(rf"^(class|def) {n}\b", text, re.M)]
    assert not missing, (f"{stub.name} is stale (missing {missing}): run python "
                         "scripts/generate_pycunls_stubs.py")
