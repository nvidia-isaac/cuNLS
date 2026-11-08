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

"""The catalog matches the classes: every built-in batch is listed, and the listed sizes are the
sizes the classes report."""

import cupy as cp
import pytest

import pycunls
from pycunls import _pycunls_core as core
from pycunls.catalog import factor_info, state_info

_BASES = {"FactorBatch", "StateBatch", "CustomFactorBatch", "CustomStateBatch",
          "ConstraintFactorBatchBase"}


def _exported(base):
    return [n for n in pycunls.__all__
            if isinstance(getattr(pycunls, n), type) and issubclass(getattr(pycunls, n), base)
            and n not in _BASES]


def test_every_exported_batch_is_in_the_catalog():
    names = {e["name"] for e in pycunls.catalog()}
    missing = [n for n in _exported(core.FactorBatch) + _exported(core.StateBatch)
               if n not in names]
    assert not missing, missing


def test_every_catalog_entry_exists():
    for e in pycunls.catalog():
        assert hasattr(pycunls, e["name"]), e["name"]


@pytest.mark.parametrize("name", _exported(core.StateBatch))
def test_state_sizes_match(name):
    info = state_info(name)
    batch = getattr(pycunls, name)(cp.zeros(4 * info.ambient, cp.float32), 4)
    assert (batch.ambient_size, batch.tangent_size) == (info.ambient, info.tangent)


def _simple_factors():
    """Factors whose constructor is (one or more arrays..., capacity)."""
    out = []
    for e in pycunls.catalog("factor"):
        if not e["states"] or not e["arrays"]:
            continue
        args = list(e["arrays"])
        if any(a in ("imu_samples", "time_steps", "obstacles", "poses_camera_from_rig",
                     "lower") for a in args):
            continue
        out.append(e["name"])
    return out


@pytest.mark.parametrize("name", _simple_factors())
def test_factor_sizes_match(name):
    info = factor_info(name)
    arrays = [cp.zeros(64 * 4, cp.float32) for a in info.arrays
              if a != "poses_camera_from_rig"]
    fb = getattr(pycunls, name)(*arrays, 4)
    assert fb.residuals_size == info.residual_size
    tangents = [state_info(s).tangent for s in info.states]
    assert list(fb.state_sizes()) == tangents


def test_describe_explains_the_between_convention():
    text = pycunls.describe(pycunls.SE3BetweenFactorBatch)
    assert "T_right^-1 * T_left" in text and "16 floats" in text


def test_describe_suggests_close_names():
    with pytest.raises(KeyError, match="Did you mean.*SE3BetweenFactorBatch"):
        pycunls.describe("SE3Between")


def test_query_filters():
    names = [e["name"] for e in pycunls.catalog("factor", "between")]
    assert "SE3BetweenFactorBatch" in names and "SE3PriorFactorBatch" not in names
