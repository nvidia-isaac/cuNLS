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

"""Tests for state batch bindings.

Covers VectorStateBatch{1,2,3,6} and all Lie-group state batches (SE3, SO3,
SO2, SE2, Similarity2, Similarity3).  Each test class verifies:
  - Construction from CuPy arrays with expected ambient/tangent dimensions.
  - Device pointer arithmetic (stride matches ambient_size * sizeof(float)).
  - The ``const_state_ids`` constructor overload for holding states constant.
"""

import cupy as cp
import numpy as np
import pytest

import pycunls


class TestVectorStateBatches:
    """Euclidean state batches where Plus is element-wise addition."""
    @pytest.mark.parametrize("cls,dim", [
        (pycunls.VectorStateBatch1, 1),
        (pycunls.VectorStateBatch2, 2),
        (pycunls.VectorStateBatch3, 3),
        (pycunls.VectorStateBatch6, 6),
    ])
    def test_basic_properties(self, cls, dim):
        num = 10
        data = cp.zeros(num * dim, dtype=cp.float32)
        sb = cls(data, num)
        assert sb.capacity == num
        assert sb.num_active_states == 0  # zero until set
        sb.set_num_active_states(num)
        assert sb.num_active_states == num
        assert sb.tangent_size == dim
        assert sb.ambient_size == dim

    def test_state_device_ptr(self):
        data = cp.arange(9, dtype=cp.float32)
        sb = pycunls.VectorStateBatch3(data, 3)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        ptr0 = sb.state_device_ptr(0)
        ptr1 = sb.state_device_ptr(1)
        assert ptr0 != 0
        assert ptr1 != ptr0
        assert ptr1 - ptr0 == 3 * 4  # 3 floats * 4 bytes

    def test_with_const_ids(self):
        data = cp.zeros(30, dtype=cp.float32)
        const_ids = cp.array([0, 2], dtype=cp.int32)
        sb = pycunls.VectorStateBatch3(data, 10, const_ids, 2)
        assert (sb.capacity, sb.const_capacity) == (10, 2)
        sb.set_num_active_states(10, 2)
        assert sb.num_active_states == 10
        with pytest.raises(ValueError):
            sb.set_num_active_states(10, 3)


class TestSE3StateBatch:
    """SE(3) poses stored as 4x4 row-major matrices (ambient=16, tangent=6)."""
    def test_basic(self):
        num = 5
        data = cp.zeros(num * 16, dtype=cp.float32)
        sb = pycunls.SE3StateBatch(data, num)
        assert sb.num_active_states == 0  # zero until set
        sb.set_num_active_states(num)
        assert sb.num_active_states == num
        assert sb.tangent_size == 6
        assert sb.ambient_size == 16

    def test_with_const_ids(self):
        data = cp.zeros(3 * 16, dtype=cp.float32)
        const_ids = cp.array([0], dtype=cp.int32)
        sb = pycunls.SE3StateBatch(data, 3, const_ids, 1)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        assert sb.num_active_states == 3

    def test_pointer_stride(self):
        num = 4
        data = cp.zeros(num * 16, dtype=cp.float32)
        sb = pycunls.SE3StateBatch(data, num)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        p0 = sb.state_device_ptr(0)
        p1 = sb.state_device_ptr(1)
        assert p1 - p0 == 16 * 4


class TestSO3StateBatch:
    def test_basic(self):
        data = cp.zeros(3 * 9, dtype=cp.float32)
        sb = pycunls.SO3StateBatch(data, 3)
        assert sb.tangent_size == 3
        assert sb.ambient_size == 9


class TestSO2StateBatch:
    def test_basic(self):
        data = cp.zeros(3 * 4, dtype=cp.float32)
        sb = pycunls.SO2StateBatch(data, 3)
        assert sb.tangent_size == 1
        assert sb.ambient_size == 4


class TestSE2StateBatch:
    def test_basic(self):
        data = cp.zeros(3 * 9, dtype=cp.float32)
        sb = pycunls.SE2StateBatch(data, 3)
        assert sb.tangent_size == 3
        assert sb.ambient_size == 9


class TestSimilarity2StateBatch:
    def test_basic(self):
        data = cp.zeros(3 * 9, dtype=cp.float32)
        sb = pycunls.Similarity2StateBatch(data, 3)
        assert sb.tangent_size == 4
        assert sb.ambient_size == 9


class TestSimilarity3StateBatch:
    def test_basic(self):
        data = cp.zeros(3 * 16, dtype=cp.float32)
        sb = pycunls.Similarity3StateBatch(data, 3)
        assert sb.tangent_size == 7
        assert sb.ambient_size == 16
