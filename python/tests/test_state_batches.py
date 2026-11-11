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

Covers VectorStateBatch{1,2,3,6} and the Lie-group state batches (SE3, SO3, SO2, SE2,
Similarity2, Similarity3):
  - Construction from Warp arrays with the expected ambient/tangent dimensions.
  - Device pointer arithmetic (stride matches ambient_size * sizeof(float)).
  - The ``const_state_ids`` constructor overload for holding states constant.
"""

import numpy as np
import pytest
import warp as wp

import pycunls

# (class, ambient size, tangent size)
STATES = [
    (pycunls.VectorStateBatch1, 1, 1),
    (pycunls.VectorStateBatch2, 2, 2),
    (pycunls.VectorStateBatch3, 3, 3),
    (pycunls.VectorStateBatch6, 6, 6),
    (pycunls.SE3StateBatch, 16, 6),  # 4x4 row-major
    (pycunls.SO3StateBatch, 9, 3),
    (pycunls.SO2StateBatch, 4, 1),
    (pycunls.SE2StateBatch, 9, 3),
    (pycunls.Similarity2StateBatch, 9, 4),
    (pycunls.Similarity3StateBatch, 16, 7),
]


@pytest.mark.parametrize("cls, ambient, tangent", STATES, ids=[c.__name__ for c, *_ in STATES])
def test_sizes_and_pointer_stride(cls, ambient, tangent):
    num = 4
    sb = cls(wp.zeros(num * ambient, dtype=wp.float32), num)
    assert sb.capacity == num
    assert sb.num_active_states == 0  # zero until set
    sb.set_num_active_states(num)
    assert sb.num_active_states == num
    assert (sb.ambient_size, sb.tangent_size) == (ambient, tangent)
    p0, p1 = sb.state_device_ptr(0), sb.state_device_ptr(1)
    assert p0 != 0
    assert p1 - p0 == ambient * 4  # ambient floats * 4 bytes


def test_vector_state_device_ptr_points_into_the_array():
    data = wp.array(np.arange(9, dtype=np.float32))
    sb = pycunls.VectorStateBatch3(data, 3)
    sb.set_num_active_states(sb.capacity, sb.const_capacity)
    assert sb.state_device_ptr(0) == data.ptr


def test_vector_const_ids():
    data = wp.zeros(30, dtype=wp.float32)
    const_ids = wp.array([0, 2], dtype=wp.int32)
    sb = pycunls.VectorStateBatch3(data, 10, const_ids, 2)
    assert (sb.capacity, sb.const_capacity) == (10, 2)
    sb.set_num_active_states(10, 2)
    assert sb.num_active_states == 10
    with pytest.raises(ValueError):
        sb.set_num_active_states(10, 3)


def test_se3_const_ids():
    data = wp.zeros(3 * 16, dtype=wp.float32)
    const_ids = wp.array([0], dtype=wp.int32)
    sb = pycunls.SE3StateBatch(data, 3, const_ids, 1)
    assert sb.num_const_states == 0
    sb.set_num_active_states(sb.capacity, sb.const_capacity)
    assert sb.num_active_states == 3
    assert sb.num_const_states == 1
    sb.set_num_active_states(2)
    assert sb.num_const_states == 0
