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

"""Array arguments are checked where the mistake is made, with the fix in the message."""

import numpy as np
import pytest
import warp as wp

import pycunls


def test_host_array_is_rejected_with_the_fix():
    with pytest.raises(TypeError,
                       match=r"host \(NumPy\) array.*"
                             r"wp\.array\(data, dtype=wp\.float32, device=\"cuda\"\)"):
        pycunls.VectorStateBatch3(np.zeros(9, np.float32), 3)


def test_wrong_dtype_names_the_expected_one():
    with pytest.raises(TypeError, match=r"must have dtype float32, got float64.*"
                                        r"wp\.array\(data, dtype=wp\.float32\)"):
        pycunls.VectorStateBatch3(wp.zeros(9, wp.float64), 3)


def test_strided_view_is_rejected():
    with pytest.raises(ValueError,
                       match=r"C-contiguous.*strided view.*wp\.clone\(wp\.array\(data\)\)"):
        pycunls.VectorStateBatch3(wp.zeros((3, 6), wp.float32)[:, ::2], 3)


def test_non_array_is_rejected():
    with pytest.raises(TypeError, match="must be a GPU array"):
        pycunls.VectorStateBatch3([1.0, 2.0, 3.0], 1)


def test_valid_arrays_and_raw_pointers_still_work():
    data = wp.zeros((3, 4, 4), wp.float32)
    pycunls.SE3StateBatch(data, 3)
    pycunls.SE3StateBatch(data.ptr, 3)  # an int pointer is not checked
    pycunls.VectorStateBatch3(wp.zeros(9, wp.float64).ptr, 3)  # not even its dtype


class _Vec2(pycunls.CustomStateBatch):
    def plus(self, *args):
        pass


@pytest.mark.parametrize("make, needle", [
    (lambda: pycunls.SE3StateBatch(wp.zeros((3, 12), wp.float32), 3),
     r"has 36 float32 values, but 48 are needed.*16 float32"),
    (lambda: pycunls.VectorStateBatch2(wp.zeros(8, wp.float32), 4, wp.zeros(1, wp.int32), 2),
     "const_ids has 1 int32 values, but 2 are needed"),
    (lambda: _Vec2(wp.zeros(6, wp.float32), 2, 2, 4),
     "data has 6 float32 values, but 8 are needed"),
    (lambda: _Vec2(wp.zeros(8, wp.float32), 2, 2, 4, wp.zeros(1, wp.int32), 2),
     "const_ids has 1 int32 values, but 2 are needed"),
    (lambda: pycunls.VectorStateBatch3(wp.zeros(9, wp.float32), 3).set_bounds(
        wp.zeros(9, wp.float32), wp.zeros(6, wp.float32)),
     "upper has 6 float32 values, but 9 are needed"),
    (lambda: pycunls.SE3BetweenFactorBatch(wp.zeros((2, 4, 4), wp.float32), 3),
     "deltas has 32 float32 values, but 48 are needed"),
    (lambda: pycunls.PnPFactorBatch(wp.zeros((4, 2), wp.float32), wp.zeros((2, 4, 4), wp.float32),
                                    wp.zeros((4, 3), wp.float32), 4),  # one rig pose per factor
     "poses_camera_from_rig has 32 float32 values, but 64"),
    (lambda: pycunls.ConstantVelocitySE3FactorBatch(wp.zeros(3, wp.float32), 4),
     "time_steps has 3 float32 values, but 4 are needed"),
    (lambda: pycunls.ConstantVelocityInformationSE3FactorBatch(
        pycunls.CudaStream(), wp.zeros(4, wp.float32), wp.zeros(3, wp.float32), 4),
     "qc_diag has 3 float32 values, but 6 are needed"),
    (lambda: pycunls.BoundFactorBatch3(wp.zeros(6, wp.float32), wp.zeros(9, wp.float32), 3),
     "lower has 6 float32 values, but 9 are needed"),
    (lambda: pycunls.HalfspaceFactorBatch2(wp.zeros(4, wp.float32), wp.zeros(3, wp.float32), 3),
     "normals has 4 float32 values, but 6 are needed"),
    (lambda: pycunls.WeightedFactorBatch(
        pycunls.PriorVectorFactorBatch3(wp.zeros(9, wp.float32), 3),
        weights=wp.zeros(2, wp.float32)),
     "weights has 2 float32 values, but 3 are needed"),
    (lambda: pycunls.SE2DiskClearanceFactorBatch(wp.zeros(5, wp.float32), 0.1, 2),
     "obstacles has 5 float32 values, but 6 are needed"),
    (lambda: pycunls.ImuFactorBatch(wp.zeros(13, wp.float32), wp.zeros(3, wp.int32), 2,
                                    pycunls.ImuParameters(), 2),
     "imu_samples has 13 float32 values, but 14 are needed"),
])
def test_too_small_arrays_are_rejected(make, needle):
    with pytest.raises(ValueError, match=needle):
        make()


@pytest.mark.parametrize("make, needle", [
    # x 16 floats per state wraps size_t to a small number.
    (lambda: pycunls.SE3StateBatch(wp.zeros((1, 4, 4), wp.float32), 2**64 // 16 + 1),
     "overflows"),
    (lambda: pycunls.BoundFactorBatch3(wp.zeros(9, wp.float32), wp.zeros(9, wp.float32), 2**63),
     "overflows size_t"),
], ids=["state", "factor"])
def test_capacity_that_overflows_the_size_is_rejected(make, needle):
    with pytest.raises(ValueError, match=needle):
        make()


def test_every_device_buffer_dtype_is_checked():
    """float32 data, int32 ids, uint64 pointer tables."""
    with pytest.raises(TypeError, match="const_ids must have dtype int32, got int64"):
        pycunls.SE3StateBatch(wp.zeros(16 * 2, wp.float32), 2, wp.zeros(1, wp.int64), 1)
    with pytest.raises(TypeError, match="observations must have dtype float32"):
        pycunls.ReprojectionFactorBatch(wp.zeros(4, wp.float64), 2)
    with pytest.raises(TypeError, match="lower must have dtype float32"):
        pycunls.BoundFactorBatch2(wp.zeros(4, wp.float64), wp.zeros(4, wp.float32), 2)
    x = wp.zeros(4, wp.float32)
    states = pycunls.VectorStateBatch2(x, 2)
    states.set_num_active_states(2)
    with pytest.raises(TypeError, match="upper must have dtype float32"):
        states.set_bounds(wp.zeros(4, wp.float32), wp.zeros(4, wp.int32))
    prior = pycunls.PriorVectorFactorBatch2(wp.zeros(4, wp.float32), 2)
    prior.set_num_active_factors(2)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    table = wp.array([states.state_device_ptr(0), states.state_device_ptr(1)], dtype=wp.int64)
    with pytest.raises(TypeError, match="state_pointer_table must have dtype uint64"):
        problem.add_factor_batch(prior, state_pointer_table=table)
    with pytest.raises(TypeError, match="state_problem_ids must have dtype int32"):
        problem.set_problem_partition(2, [wp.array(np.arange(2, dtype=np.int64))])
