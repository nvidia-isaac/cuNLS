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

import cupy as cp
import numpy as np
import pytest

import pycunls


def test_host_array_is_rejected_with_the_fix():
    with pytest.raises(TypeError, match=r"host \(NumPy\) array.*cupy.asarray\(data\)"):
        pycunls.VectorStateBatch3(np.zeros(9, np.float32), 3)


def test_wrong_dtype_names_the_expected_one():
    with pytest.raises(TypeError, match="must have dtype float32, got float64"):
        pycunls.VectorStateBatch3(cp.zeros(9, cp.float64), 3)


def test_too_small_state_array_counts_the_floats():
    with pytest.raises(ValueError, match=r"has 36 float32 values, but 48 are needed.*16 float32"):
        pycunls.SE3StateBatch(cp.zeros((3, 12), cp.float32), 3)


def test_too_small_measurement_array():
    with pytest.raises(ValueError, match="deltas has 32 float32 values, but 48 are needed"):
        pycunls.SE3BetweenFactorBatch(cp.zeros((2, 4, 4), cp.float32), 3)


def test_too_small_constant_id_array():
    with pytest.raises(ValueError, match="const_ids has 1 int32 values, but 2 are needed"):
        pycunls.VectorStateBatch2(cp.zeros(8, cp.float32), 4, cp.zeros(1, cp.int32), 2)


def test_strided_view_is_rejected():
    with pytest.raises(ValueError, match="C-contiguous"):
        pycunls.VectorStateBatch3(cp.zeros((3, 6), cp.float32)[:, ::2], 3)


def test_non_array_is_rejected():
    with pytest.raises(TypeError, match="must be a GPU array"):
        pycunls.VectorStateBatch3([1.0, 2.0, 3.0], 1)


def test_valid_arrays_and_raw_pointers_still_work():
    data = cp.zeros((3, 4, 4), cp.float32)
    pycunls.SE3StateBatch(data, 3)
    pycunls.SE3StateBatch(data.data.ptr, 3)  # an int pointer is not checked


def test_too_small_pnp_rig_pose_array():
    obs = cp.zeros((4, 2), cp.float32)
    points = cp.zeros((4, 3), cp.float32)
    rig = cp.zeros((2, 4, 4), cp.float32)  # one rig pose per factor is needed
    with pytest.raises(ValueError, match="poses_camera_from_rig has 32 float32 values, but 64"):
        pycunls.PnPFactorBatch(obs, rig, points, 4)


def test_too_small_custom_state_arrays():
    class Vec2(pycunls.CustomStateBatch):
        def plus(self, *args):
            pass

    with pytest.raises(ValueError, match="data has 6 float32 values, but 8 are needed"):
        Vec2(cp.zeros(6, cp.float32), 2, 2, 4)
    with pytest.raises(ValueError, match="const_ids has 1 int32 values, but 2 are needed"):
        Vec2(cp.zeros(8, cp.float32), 2, 2, 4, cp.zeros(1, cp.int32), 2)


def test_capacity_overflow_is_rejected():
    huge = 2**64 // 16 + 1  # x 16 floats per state wraps size_t to a small number
    with pytest.raises(ValueError, match="overflows"):
        pycunls.SE3StateBatch(cp.zeros((1, 4, 4), cp.float32), huge)


@pytest.mark.parametrize("make, needle", [
    (lambda: pycunls.ConstantVelocitySE3FactorBatch(cp.zeros(3, cp.float32), 4),
     "time_steps has 3 float32 values, but 4 are needed"),
    (lambda: pycunls.ConstantVelocityInformationSE3FactorBatch(
        pycunls.CudaStream(), cp.zeros(4, cp.float32), cp.zeros(3, cp.float32), 4),
     "qc_diag has 3 float32 values, but 6 are needed"),
    (lambda: pycunls.BoundFactorBatch3(cp.zeros(6, cp.float32), cp.zeros(9, cp.float32), 3),
     "lower has 6 float32 values, but 9 are needed"),
    (lambda: pycunls.HalfspaceFactorBatch2(cp.zeros(4, cp.float32), cp.zeros(3, cp.float32), 3),
     "normals has 4 float32 values, but 6 are needed"),
    (lambda: pycunls.WeightedFactorBatch(
        pycunls.PriorVectorFactorBatch3(cp.zeros(9, cp.float32), 3), weights=cp.zeros(2, cp.float32)),
     "weights has 2 float32 values, but 3 are needed"),
    (lambda: pycunls.SE2DiskClearanceFactorBatch(cp.zeros(5, cp.float32), 0.1, 2),
     "obstacles has 5 float32 values, but 6 are needed"),
    (lambda: pycunls.ImuFactorBatch(cp.zeros(13, cp.float32), cp.zeros(3, cp.int32), 2,
                                    pycunls.ImuParameters(), 2),
     "imu_samples has 13 float32 values, but 14 are needed"),
])
def test_too_small_factor_arrays_are_rejected(make, needle):
    with pytest.raises(ValueError, match=needle):
        make()


def test_too_small_bounds_are_rejected():
    states = pycunls.VectorStateBatch3(cp.zeros(9, cp.float32), 3)
    with pytest.raises(ValueError, match="upper has 6 float32 values, but 9 are needed"):
        states.set_bounds(cp.zeros(9, cp.float32), cp.zeros(6, cp.float32))


def test_capacity_that_overflows_the_size_is_rejected():
    with pytest.raises(ValueError, match="overflows size_t"):
        pycunls.BoundFactorBatch3(cp.zeros(9, cp.float32), cp.zeros(9, cp.float32), 2**63)
