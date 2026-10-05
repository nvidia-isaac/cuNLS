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

"""Tests for CuPy array interoperability.

Validates that pycunls constructors accept both raw integer device pointers
and ``cupy.ndarray`` objects, and that the two paths resolve to the same
GPU address.  Also verifies full round-trip: CuPy → pycunls optimise →
CuPy read-back.
"""

import cupy as cp
import numpy as np
import pytest

import pycunls


class TestDevicePointerExtraction:
    """Verify that extract_device_ptr handles both int and CuPy inputs."""

    def test_accepts_cupy_array(self):
        data = cp.zeros(30, dtype=cp.float32)
        sb = pycunls.VectorStateBatch3(data, 10)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        assert sb.num_active_states == 10

    def test_accepts_int_pointer(self):
        data = cp.zeros(30, dtype=cp.float32)
        ptr = int(data.data.ptr)
        sb = pycunls.VectorStateBatch3(ptr, 10)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        assert sb.num_active_states == 10

    def test_cupy_and_int_give_same_ptr(self):
        data = cp.zeros(30, dtype=cp.float32)
        sb_cp = pycunls.VectorStateBatch3(data, 10)
        sb_cp.set_num_active_states(sb_cp.capacity, sb_cp.const_capacity)
        sb_int = pycunls.VectorStateBatch3(int(data.data.ptr), 10)
        sb_int.set_num_active_states(sb_int.capacity, sb_int.const_capacity)
        assert sb_cp.state_device_ptr(0) == sb_int.state_device_ptr(0)


class TestCuPyRoundTrip:
    """End-to-end: allocate via CuPy, optimise with pycunls, read back via CuPy."""

    def test_optimize_and_read_back(self, stream):
        """Write data via CuPy, optimize, read back updated data via CuPy."""
        target = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        states_gpu = cp.asarray(initial)
        obs_gpu = cp.asarray(target)

        sb = pycunls.VectorStateBatch3(states_gpu, 1)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        fb = pycunls.PriorVectorFactorBatch3(obs_gpu, 1)
        fb.set_num_active_factors(fb.capacity)

        problem = pycunls.Problem()
        problem.add_state_batch(sb)
        problem.add_factor_batch(fb, [sb.state_device_ptr(0)])
        assert problem.check_consistency()

        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 10
        opts.disable_safety_checks = False
        minimizer = pycunls.GaussNewtonMinimizer(opts)
        minimizer.minimize(stream, problem)
        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        result = cp.asnumpy(states_gpu)
        np.testing.assert_allclose(result, target, atol=1e-3)

    def test_factor_accepts_cupy(self):
        """Factor constructors should accept CuPy arrays directly."""
        obs = cp.zeros(20, dtype=cp.float32)
        fb = pycunls.ReprojectionFactorBatch(obs, 10)
        fb.set_num_active_factors(fb.capacity)
        assert fb.num_active_factors == 10


class TestDtypeChecks:
    """Every device buffer passed as a CuPy array is checked against the dtype
    the kernels read (float32 data, int32 ids, uint64 pointer tables); raw
    integer pointers are accepted unchecked."""

    def test_state_batch_data_and_const_ids(self):
        with pytest.raises(TypeError, match="data must have dtype float32, got float64"):
            pycunls.VectorStateBatch3(cp.zeros(9, dtype=cp.float64), 3)
        with pytest.raises(TypeError, match="const_ids must have dtype int32, got int64"):
            pycunls.SE3StateBatch(cp.zeros(16 * 2, dtype=cp.float32), 2,
                                  cp.arange(1), 1)
        # Raw pointers are not checked.
        data = cp.zeros(9, dtype=cp.float32)
        pycunls.VectorStateBatch3(data.data.ptr, 3)

    def test_factor_measurements_and_bounds(self):
        with pytest.raises(TypeError, match="observations must have dtype float32"):
            pycunls.ReprojectionFactorBatch(cp.zeros(4, dtype=cp.float64), 2)
        with pytest.raises(TypeError, match="lower must have dtype float32"):
            pycunls.BoundFactorBatch2(cp.zeros(4, dtype=cp.float64),
                                      cp.zeros(4, dtype=cp.float32), 2)
        states = pycunls.VectorStateBatch2(cp.zeros(4, dtype=cp.float32), 2)
        with pytest.raises(TypeError, match="upper must have dtype float32"):
            states.set_bounds(cp.zeros(4, dtype=cp.float32), cp.zeros(4, dtype=cp.int32))

    def test_problem_tables(self):
        x = cp.zeros(4, dtype=cp.float32)
        states = pycunls.VectorStateBatch2(x, 2)
        states.set_num_active_states(2)
        prior = pycunls.PriorVectorFactorBatch2(cp.zeros(4, dtype=cp.float32), 2)
        prior.set_num_active_factors(2)
        problem = pycunls.Problem()
        problem.add_state_batch(states)
        table = cp.asarray([states.state_device_ptr(0), states.state_device_ptr(1)],
                           dtype=cp.int64)
        with pytest.raises(TypeError, match="state_pointer_table must have dtype uint64"):
            problem.add_factor_batch(prior, state_pointer_table=table)
        with pytest.raises(TypeError, match="state_problem_ids must have dtype int32"):
            problem.set_problem_partition(2, [cp.arange(2)])
