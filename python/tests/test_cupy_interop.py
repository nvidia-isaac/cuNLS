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
        assert sb.num_state_blocks == 10

    def test_accepts_int_pointer(self):
        data = cp.zeros(30, dtype=cp.float32)
        ptr = int(data.data.ptr)
        sb = pycunls.VectorStateBatch3(ptr, 10)
        assert sb.num_state_blocks == 10

    def test_cupy_and_int_give_same_ptr(self):
        data = cp.zeros(30, dtype=cp.float32)
        sb_cp = pycunls.VectorStateBatch3(data, 10)
        sb_int = pycunls.VectorStateBatch3(int(data.data.ptr), 10)
        assert sb_cp.state_block_device_ptr(0) == sb_int.state_block_device_ptr(0)


class TestCuPyRoundTrip:
    """End-to-end: allocate via CuPy, optimise with pycunls, read back via CuPy."""

    def test_optimize_and_read_back(self, stream):
        """Write data via CuPy, optimize, read back updated data via CuPy."""
        target = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        states_gpu = cp.asarray(initial)
        obs_gpu = cp.asarray(target)

        sb = pycunls.VectorStateBatch3(states_gpu, 1)
        fb = pycunls.PriorVectorFactorBatch3(obs_gpu, 1)

        problem = pycunls.Problem()
        problem.add_state_batch(sb)
        problem.add_factor_batch(fb, [sb.state_block_device_ptr(0)])
        assert problem.check_consistency()

        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 10
        minimizer = pycunls.GaussNewtonMinimizer(opts)
        minimizer.minimize(stream, problem)
        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        result = cp.asnumpy(states_gpu)
        np.testing.assert_allclose(result, target, atol=1e-3)

    def test_factor_accepts_cupy(self, cublas):
        """Factor constructors should accept CuPy arrays directly."""
        obs = cp.zeros(20, dtype=cp.float32)
        fb = pycunls.ReprojectionFactorBatch(cublas, obs, 10)
        assert fb.num_factors == 10
