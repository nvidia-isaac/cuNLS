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

"""Tests for the Warp-based custom factor integration.

Defines a ``WarpPriorFactor`` (vector-prior: ``residual = state - obs``,
``Jacobian = I``) entirely in Python/Warp and uses it in a full pycunls
optimisation loop.  This exercises:
  - ``WarpFactorBatch.wrap_array`` (zero-copy pointer wrapping)
  - ``WarpFactorBatch.make_warp_stream`` (stream forwarding)
  - The ``PyFactorBatch`` C++ trampoline (GIL acquire, Python callback)
  - End-to-end convergence with a Warp-based factor

The test is intentionally kept simple (a linear least-squares prior) so that
convergence is guaranteed in a single Gauss-Newton step.
"""

import cupy as cp
import numpy as np
import pytest
import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch

wp.init()


@wp.kernel
def _prior_kernel(
    observations: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.float32),
    residuals: wp.array(dtype=wp.float32),
    jacobians: wp.array(dtype=wp.float32),
    dim: int,
    num_factors: int,
    write_jac: int,
):
    """Per-factor: residual[d] = state[d] - observation[d], Jacobian = I."""
    i = wp.tid()
    if i >= num_factors:
        return
    for d in range(dim):
        idx = i * dim + d
        residuals[idx] = states[idx] - observations[idx]
        if write_jac != 0:
            for d2 in range(dim):
                if d == d2:
                    jacobians[i * dim * dim + d * dim + d2] = 1.0
                else:
                    jacobians[i * dim * dim + d * dim + d2] = 0.0


class WarpPriorFactor(WarpFactorBatch):
    """Dim-D vector-prior factor implemented entirely in Warp.

    This factor reads scattered state block pointers from the
    ``state_pointers`` device array, wraps them as contiguous Warp arrays,
    and launches ``_prior_kernel`` on the provided CUDA stream.
    """

    def __init__(self, observations_wp, dim, num_factors):
        super().__init__(residual_size=dim, state_block_sizes=[dim],
                         num_factors=num_factors)
        self.observations = observations_wp
        self._dim = dim
        self._num = num_factors

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle):
        n = self._num
        dim = self._dim

        sp_cp = cp.array(
            cp.ndarray(shape=(n,), dtype=cp.uint64,
                       memptr=cp.cuda.MemoryPointer(
                           cp.cuda.UnownedMemory(sp_ptr, n * 8, None), 0)))
        host_ptrs = cp.asnumpy(sp_cp)

        state_data = cp.ndarray(
            shape=(n * dim,), dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(int(host_ptrs[0]), n * dim * 4, None), 0))

        states_wp = wp.array(ptr=int(state_data.data.ptr), dtype=wp.float32,
                             shape=(n * dim,), device=self._device, copy=False)

        res = self.wrap_array(res_ptr, wp.float32, n * dim)
        write_jac = 1 if jac_ptr != 0 else 0
        jac = (self.wrap_array(jac_ptr, wp.float32, n * dim * dim)
               if jac_ptr != 0 else wp.zeros(1, dtype=wp.float32, device=self._device))

        s = self.make_warp_stream(stream_handle)
        wp.launch(_prior_kernel, dim=n,
                  inputs=[self.observations, states_wp, res, jac,
                          dim, n, write_jac],
                  stream=s)
        return True


class TestWarpFactorBatch:
    """Verify WarpFactorBatch helpers and end-to-end convergence."""
    def test_wrap_array(self):
        fb = WarpFactorBatch(1, [1], 10)
        data = wp.zeros(5, dtype=wp.float32, device="cuda:0")
        arr = fb.wrap_array(data.ptr, wp.float32, 5)
        assert arr.shape == (5,)

    def test_end_to_end_convergence(self, stream):
        """Solve a 3D vector-prior problem using a Warp-based factor."""
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        states_gpu = cp.asarray(initial)
        obs_wp = wp.array(target, dtype=wp.float32, device="cuda:0")

        sb = pycunls.VectorStateBatch3(states_gpu, 1)
        fb = WarpPriorFactor(obs_wp, 3, 1)

        problem = pycunls.Problem()
        problem.add_state_batch(sb)
        problem.add_factor_batch(fb, [sb.state_block_device_ptr(0)])
        assert problem.check_consistency()

        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 10
        minimizer = pycunls.GaussNewtonMinimizer(opts)
        summary = minimizer.minimize(stream, problem)
        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        assert summary.final_cost < 1e-6
        result = cp.asnumpy(states_gpu)
        np.testing.assert_allclose(result, target, atol=1e-3)
