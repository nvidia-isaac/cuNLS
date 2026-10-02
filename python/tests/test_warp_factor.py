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
    factor_ids: wp.array(dtype=wp.int32),
    states: wp.array(dtype=wp.float32),
    residuals: wp.array(dtype=wp.float32),
    jacobians: wp.array(dtype=wp.float32),
    dim: int,
    num_items: int,
    write_jac: int,
):
    """Per item: residual[d] = state[d] - observation_of_its_factor[d], Jacobian = I."""
    i = wp.tid()
    if i >= num_items:
        return
    m = factor_ids[i]
    for d in range(dim):
        residuals[i * dim + d] = states[i * dim + d] - observations[m * dim + d]
        if write_jac != 0:
            for d2 in range(dim):
                if d == d2:
                    jacobians[i * dim * dim + d * dim + d2] = 1.0
                else:
                    jacobians[i * dim * dim + d * dim + d2] = 0.0


# Copies each item's dim-float state (one pointer per item) into a
# contiguous array: Warp kernels cannot dereference raw pointers themselves.
_gather_blocks_kernel = cp.RawKernel(r"""
extern "C" __global__
void gather_blocks(const unsigned long long* ptrs, float* out, int items, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < items * dim) {
        const float* p = reinterpret_cast<const float*>(ptrs[i / dim]);
        out[i] = p[i % dim];
    }
}
""", "gather_blocks")


class WarpPriorFactor(WarpFactorBatch):
    """Dim-D vector-prior factor implemented entirely in Warp.

    ``evaluate`` gathers each item's state through the ``state_pointers``
    device array (a CuPy kernel), then launches ``_prior_kernel`` on the
    provided CUDA stream with one thread per item.
    """

    def __init__(self, observations_wp, dim, capacity):
        super().__init__(residual_size=dim, state_sizes=[dim],
                         capacity=capacity)
        self.observations = observations_wp
        self._dim = dim

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, factor_ids_ptr, num_factor_ids):
        n = num_factor_ids
        dim = self._dim
        ptrs = cp.ndarray(
            shape=(n,), dtype=cp.uint64,
            memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(sp_ptr, n * 8, None), 0))
        states = cp.empty(n * dim, dtype=cp.float32)
        threads = 256
        _gather_blocks_kernel(((n * dim + threads - 1) // threads,), (threads,),
                              (ptrs, states, np.int32(n), np.int32(dim)))
        states_wp = wp.array(ptr=int(states.data.ptr), dtype=wp.float32,
                             shape=(n * dim,), device=self._device, copy=False)

        res = self.wrap_array(res_ptr, wp.float32, n * dim)
        write_jac = 1 if jac_ptr != 0 else 0
        jac = (self.wrap_array(jac_ptr, wp.float32, n * dim * dim)
               if jac_ptr != 0 else wp.zeros(1, dtype=wp.float32, device=self._device))

        s = self.make_warp_stream(stream_handle)
        wp.launch(_prior_kernel, dim=n,
                  inputs=[self.observations, self.factor_ids(factor_ids_ptr, n), states_wp,
                          res, jac, dim, n, write_jac],
                  stream=s)
        return True


class TestWarpFactorBatch:
    """Verify WarpFactorBatch helpers and end-to-end convergence."""
    def test_wrap_array(self):
        fb = WarpFactorBatch(1, [1], 10)
        data = wp.zeros(5, dtype=wp.float32, device="cuda:0")
        arr = fb.wrap_array(data.ptr, wp.float32, 5)
        assert arr.shape == (5,)

    def test_default_factor_ids_cache_is_bounded(self):
        """Default ids are t % num_active_factors; only two active counts stay cached."""
        fb = WarpFactorBatch(1, [1], 10)
        for count in [3, 5, 7, 5, 9]:
            fb.set_num_active_factors(count)
            ids = fb.factor_ids(0, 2 * count).numpy()
            np.testing.assert_array_equal(ids, np.arange(2 * count) % count)
            assert {key[1] for key in fb._default_ids} <= {count, *fb._default_ids_counts}
            assert len(fb._default_ids_counts) <= 2
        assert sorted(fb._default_ids_counts) == [5, 9]

    def test_end_to_end_convergence(self, stream):
        """Solve a 3D vector-prior problem using a Warp-based factor."""
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        states_gpu = cp.asarray(initial)
        obs_wp = wp.array(target, dtype=wp.float32, device="cuda:0")

        sb = pycunls.VectorStateBatch3(states_gpu, 1)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        fb = WarpPriorFactor(obs_wp, 3, 1)
        fb.set_num_active_factors(fb.capacity)

        problem = pycunls.Problem()
        problem.add_state_batch(sb)
        problem.add_factor_batch(fb, [sb.state_device_ptr(0)])
        assert problem.check_consistency()

        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 10
        opts.disable_safety_checks = False
        minimizer = pycunls.GaussNewtonMinimizer(opts)
        summary = minimizer.minimize(stream, problem)
        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        assert summary.final_cost < 1e-6
        result = cp.asnumpy(states_gpu)
        np.testing.assert_allclose(result, target, atol=1e-3)
