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

"""Tests for the Warp-based custom state batch integration.

Defines a positive-scalar state batch where the Plus operation is
multiplicative (x (+) delta = x * exp(delta)).  Paired with a log-space
prior factor, this exercises the full CustomStateBatch → C++ trampoline →
Python callback round-trip during optimisation.
"""

import numpy as np
import pytest
import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch, WarpStateBatch

wp.init()


# ── Warp kernels ────────────────────────────────────────────────────────────

@wp.kernel
def _positive_plus_kernel(
    x: wp.array(dtype=wp.float32),
    delta: wp.array(dtype=wp.float32),
    x_plus_delta: wp.array(dtype=wp.float32),
    n: int,
):
    """Multiplicative retraction: x_plus = x * exp(delta)."""
    i = wp.tid()
    if i >= n:
        return
    x_plus_delta[i] = x[i] * wp.exp(delta[i])


@wp.kernel
def _log_prior_kernel(
    observations: wp.array(dtype=wp.float32),
    factor_ids: wp.array(dtype=wp.int32),
    state_ptrs: wp.array(dtype=wp.uint64),
    residuals: wp.array(dtype=wp.float32),
    jacobians: wp.array(dtype=wp.float32),
    num_factors: int,
    write_jac: int,
):
    """Log-space prior: residual = log(x) - log(target), Jacobian = 1.

    Item i reads its state through its own pointer ``state_ptrs[i]``.
    """
    i = wp.tid()
    if i >= num_factors:
        return
    x = wp.array(ptr=state_ptrs[i], shape=(1,), dtype=wp.float32)
    residuals[i] = wp.log(x[0]) - wp.log(observations[factor_ids[i]])
    if write_jac != 0:
        jacobians[i] = 1.0


# ── Custom state batch ──────────────────────────────────────────────────────

class PositiveScalarStateBatch(WarpStateBatch):
    """Positive-scalar manifold: Plus(x, delta) = x * exp(delta)."""

    def __init__(self, data, capacity, **kwargs):
        super().__init__(data, ambient_size=1, tangent_size=1,
                         capacity=capacity, **kwargs)

    def plus(self, x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle, num_replicas):
        n = self.num_active_states * num_replicas  # contiguous replicas, per-state manifold
        x = self.wrap_array(x_ptr, wp.float32, n)
        delta = self.wrap_array(delta_ptr, wp.float32, n)
        x_out = self.wrap_array(x_plus_delta_ptr, wp.float32, n)
        stream = self.make_warp_stream(stream_handle)
        wp.launch(_positive_plus_kernel, dim=n,
                  inputs=[x, delta, x_out, n], stream=stream)


# ── Custom factor for log-space prior ───────────────────────────────────────

class LogPriorFactor(WarpFactorBatch):
    """Prior in log-space: residual = log(x) - log(target), Jacobian = 1.

    The Jacobian of 1 is exact because
    d/d(delta)[log(x * exp(delta))] at delta=0 equals 1.
    """

    def __init__(self, observations_wp: wp.array, capacity: int):
        super().__init__(residual_size=1, state_sizes=[1],
                         capacity=capacity)
        self.observations = observations_wp

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, factor_ids_ptr, num_factor_ids):
        n = num_factor_ids
        ids = self.factor_ids(factor_ids_ptr, n)
        state_ptrs = self.wrap_array(sp_ptr, wp.uint64, n)

        res = self.wrap_array(res_ptr, wp.float32, n)
        write_jac = 1 if jac_ptr != 0 else 0
        jac = (self.wrap_array(jac_ptr, wp.float32, n)
               if jac_ptr != 0
               else wp.zeros(1, dtype=wp.float32, device=self._device))

        stream = self.make_warp_stream(stream_handle)
        wp.launch(_log_prior_kernel, dim=n,
                  inputs=[self.observations, ids, state_ptrs, res, jac,
                          n, write_jac],
                  stream=stream)
        return True


# ── Tests ───────────────────────────────────────────────────────────────────

class TestCustomStateBatch:
    """Verify CustomStateBatch metadata and the plus() override contract."""

    def test_basic_properties(self):
        data = wp.ones(10, dtype=wp.float32)
        sb = PositiveScalarStateBatch(data, 10)
        assert sb.capacity == 10
        assert sb.num_active_states == 0  # zero until set
        sb.set_num_active_states(10)
        assert sb.num_active_states == 10
        assert sb.ambient_size == 1
        assert sb.tangent_size == 1

    def test_state_device_ptr(self):
        data = wp.ones(5, dtype=wp.float32)
        sb = PositiveScalarStateBatch(data, 5)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        p0 = sb.state_device_ptr(0)
        p1 = sb.state_device_ptr(1)
        assert p0 != 0
        assert p1 - p0 == 4  # 1 float * 4 bytes

    def test_plus_raises_without_override(self):
        data = wp.ones(3, dtype=wp.float32)
        sb = pycunls.CustomStateBatch(data, 1, 1, 3)
        with pytest.raises(RuntimeError, match="must be overridden"):
            sb.plus(0, 0, 0, 0, 1)

    def test_with_const_ids(self):
        data = wp.ones(10, dtype=wp.float32)
        const_ids = wp.array([0], dtype=wp.int32)
        sb = PositiveScalarStateBatch(data, 10,
                                      const_state_ids=const_ids,
                                      const_capacity=1)
        assert (sb.capacity, sb.const_capacity) == (10, 1)
        sb.set_num_active_states(10, 1)
        assert sb.num_active_states == 10


class TestPositiveScalarEndToEnd:
    """End-to-end optimisation with the positive-scalar custom state batch."""

    def test_converges_to_target(self, stream):
        """Solve a log-space prior problem and verify recovery of targets."""
        rng = np.random.default_rng(42)
        n = 20

        targets = rng.uniform(1.0, 10.0, n).astype(np.float32)
        initial = targets * rng.uniform(0.5, 2.0, n).astype(np.float32)

        states_gpu = wp.array(initial)
        obs_wp = wp.array(targets, dtype=wp.float32, device="cuda:0")

        sb = PositiveScalarStateBatch(states_gpu, n)
        sb.set_num_active_states(sb.capacity, sb.const_capacity)
        fb = LogPriorFactor(obs_wp, n)
        fb.set_num_active_factors(fb.capacity)

        ptrs = [sb.state_device_ptr(i) for i in range(n)]

        problem = pycunls.Problem()
        problem.add_state_batch(sb)
        problem.add_factor_batch(fb, ptrs)
        assert problem.check_consistency()

        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 20
        opts.disable_safety_checks = False
        minimizer = pycunls.GaussNewtonMinimizer(opts)
        summary = minimizer.minimize(stream, problem)
        wp.synchronize()

        assert summary.final_cost < 1e-6
        result = states_gpu.numpy()
        np.testing.assert_allclose(result, targets, rtol=1e-3)
