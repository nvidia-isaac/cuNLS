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

A positive-scalar state batch whose Plus is multiplicative (x (+) delta = x * exp(delta)),
written twice: as a ``WarpStateBatch`` kernel (its last three parameters filled by pycunls), with
its signature checks, replicas and inputs, and as a subclass overriding ``plus`` with the raw
pointers (the full-control route). Paired with a log-space prior factor, both exercise the
CustomStateBatch -> C++ trampoline -> Python callback round-trip during optimisation.
"""

import numpy as np
import pytest
import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch, WarpStateBatch

wp.init()


# ── Warp kernels ────────────────────────────────────────────────────────────

@wp.kernel
def _positive_plus_kernel(x: wp.array2d(dtype=wp.float32), delta: wp.array2d(dtype=wp.float32),
                          x_plus_delta: wp.array2d(dtype=wp.float32)):
    """Multiplicative retraction: x_plus = x * exp(delta), one thread per state."""
    i = wp.tid()
    x_plus_delta[i, 0] = x[i, 0] * wp.exp(delta[i, 0])


@wp.kernel
def _scaled_plus_kernel(scale: float, x: wp.array2d(dtype=wp.float32),
                        delta: wp.array2d(dtype=wp.float32),
                        x_plus_delta: wp.array2d(dtype=wp.float32)):
    """x + scale * delta on a 2-vector state: a kernel with an input of its own."""
    i = wp.tid()
    for k in range(2):
        x_plus_delta[i, k] = x[i, k] + scale * delta[i, k]


@wp.kernel
def _flat_plus_kernel(x: wp.array(dtype=wp.float32), delta: wp.array(dtype=wp.float32),
                      x_plus_delta: wp.array(dtype=wp.float32)):
    """The same retraction on flat arrays, for the plus() override."""
    i = wp.tid()
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


# ── Custom state batches ────────────────────────────────────────────────────

def positive_states(data, capacity, **kwargs):
    """Positive-scalar manifold from its kernel: Plus(x, delta) = x * exp(delta)."""
    return WarpStateBatch(_positive_plus_kernel, data=data, ambient_size=1, tangent_size=1,
                          capacity=capacity, **kwargs)


class PositiveScalarOverride(WarpStateBatch):
    """The same manifold through a plus() override (raw pointers, full control)."""

    def __init__(self, data, capacity, **kwargs):
        super().__init__(data=data, ambient_size=1, tangent_size=1, capacity=capacity, **kwargs)

    def plus(self, x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle, num_replicas):
        n = self.num_active_states * num_replicas  # contiguous replicas, per-state manifold
        wp.launch(_flat_plus_kernel, dim=n,
                  inputs=[self.wrap_array(x_ptr, wp.float32, n),
                          self.wrap_array(delta_ptr, wp.float32, n)],
                  outputs=[self.wrap_array(x_plus_delta_ptr, wp.float32, n)],
                  stream=self.make_warp_stream(stream_handle))


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
        sb = positive_states(data, 10)
        assert sb.capacity == 10
        assert sb.num_active_states == 0  # zero until set
        sb.set_num_active_states(10)
        assert sb.num_active_states == 10
        assert sb.ambient_size == 1
        assert sb.tangent_size == 1

    def test_state_device_ptr(self):
        data = wp.ones(5, dtype=wp.float32)
        sb = positive_states(data, 5)
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

    def test_plus_covers_every_replica(self, stream):
        """n = replicas x active states: the kernel never sees the replica count."""
        sb = positive_states(wp.ones(4, dtype=wp.float32), 4)
        sb.set_num_active_states(3)
        x = wp.array(np.arange(1, 7, dtype=np.float32))            # 2 replicas x 3 states
        delta = wp.array(np.log(np.full(6, 2.0, np.float32)))
        out = wp.zeros(6, dtype=wp.float32)
        sb.plus(x.ptr, delta.ptr, out.ptr, stream.get_stream(), 2)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), 2.0 * np.arange(1, 7), rtol=1e-6)

    def test_inputs_are_read_at_every_update(self, stream):
        sb = WarpStateBatch(_scaled_plus_kernel, inputs=[1.0], data=wp.zeros(4, dtype=float),
                            ambient_size=2, tangent_size=2, capacity=2)
        sb.set_num_active_states(2)
        x, delta, out = (wp.array(np.ones(4, np.float32)) for _ in range(3))
        for scale in (1.0, 3.0):
            sb.inputs[0] = scale
            sb.plus(x.ptr, delta.ptr, out.ptr, stream.get_stream(), 1)
            wp.synchronize()
            np.testing.assert_allclose(out.numpy(), 1.0 + scale)

    def test_signature_errors_say_what_and_how_to_fix(self):
        data = wp.ones(4, dtype=wp.float32)
        sizes = dict(data=data, ambient_size=1, tangent_size=1, capacity=4)
        with pytest.raises(TypeError, match="x: wp.array2d.*Fix:"):
            WarpStateBatch(_flat_plus_kernel, **sizes)
        with pytest.raises(ValueError, match="got 0 inputs, expected 1: scale.*Fix:"):
            WarpStateBatch(_scaled_plus_kernel, **sizes)
        with pytest.raises(ValueError, match="inputs without a kernel.*Fix:"):
            WarpStateBatch(inputs=[1.0], **sizes)
        no_kernel = WarpStateBatch(**sizes)
        no_kernel.set_num_active_states(1)
        with pytest.raises(NotImplementedError, match="Fix:"):
            no_kernel.plus(0, 0, 0, 0, 1)

    def test_with_const_ids(self):
        data = wp.ones(10, dtype=wp.float32)
        const_ids = wp.array([0], dtype=wp.int32)
        sb = positive_states(data, 10, const_state_ids=const_ids, const_capacity=1)
        assert (sb.capacity, sb.const_capacity) == (10, 1)
        sb.set_num_active_states(10, 1)
        assert sb.num_active_states == 10


class TestPositiveScalarEndToEnd:
    """End-to-end optimisation with the positive-scalar custom state batch."""

    @pytest.mark.parametrize("make", [positive_states, PositiveScalarOverride])
    def test_converges_to_target(self, stream, make):
        """Solve a log-space prior problem and verify recovery of targets."""
        rng = np.random.default_rng(42)
        n = 20

        targets = rng.uniform(1.0, 10.0, n).astype(np.float32)
        initial = targets * rng.uniform(0.5, 2.0, n).astype(np.float32)

        states_gpu = wp.array(initial)
        obs_wp = wp.array(targets, dtype=wp.float32, device="cuda:0")

        sb = make(states_gpu, n)
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
