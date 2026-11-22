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

A vector prior (``residual = state - obs``, ``Jacobian = I``) written twice:
  - as a ``WarpFactorBatch`` kernel (the kernel's last four parameters filled by pycunls), with
    its signature checks, residual-only evaluations and repeated factor ids (RANSAC);
  - as a subclass overriding ``evaluate`` with the raw pointers (``wrap_array``,
    ``factor_ids``, ``make_warp_stream``: the full-control route).
Both solve a linear least-squares prior in one Gauss-Newton step.
"""

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
    state_ptrs: wp.array(dtype=wp.uint64),
    residuals: wp.array(dtype=wp.float32),
    jacobians: wp.array(dtype=wp.float32),
    dim: int,
    num_items: int,
    write_jac: int,
):
    """Per item: residual[d] = state[d] - observation_of_its_factor[d], Jacobian = I.

    Item i reads its dim-float state through its own pointer ``state_ptrs[i]``.
    """
    i = wp.tid()
    if i >= num_items:
        return
    m = factor_ids[i]
    state = wp.array(ptr=state_ptrs[i], shape=(dim,), dtype=wp.float32)
    for d in range(dim):
        residuals[i * dim + d] = state[d] - observations[m * dim + d]
        if write_jac != 0:
            for d2 in range(dim):
                if d == d2:
                    jacobians[i * dim * dim + d * dim + d2] = 1.0
                else:
                    jacobians[i * dim * dim + d * dim + d2] = 0.0


class WarpPriorFactor(WarpFactorBatch):
    """Dim-D vector-prior factor implemented entirely in Warp.

    ``evaluate`` launches ``_prior_kernel`` on the provided CUDA stream with
    one thread per item; each item reads its state through the
    ``state_pointers`` device array.
    """

    def __init__(self, observations_wp, dim, capacity):
        super().__init__(residual_size=dim, state_sizes=[dim],
                         capacity=capacity)
        self.observations = observations_wp
        self._dim = dim

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, factor_ids_ptr, num_factor_ids):
        n = num_factor_ids
        dim = self._dim
        state_ptrs = self.wrap_array(sp_ptr, wp.uint64, n)

        res = self.wrap_array(res_ptr, wp.float32, n * dim)
        write_jac = 1 if jac_ptr != 0 else 0
        jac = (self.wrap_array(jac_ptr, wp.float32, n * dim * dim)
               if jac_ptr != 0 else wp.zeros(1, dtype=wp.float32, device=self._device))

        s = self.make_warp_stream(stream_handle)
        wp.launch(_prior_kernel, dim=n,
                  inputs=[self.observations, self.factor_ids(factor_ids_ptr, n), state_ptrs,
                          res, jac, dim, n, write_jac],
                  stream=s)
        return True


@wp.kernel
def _prior3_kernel(observations: wp.array(dtype=wp.vec3), factor_ids: wp.array(dtype=wp.int32),
                   states: wp.array2d(dtype=wp.uint64), residuals: wp.array2d(dtype=float),
                   jacobians: wp.array3d(dtype=float)):
    t = wp.tid()
    x = wp.array(ptr=states[t, 0], shape=(3,), dtype=float)
    obs = observations[factor_ids[t]]
    for d in range(3):
        residuals[t, d] = x[d] - obs[d]
    if jacobians.shape[0] > 0:
        for d in range(3):
            for c in range(3):
                jacobians[t, d, c] = wp.where(d == c, 1.0, 0.0)


def _prior_problem(n):
    """n 3-vectors at 0 pulled to their observations by the kernel factor."""
    targets = np.arange(3 * n, dtype=np.float32).reshape(n, 3)
    x = wp.zeros(3 * n, dtype=float)
    states = pycunls.VectorStateBatch3(x, n)
    states.set_num_active_states(n)
    prior = WarpFactorBatch(_prior3_kernel, inputs=[wp.array(targets, dtype=wp.vec3)],
                            residual_size=3, state_sizes=[3], capacity=n)
    prior.set_num_active_factors(n)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    return problem, prior, states, x, targets


@wp.kernel
def _wrong_tail(observations: wp.array(dtype=wp.vec3), factor_ids: wp.array(dtype=wp.int32),
                states: wp.array(dtype=wp.uint64), residuals: wp.array2d(dtype=float),
                jacobians: wp.array3d(dtype=float)):
    pass


class TestKernelFactor:
    """A WarpFactorBatch built from a kernel: no subclass, shaped arrays."""

    def test_solves_and_passes_check_jacobians(self, stream):
        problem, prior, states, x, targets = _prior_problem(64)
        problem.add_factor_batch(prior, [states.state_device_ptr(i) for i in range(64)])
        for check in pycunls.check_jacobians(problem, stream):
            assert check.ok, check
        summary = pycunls.GaussNewtonMinimizer().minimize(stream, problem)
        assert summary.converged(), summary.message
        np.testing.assert_allclose(x.numpy().reshape(-1, 3), targets, atol=1e-4)

    def test_residuals_only_with_a_numeric_jacobian(self, stream):
        """JacobianMode.numeric evaluates residuals only: the kernel sees zero-row jacobians."""
        problem, prior, states, x, targets = _prior_problem(8)
        problem.add_factor_batch(prior, [states.state_device_ptr(i) for i in range(8)],
                                 jacobian_mode_override=pycunls.JacobianMode.numeric)
        summary = pycunls.LevenbergMarquardtMinimizer().minimize(stream, problem)
        assert summary.converged(), summary.message
        np.testing.assert_allclose(x.numpy().reshape(-1, 3), targets, atol=1e-3)

    def test_inputs_are_read_at_every_evaluation(self, stream):
        problem, prior, states, x, targets = _prior_problem(4)
        problem.add_factor_batch(prior, [states.state_device_ptr(i) for i in range(4)])
        pycunls.GaussNewtonMinimizer().minimize(stream, problem)
        prior.inputs[0] = wp.array(-targets, dtype=wp.vec3)   # new data, same batch
        pycunls.GaussNewtonMinimizer().minimize(stream, problem)
        np.testing.assert_allclose(x.numpy().reshape(-1, 3), -targets, atol=1e-4)

    def test_repeated_factor_ids(self, stream):
        """Explicit ids (as RANSAC passes them) select each thread's factor."""
        problem, prior, states, x, targets = _prior_problem(4)
        problem.add_factor_batch(prior, [states.state_device_ptr(i) for i in range(4)])
        n = 6
        ids = wp.array(np.array([3, 1, 3, 0, 2, 1], np.int32))
        table = wp.array(np.full(n, states.state_device_ptr(0), np.uint64))
        residuals = wp.zeros(3 * n, dtype=float)
        prior.evaluate(residuals.ptr, 0, table.ptr, stream.get_stream(), ids.ptr, n)
        wp.synchronize()
        np.testing.assert_allclose(residuals.numpy().reshape(n, 3), -targets[ids.numpy()])

    def test_signature_errors_say_what_and_how_to_fix(self):
        obs = wp.zeros(1, dtype=wp.vec3)
        with pytest.raises(TypeError, match="states: wp.array2d.*Fix:"):
            WarpFactorBatch(_wrong_tail, inputs=[obs], residual_size=3, state_sizes=[3],
                            capacity=1)
        with pytest.raises(ValueError, match="got 2 inputs, expected 1: observations.*Fix:"):
            WarpFactorBatch(_prior3_kernel, inputs=[obs, obs], residual_size=3,
                            state_sizes=[3], capacity=1)
        with pytest.raises(TypeError, match="not a Warp kernel.*Fix:"):
            WarpFactorBatch(lambda: None, residual_size=3, state_sizes=[3], capacity=1)
        with pytest.raises(ValueError, match="inputs without a kernel.*Fix:"):
            WarpFactorBatch(inputs=[obs], residual_size=3, state_sizes=[3], capacity=1)
        no_kernel = WarpFactorBatch(residual_size=1, state_sizes=[1], capacity=1)
        with pytest.raises(NotImplementedError, match="Fix:"):
            no_kernel.evaluate(0, 0, 0, 0, 0, 1)


class TestWarpFactorBatch:
    """Verify WarpFactorBatch helpers and end-to-end convergence of an evaluate() override."""
    def test_wrap_array(self):
        fb = WarpFactorBatch(residual_size=1, state_sizes=[1], capacity=10)
        data = wp.zeros(5, dtype=wp.float32, device="cuda:0")
        arr = fb.wrap_array(data.ptr, wp.float32, 5)
        assert arr.shape == (5,)

    def test_default_factor_ids_cache_is_bounded(self):
        """Default ids are t % num_active_factors; only two active counts stay cached."""
        fb = WarpFactorBatch(residual_size=1, state_sizes=[1], capacity=10)
        for count in [3, 5, 7, 5, 9]:
            fb.set_num_active_factors(count)
            ids = fb.factor_ids(0, 2 * count).numpy()
            np.testing.assert_array_equal(ids, np.arange(2 * count) % count)
            assert {key[1] for key in fb._default_ids} <= {count, *fb._default_ids_counts}
            assert len(fb._default_ids_counts) <= 2
        assert sorted(fb._default_ids_counts) == [5, 9]

    def test_default_factor_ids_need_active_factors(self):
        """Items without active factors are an error; zero items are fine."""
        fb = WarpFactorBatch(residual_size=1, state_sizes=[1], capacity=10)  # 0 active factors
        assert fb.factor_ids(0, 0).shape == (0,)
        with pytest.raises(ValueError, match="set_num_active_factors"):
            fb.factor_ids(0, 4)

    def test_end_to_end_convergence(self, stream):
        """Solve a 3D vector-prior problem using a Warp-based factor."""
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        states_gpu = wp.array(initial)
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
        wp.synchronize()

        assert summary.final_cost < 1e-6
        result = states_gpu.numpy()
        np.testing.assert_allclose(result, target, atol=1e-3)
