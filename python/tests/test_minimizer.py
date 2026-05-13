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

"""End-to-end minimizer tests using a simple vector-prior problem.

All tests share a helper ``_make_prior_problem()`` that assembles a
one-factor, one-state problem: minimise ``||x - target||^2`` with
``x ∈ R^3``.  This is a convex quadratic, so both Gauss-Newton and LM
should converge in very few iterations (often 1), making it a fast
correctness check for the full Python → C++ → GPU round-trip.
"""

import cupy as cp
import numpy as np
import pytest

import pycunls


def _make_prior_problem():
    """Build a trivial 3D vector-prior problem: minimise ||x - target||^2.

    Returns ``(problem, states_gpu, target)`` so the caller can inspect
    both the CuPy state array and the NumPy ground-truth after solving.
    """
    target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    states_gpu = cp.asarray(initial)
    obs_gpu = cp.asarray(target)

    sb = pycunls.VectorStateBatch3(states_gpu, 1)
    fb = pycunls.PriorVectorFactorBatch3(obs_gpu, 1)

    ptrs = [sb.state_block_device_ptr(0)]

    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    problem.add_factor_batch(fb, ptrs)
    assert problem.check_consistency()
    return problem, states_gpu, target


class TestGaussNewtonMinimizer:
    """Gauss-Newton convergence on the simple prior problem."""
    def test_converges(self, stream):
        problem, states_gpu, target = _make_prior_problem()

        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 10
        opts.disable_safety_checks = False
        minimizer = pycunls.GaussNewtonMinimizer(opts)
        summary = minimizer.minimize(stream, problem)

        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        assert summary.final_cost < 1e-6
        result = cp.asnumpy(states_gpu)
        np.testing.assert_allclose(result, target, atol=1e-3)

    def test_converges_with_hessian_column_scaling(self, stream):
        problem, states_gpu, target = _make_prior_problem()

        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 10
        opts.column_scaling = pycunls.ColumnScaling.hessian_diagonal
        opts.disable_safety_checks = False
        minimizer = pycunls.GaussNewtonMinimizer(opts)
        summary = minimizer.minimize(stream, problem)

        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        assert summary.final_cost < 1e-6
        result = cp.asnumpy(states_gpu)
        np.testing.assert_allclose(result, target, atol=1e-3)


class TestLevenbergMarquardtMinimizer:
    """LM convergence, summary field validation, and robust-loss integration."""
    def test_converges(self, stream):
        problem, states_gpu, target = _make_prior_problem()

        lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
        lm_opts.base_options.max_num_iterations = 20
        lm_opts.base_options.disable_safety_checks = False
        minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
        summary = minimizer.minimize(stream, problem)

        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        assert summary.final_cost < 1e-6
        result = cp.asnumpy(states_gpu)
        np.testing.assert_allclose(result, target, atol=1e-3)

    @pytest.mark.parametrize(
        "scaling",
        [
            pycunls.ColumnScaling.hessian_diagonal,
            pycunls.ColumnScaling.jacobian_column_norm,
        ],
    )
    def test_converges_with_column_scaling(self, stream, scaling):
        problem, states_gpu, target = _make_prior_problem()

        lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
        lm_opts.base_options.max_num_iterations = 20
        lm_opts.base_options.column_scaling = scaling
        lm_opts.base_options.disable_safety_checks = False
        minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
        summary = minimizer.minimize(stream, problem)

        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        assert summary.final_cost < 1e-6
        result = cp.asnumpy(states_gpu)
        np.testing.assert_allclose(result, target, atol=1e-3)

    def test_summary_fields(self, stream):
        problem, _, _ = _make_prior_problem()
        minimizer = pycunls.LevenbergMarquardtMinimizer()
        summary = minimizer.minimize(stream, problem)
        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        assert summary.initial_cost > 0
        assert summary.final_cost < summary.initial_cost
        assert len(summary.iteration_costs) > 0
        assert "MinimizerSummary" in repr(summary)

    def test_with_huber_loss(self, stream):
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        states_gpu = cp.asarray(initial)
        obs_gpu = cp.asarray(target)

        sb = pycunls.VectorStateBatch3(states_gpu, 1)
        fb = pycunls.PriorVectorFactorBatch3(obs_gpu, 1)
        loss = pycunls.HuberLossFunctionBatch(1.0)

        problem = pycunls.Problem()
        problem.add_state_batch(sb)
        problem.add_factor_batch(fb, loss, [sb.state_block_device_ptr(0)])
        assert problem.check_consistency()

        minimizer = pycunls.LevenbergMarquardtMinimizer()
        summary = minimizer.minimize(stream, problem)
        cp.cuda.runtime.streamSynchronize(stream.get_stream())

        assert summary.final_cost < summary.initial_cost


class TestMinimizerOptions:
    """Verify default values and mutability of option structs."""
    def test_defaults(self):
        opts = pycunls.MinimizerOptions()
        assert opts.max_num_iterations == 50
        assert opts.state_tolerance == pytest.approx(1e-6)
        assert opts.cost_tolerance == pytest.approx(1e-6)
        assert (opts.sparse_linear_solver_type
                == pycunls.SparseLinearSolverType.BlockSparsePCG)
        assert opts.column_scaling == pycunls.ColumnScaling.none
        assert opts.disable_safety_checks is True

    def test_modification(self):
        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 100
        opts.state_tolerance = 1e-9
        assert opts.max_num_iterations == 100
        assert opts.state_tolerance == pytest.approx(1e-9)

    def test_lm_defaults(self):
        lm = pycunls.LevenbergMarquardtMinimizerOptions()
        assert lm.initial_lambda == pytest.approx(1e-3)
        assert lm.lambda_upscale == pytest.approx(2.0)
