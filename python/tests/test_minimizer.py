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

Most tests share ``_make_prior_problem()``, a one-factor, one-state problem: minimise
``||x - target||^2`` with ``x ∈ R^3``. This is a convex quadratic, so both Gauss-Newton and LM
converge in very few iterations (often 1), making it a fast correctness check for the full
Python → C++ → GPU round-trip.
"""

import numpy as np
import pytest
import warp as wp

import pycunls

TARGET = np.array([1.0, 2.0, 3.0], dtype=np.float32)


def _make_prior_problem(loss=None, **add_factor_options):
    """Build a trivial 3D vector-prior problem: minimise ||x - TARGET||^2 from x = 0.

    Returns ``(problem, states_gpu)``: the Warp state array holds the solution after solving.
    ``loss`` and ``add_factor_options`` (e.g. ``jacobian_mode_override``) go to
    ``add_factor_batch``.
    """
    states_gpu = wp.zeros(3, dtype=wp.float32)
    obs_gpu = wp.array(TARGET)
    sb = pycunls.VectorStateBatch3(states_gpu, 1)
    sb.set_num_active_states(sb.capacity, sb.const_capacity)
    fb = pycunls.PriorVectorFactorBatch3(obs_gpu, 1)
    fb.set_num_active_factors(fb.capacity)
    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    if loss is None:
        problem.add_factor_batch(fb, [sb.state_device_ptr(0)], **add_factor_options)
    else:
        problem.add_factor_batch(fb, loss, [sb.state_device_ptr(0)], **add_factor_options)
    assert problem.check_consistency()
    return problem, states_gpu


def _minimizer(kind, **base_options):
    """A Gauss-Newton ("gn") or LM ("lm") minimizer with the given MinimizerOptions fields."""
    opts = pycunls.MinimizerOptions()
    opts.max_num_iterations = 20
    opts.disable_safety_checks = False
    for name, value in base_options.items():
        setattr(opts, name, value)
    if kind == "gn":
        return pycunls.GaussNewtonMinimizer(opts)
    lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
    lm_opts.base_options = opts
    return pycunls.LevenbergMarquardtMinimizer(lm_opts)


@pytest.mark.parametrize("kind", ["gn", "lm"])
@pytest.mark.parametrize("scaling", [pycunls.ColumnScaling.none,
                                     pycunls.ColumnScaling.hessian_diagonal])
def test_converges(stream, kind, scaling):
    problem, states_gpu = _make_prior_problem()
    summary = _minimizer(kind, column_scaling=scaling).minimize(stream, problem)
    wp.synchronize()
    assert summary.initial_cost > 0
    assert summary.final_cost < 1e-6
    assert len(summary.iteration_costs) > 0
    assert "MinimizerSummary" in repr(summary)
    np.testing.assert_allclose(states_gpu.numpy(), TARGET, atol=1e-3)


def test_lm_with_huber_loss(stream):
    problem, _ = _make_prior_problem(loss=pycunls.HuberLossFunctionBatch(1.0))
    summary = pycunls.LevenbergMarquardtMinimizer().minimize(stream, problem)
    wp.synchronize()
    assert summary.final_cost < summary.initial_cost


class TestMinimizerOptions:
    """Verify default values and mutability of option structs."""
    def test_defaults(self):
        opts = pycunls.MinimizerOptions()
        assert opts.max_num_iterations == 50
        assert opts.state_tolerance == pytest.approx(1e-6)
        assert opts.cost_tolerance == pytest.approx(1e-6)
        assert (opts.sparse_linear_solver_type
                == pycunls.SparseLinearSolverType.BlockSparseCholesky)
        assert opts.column_scaling == pycunls.ColumnScaling.none
        assert opts.jacobian_mode == pycunls.JacobianMode.analytic
        assert opts.numeric_diff_options.method == pycunls.NumericDiffMethod.central
        assert opts.numeric_diff_options.relative_step_size == pytest.approx(1e-4)
        assert opts.disable_safety_checks is True

    def test_modification(self):
        opts = pycunls.MinimizerOptions()
        opts.max_num_iterations = 100
        opts.state_tolerance = 1e-9
        opts.jacobian_mode = pycunls.JacobianMode.numeric
        assert opts.max_num_iterations == 100
        assert opts.state_tolerance == pytest.approx(1e-9)
        assert opts.jacobian_mode == pycunls.JacobianMode.numeric

    def test_lm_defaults(self):
        lm = pycunls.LevenbergMarquardtMinimizerOptions()
        assert lm.initial_lambda == pytest.approx(1e-3)
        assert lm.lambda_upscale == pytest.approx(2.0)


@pytest.mark.parametrize("where", ["global_default", "per_group_override"])
def test_numeric_jacobians_converge(stream, where):
    """Finite-difference Jacobians, set for every group in MinimizerOptions or for one group
    through the jacobian_mode_override of add_factor_batch."""
    if where == "global_default":
        problem, states_gpu = _make_prior_problem()
        minimizer = _minimizer("gn", jacobian_mode=pycunls.JacobianMode.numeric)
    else:
        problem, states_gpu = _make_prior_problem(
            jacobian_mode_override=pycunls.JacobianMode.numeric)
        minimizer = _minimizer("gn")
    summary = minimizer.minimize(stream, problem)
    wp.synchronize()
    assert summary.final_cost < 1e-3
    np.testing.assert_allclose(states_gpu.numpy(), TARGET, atol=1e-2)


# ---------------------------------------------------------------------------
# Schur complement linear system
# ---------------------------------------------------------------------------

def _bundle(num_poses=6, num_points=200, seed=0):
    """Cameras along x looking down +z (world_from_camera, 4x4 row-major) at a
    point cloud, one reprojection factor per (pose, point); poses 0 and 1 are
    constant (gauge). Returns the problem, its point batch and the buffers."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4, dtype=np.float32), (num_poses, 1, 1))
    poses[:, 0, 3] = np.linspace(-1.0, 1.0, num_poses)
    points = np.column_stack([rng.uniform(-2, 2, num_points), rng.uniform(-2, 2, num_points),
                              rng.uniform(4, 8, num_points)]).astype(np.float32)
    observations = []
    for k in range(num_poses):
        cam = points - poses[k, :3, 3]
        observations.append(cam[:, :2] / cam[:, 2:3])
    observations = np.concatenate(observations).astype(np.float32)
    initial_poses = poses.copy()
    initial_poses[2:, :3, 3] += rng.normal(0, 0.05, (num_poses - 2, 3)).astype(np.float32)
    initial_points = points + rng.normal(0, 0.1, points.shape).astype(np.float32)

    buffers = {
        "poses": wp.array(initial_poses.reshape(-1)),
        "points": wp.array(initial_points.reshape(-1)),
        "observations": wp.array(observations.reshape(-1)),
        "const_ids": wp.array([0, 1], dtype=wp.int32),
    }
    pose_states = pycunls.SE3StateBatch(buffers["poses"], num_poses, buffers["const_ids"], 2)
    pose_states.set_num_active_states(num_poses, 2)
    point_states = pycunls.VectorStateBatch3(buffers["points"], num_points)
    point_states.set_num_active_states(num_points)
    reprojection = pycunls.ReprojectionFactorBatch(buffers["observations"],
                                                   num_poses * num_points, 1e-3)
    reprojection.set_num_active_factors(num_poses * num_points)
    pointers = []
    for k in range(num_poses):
        for i in range(num_points):
            pointers += [pose_states.state_device_ptr(k), point_states.state_device_ptr(i)]
    problem = pycunls.Problem()
    problem.add_state_batch(pose_states)
    problem.add_state_batch(point_states)
    problem.add_factor_batch(reprojection, pointers)
    buffers["batches"] = (pose_states, point_states, reprojection)
    return problem, point_states, buffers


class TestSchurComplement:
    """LinearSystemType.SchurComplement reaches the assembled system's minimum."""

    def test_matches_assembled(self, stream):
        costs = []
        for system in (pycunls.LinearSystemType.Assembled,
                       pycunls.LinearSystemType.SchurComplement):
            problem, points, _buffers = _bundle()
            problem.set_eliminated_state_batches([points])
            options = pycunls.LevenbergMarquardtMinimizerOptions()
            options.base_options.max_num_iterations = 30
            options.base_options.linear_system = system
            # The exact reference for Assembled; SchurComplement has its own solver.
            options.base_options.sparse_linear_solver_type = (
                pycunls.SparseLinearSolverType.DenseCholesky)
            summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)
            wp.synchronize()
            assert summary.final_cost < 1e-3 * summary.initial_cost
            costs.append(summary.final_cost)
        assert costs[1] == pytest.approx(costs[0], rel=1e-2, abs=1e-6)

    def test_requires_eliminated_batches(self, stream):
        problem, _points, _buffers = _bundle()
        options = pycunls.LevenbergMarquardtMinimizerOptions()
        options.base_options.linear_system = pycunls.LinearSystemType.SchurComplement
        with pytest.raises(ValueError, match="SetEliminatedStateBatches"):
            pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)
