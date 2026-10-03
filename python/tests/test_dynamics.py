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

"""Dynamics factor batches: layout, consistency with a NumPy rollout, and a
Carter driving to a goal under AugmentedLagrangianMinimizer."""

import cupy as cp
import numpy as np
import pytest

import pycunls


def se2(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=np.float32)


def se2_exp(v):
    vx, vy, th = v
    if abs(th) < 1e-9:
        a, b = 1.0, 0.0
    else:
        a, b = np.sin(th) / th, (1 - np.cos(th)) / th
    return se2(a * vx - b * vy, b * vx + a * vy, th).astype(np.float64)


def test_layouts():
    dts = cp.full(10, 0.1, dtype=cp.float32)
    carter = pycunls.SE2DifferentialDriveFactorBatch(dts, 0.125, 0.5, 10)
    assert carter.state_sizes() == [3, 2, 3] and carter.residuals_size == 3
    car = pycunls.SE2KinematicBicycleFactorBatch(dts, 2.75, 10)
    assert car.state_sizes() == [3, 2, 2, 3, 2] and car.residuals_size == 5
    se3_carter = pycunls.SE3DifferentialDriveFactorBatch(dts, 0.125, 0.5, 10)
    assert se3_carter.state_sizes() == [6, 2, 6] and se3_carter.residuals_size == 4
    se3_car = pycunls.SE3KinematicBicycleFactorBatch(dts, 2.75, 10)
    assert se3_car.state_sizes() == [6, 2, 2, 6, 2] and se3_car.residuals_size == 6
    assert pycunls.SE2KinematicsFactorBatch(dts, 10).state_sizes() == [3, 3, 3]
    assert pycunls.SO3KinematicsFactorBatch(dts, 10).state_sizes() == [3, 3, 3]
    assert pycunls.SE3KinematicsFactorBatch(dts, 10).state_sizes() == [6, 6, 6]
    qp = pycunls.QuadrotorParameters()
    qp.inertia = [0.01, 0.01, 0.02]
    assert qp.inertia == pytest.approx([0.01, 0.01, 0.02])
    quad = pycunls.QuadrotorFactorBatch(dts, qp, 10)
    assert quad.state_sizes() == [6, 3, 3, 4, 6, 3, 3] and quad.residuals_size == 12
    contacts = cp.ones(40, dtype=cp.float32)
    feet = cp.zeros(120, dtype=cp.float32)
    legged = pycunls.QuadrupedFactorBatch(dts, contacts, feet, pycunls.QuadrupedParameters(), 10)
    assert legged.state_sizes() == [6, 3, 3, 12, 6, 3, 3]
    with pytest.raises(ValueError, match="positive"):
        bad = pycunls.QuadrotorParameters()
        bad.mass = 0.0
        pycunls.QuadrotorFactorBatch(dts, bad, 10)
    with pytest.raises(ValueError, match="positive"):
        pycunls.SE2DifferentialDriveFactorBatch(dts, 0.0, 0.5, 10)
    with pytest.raises(ValueError, match="positive"):
        pycunls.SE2KinematicBicycleFactorBatch(dts, -1.0, 10)


def test_carter_rollout_is_consistent(stream):
    """Poses from a NumPy rollout of the exact model leave zero residuals."""
    steps, dt, r, b = 20, 0.1, 0.125, 0.5
    rng = np.random.default_rng(0)
    u = rng.uniform(-10, 10, size=(steps, 2)).astype(np.float32)
    poses = [se2(0.3, -0.2, 0.4).astype(np.float64)]
    for k in range(steps):
        v = r * (u[k, 1] + u[k, 0]) / 2
        w = r * (u[k, 1] - u[k, 0]) / b
        poses.append(poses[-1] @ se2_exp(np.array([dt * v, 0.0, dt * w])))
    pose_buf = cp.asarray(np.stack(poses).astype(np.float32))
    u_buf = cp.asarray(u)
    pose_states = pycunls.SE2StateBatch(pose_buf, steps + 1)
    pose_states.set_num_active_states(steps + 1)
    controls = pycunls.VectorStateBatch2(u_buf, steps)
    controls.set_num_active_states(steps)
    dts = cp.full(steps, dt, dtype=cp.float32)
    dynamics = pycunls.SE2DifferentialDriveFactorBatch(dts, r, b, steps)
    dynamics.set_num_active_factors(steps)
    problem = pycunls.Problem()
    problem.add_state_batch(pose_states)
    problem.add_state_batch(controls)
    ptrs = []
    for k in range(steps):
        ptrs += [pose_states.state_device_ptr(k), controls.state_device_ptr(k),
                 pose_states.state_device_ptr(k + 1)]
    problem.add_factor_batch(dynamics, ptrs)
    options = pycunls.MinimizerOptions()
    options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.DenseLDLT
    summary = pycunls.GaussNewtonMinimizer(options).minimize(stream, problem)
    assert summary.initial_cost < 1e-8


@pytest.mark.parametrize("kind", ["gn", "lm"])
def test_carter_reaches_goal(stream, kind):
    steps, dt, umax = 30, 0.1, 25.0
    pose_buf = cp.asarray(np.stack([se2(0, 0, 0)] * (steps + 1)))
    u_buf = cp.zeros((steps, 2), dtype=cp.float32)
    const_ids = cp.asarray([0], dtype=cp.int32)
    pose_states = pycunls.SE2StateBatch(pose_buf, steps + 1, const_ids, 1)
    pose_states.set_num_active_states(steps + 1, 1)
    controls = pycunls.VectorStateBatch2(u_buf, steps)
    controls.set_num_active_states(steps)

    dts = cp.full(steps, dt, dtype=cp.float32)
    dynamics = pycunls.SE2DifferentialDriveFactorBatch(dts, 0.125, 0.5, steps)
    dynamics.set_num_active_factors(steps)
    hard_dynamics = pycunls.ConstraintFactorBatch(dynamics, pycunls.ConstraintKind.Equality)
    goal = cp.asarray(se2(1.5, 0.8, 1.2)[None])
    goal_prior = pycunls.SE2PriorFactorBatch(goal, 1)
    goal_prior.set_num_active_factors(1)
    at_goal = pycunls.ConstraintFactorBatch(goal_prior, pycunls.ConstraintKind.Equality)
    zeros = cp.zeros(steps * 2, dtype=cp.float32)
    effort = pycunls.WeightedFactorBatch(pycunls.PriorVectorFactorBatch2(zeros, steps), 0.1)
    effort.set_num_active_factors(steps)
    lo = cp.full(steps * 2, -umax, dtype=cp.float32)
    hi = cp.full(steps * 2, umax, dtype=cp.float32)
    limits = pycunls.BoundFactorBatch2(lo, hi, steps)
    limits.set_num_active_factors(steps)

    problem = pycunls.Problem()
    problem.add_state_batch(pose_states)
    problem.add_state_batch(controls)
    dyn_ptrs, u_ptrs = [], []
    for k in range(steps):
        dyn_ptrs += [pose_states.state_device_ptr(k), controls.state_device_ptr(k),
                     pose_states.state_device_ptr(k + 1)]
        u_ptrs.append(controls.state_device_ptr(k))
    problem.add_factor_batch(hard_dynamics, dyn_ptrs)
    problem.add_factor_batch(at_goal, [pose_states.state_device_ptr(steps)])
    problem.add_factor_batch(effort, u_ptrs)
    problem.add_factor_batch(limits, u_ptrs)

    options = pycunls.MinimizerOptions()
    options.max_num_iterations = 100
    options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.DenseLDLT
    if kind == "gn":
        inner = pycunls.GaussNewtonMinimizer(options)
    else:
        lm = pycunls.LevenbergMarquardtMinimizerOptions()
        lm.base_options = options
        inner = pycunls.LevenbergMarquardtMinimizer(lm)
    summary = pycunls.AugmentedLagrangianMinimizer(inner).minimize(stream, problem)
    assert summary.status == pycunls.AugmentedLagrangianMinimizerStatus.Converged, summary
    poses = cp.asnumpy(pose_buf)
    np.testing.assert_allclose(poses[-1], se2(1.5, 0.8, 1.2), atol=1e-3)
    assert np.abs(cp.asnumpy(u_buf)).max() <= umax + 1e-3
