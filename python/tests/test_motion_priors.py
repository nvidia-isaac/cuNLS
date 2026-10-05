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

"""Motion priors (constant velocity / acceleration): layouts, and an SE(3)
chain recovered from perturbed poses and velocities with the closed-form
process-noise information."""

import cupy as cp
import numpy as np
import pytest

import pycunls


def se3_exp(xi):
    """SE(3) Exp of [phi; rho] (row-major 4x4): R = Exp(phi), t = J_l(phi) rho."""
    phi, rho = xi[:3], xi[3:]
    th = np.linalg.norm(phi)
    K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
    if th < 1e-9:
        R, J = np.eye(3) + K, np.eye(3) + 0.5 * K
    else:
        R = np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K
        J = np.eye(3) + (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, J @ rho
    return T


@pytest.mark.parametrize("group,tangent", [("SE3", 6), ("SO3", 3), ("SE2", 3), ("SO2", 1)])
def test_layouts(stream, group, tangent):
    dts = cp.full(4, 0.1, dtype=cp.float32)
    qc = cp.ones(tangent, dtype=cp.float32)
    cv = getattr(pycunls, f"ConstantVelocity{group}FactorBatch")(dts, 4)
    assert cv.state_sizes() == [tangent] * 4 and cv.residuals_size == 2 * tangent
    ca = getattr(pycunls, f"ConstantAcceleration{group}FactorBatch")(dts, 4)
    assert ca.state_sizes() == [tangent] * 6 and ca.residuals_size == 3 * tangent
    cvi = getattr(pycunls, f"ConstantVelocityInformation{group}FactorBatch")(stream, dts, qc, 4)
    assert cvi.state_sizes() == [tangent] * 4
    cai = getattr(pycunls, f"ConstantAccelerationInformation{group}FactorBatch")(stream, dts, qc, 4)
    assert cai.state_sizes() == [tangent] * 6
    cvi.update(stream, 4)
    with pytest.raises(ValueError):
        cvi.update(stream, 5)  # beyond the capacity
    with pytest.raises(TypeError, match="time_steps must have dtype float32"):
        getattr(pycunls, f"ConstantVelocity{group}FactorBatch")(cp.zeros(4), 4)


def test_constant_velocity_se3_chain(stream):
    """A chain of poses moving with a constant body twist (exact for the prior:
    J_l(dt v) v = v) is recovered from perturbed poses and velocities; pose 0
    and velocity 0 fix the gauge."""
    n, dt = 30, 0.1
    v = np.array([0.1, -0.2, 0.3, 1.0, 0.2, -0.1])
    poses = [se3_exp(np.array([0.2, 0.1, -0.3, 1.0, 2.0, 0.5]))]
    for _ in range(n - 1):
        poses.append(poses[-1] @ se3_exp(dt * v))
    rng = np.random.default_rng(0)
    init = [poses[0]] + [T @ se3_exp(rng.normal(0, 0.03, 6)) for T in poses[1:]]
    vel0 = np.tile(v, (n, 1))
    vel0[1:] += rng.normal(0, 0.1, (n - 1, 6))

    pose_buf = cp.asarray(np.stack(init), dtype=cp.float32)
    vel_buf = cp.asarray(vel0, dtype=cp.float32)
    const = cp.asarray([0], dtype=cp.int32)
    poses_sb = pycunls.SE3StateBatch(pose_buf, n, const, 1)
    poses_sb.set_num_active_states(n, 1)
    vels_sb = pycunls.VectorStateBatch6(vel_buf, n, const, 1)
    vels_sb.set_num_active_states(n, 1)
    dts = cp.full(n - 1, dt, dtype=cp.float32)
    qc = cp.ones(6, dtype=cp.float32)
    prior = pycunls.ConstantVelocityInformationSE3FactorBatch(stream, dts, qc, n - 1)
    prior.set_num_active_factors(n - 1)
    ptrs = []
    for k in range(n - 1):
        ptrs += [poses_sb.state_device_ptr(k), poses_sb.state_device_ptr(k + 1),
                 vels_sb.state_device_ptr(k), vels_sb.state_device_ptr(k + 1)]
    problem = pycunls.Problem()
    problem.add_state_batch(poses_sb)
    problem.add_state_batch(vels_sb)
    problem.add_factor_batch(prior, ptrs)
    assert problem.check_consistency()
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 100
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)
    assert summary.final_cost < 1e-6 * summary.initial_cost
    est = cp.asnumpy(pose_buf).reshape(n, 4, 4)
    # float32 round-off accumulates along the chain (anchored at one end): a few
    # millimeters after 30 steps (~3 m).
    assert np.abs(est[:, :3, 3] - np.stack(poses)[:, :3, 3]).max() < 1e-2
    np.testing.assert_allclose(cp.asnumpy(vel_buf), np.tile(v, (n, 1)), atol=1e-2)
