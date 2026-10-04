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

"""ImuFactorBatch: layout, parameters, and zero cost on keyframes from a NumPy
integration of the factor's Euler model."""

import cupy as cp
import numpy as np
import pytest

import pycunls


def so3_exp(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + 2 * np.sin(th / 2) ** 2 / th**2 * K @ K


def test_layout_and_parameters():
    samples = cp.zeros(7 * 4, dtype=cp.float32)
    offsets = cp.asarray([0, 2, 4], dtype=cp.int32)
    p = pycunls.ImuParameters()
    assert p.gravity == pytest.approx([0.0, 0.0, -9.80665])
    assert p.body_from_imu == pytest.approx(np.eye(4).ravel().tolist())
    p.gravity = [0.0, 0.0, 9.81]
    assert p.gravity == pytest.approx([0.0, 0.0, 9.81])
    imu = pycunls.ImuFactorBatch(samples, offsets, 4, p, 2)
    assert imu.state_sizes() == [6, 3, 6, 6, 3, 6] and imu.residuals_size == 15
    with pytest.raises(ValueError, match="positive"):
        bad = pycunls.ImuParameters()
        bad.integration_noise_density = 0.0
        pycunls.ImuFactorBatch(samples, offsets, 4, bad, 2)
    with pytest.raises(ValueError):
        p.body_from_imu = [1.0, 0.0]


def test_integrated_keyframes_have_zero_cost(stream):
    """Keyframes from a NumPy Euler integration of the samples leave zero residuals."""
    keyframes, n, dt = 4, 30, 0.005
    p = pycunls.ImuParameters()
    g = np.array(p.gravity, dtype=np.float64)
    rng = np.random.default_rng(1)
    bias = np.array([0.01, -0.02, 0.015, 0.1, -0.05, 0.08])
    R, v, pos = so3_exp(np.array([0.1, -0.3, 0.2])), np.array([1.0, -0.5, 0.2]), np.zeros(3)
    poses, vels, samples = [], [], []
    for k in range(keyframes):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pos
        poses.append(T)
        vels.append(v.copy())
        if k + 1 == keyframes:
            break
        for _ in range(n):
            w = rng.normal(0, 0.7, 3).astype(np.float32).astype(np.float64)
            a = (rng.normal(0, 1, 3) + [0, 0, 9.8]).astype(np.float32).astype(np.float64)
            acc = R @ (a - bias[3:]) + g
            pos = pos + v * dt + 0.5 * acc * dt * dt
            v = v + acc * dt
            R = R @ so3_exp((w - bias[:3]) * dt)
            samples.append(np.concatenate([w, a, [dt]]))
    pose_buf = cp.asarray(np.stack(poses).astype(np.float32))
    vel_buf = cp.asarray(np.stack(vels).astype(np.float32))
    bias_buf = cp.asarray(np.tile(bias, (keyframes, 1)).astype(np.float32))
    first = cp.asarray([0], dtype=cp.int32)  # keyframe 0 fixed (gauge)
    pose_states = pycunls.SE3StateBatch(pose_buf, keyframes, first, 1)
    pose_states.set_num_active_states(keyframes, 1)
    vel_states = pycunls.VectorStateBatch3(vel_buf, keyframes, first, 1)
    vel_states.set_num_active_states(keyframes, 1)
    bias_states = pycunls.VectorStateBatch6(bias_buf, keyframes, first, 1)
    bias_states.set_num_active_states(keyframes, 1)

    sample_buf = cp.asarray(np.stack(samples).astype(np.float32))
    offsets = cp.asarray(np.arange(keyframes) * n, dtype=cp.int32)
    imu = pycunls.ImuFactorBatch(sample_buf, offsets, len(samples), p, keyframes - 1)
    imu.set_num_active_factors(keyframes - 1)
    problem = pycunls.Problem()
    for s in (pose_states, vel_states, bias_states):
        problem.add_state_batch(s)
    ptrs = []
    for k in range(keyframes - 1):
        for j in (k, k + 1):
            ptrs += [pose_states.state_device_ptr(j), vel_states.state_device_ptr(j),
                     bias_states.state_device_ptr(j)]
    problem.add_factor_batch(imu, ptrs)
    options = pycunls.MinimizerOptions()
    options.max_num_iterations = 1
    options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.DenseLDLT
    summary = pycunls.GaussNewtonMinimizer(options).minimize(stream, problem)
    assert summary.initial_cost < 1e-2
