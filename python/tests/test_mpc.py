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

"""pycunls.mpc: closed-loop simulations with the model as the plant (Carter
path tracking, car lane keeping), obstacle avoidance, a batch of quadrotors
flying to waypoints under thrust limits, and a standing quadruped; each with
converged solves and with the real-time budget."""

import cupy as cp
import numpy as np
import pytest

import pycunls
from pycunls import mpc


def se2(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=np.float64)


def se2_exp(vx, vy, th):
    if abs(th) < 1e-9:
        a, b = 1.0, 0.0
    else:
        a, b = np.sin(th) / th, (1 - np.cos(th)) / th
    return se2(a * vx - b * vy, b * vx + a * vy, th)


def heading(T):
    return np.arctan2(T[1, 0], T[0, 0])


R_WHEEL, TRACK = 0.125, 0.5


def carter_plant(T, u, dt):
    """Exact Carter motion over dt with constant wheel speeds."""
    v = R_WHEEL * (u[1] + u[0]) / 2
    w = R_WHEEL * (u[1] - u[0]) / TRACK
    return T @ se2_exp(dt * v, 0.0, dt * w)


def inner(kind):
    """Inner minimizer of the AL loop (None: the Horizon.build default)."""
    if kind == "default":
        return None
    mo = pycunls.MinimizerOptions()
    mo.sparse_linear_solver_type = pycunls.SparseLinearSolverType.BlockTridiagonal
    mo.state_tolerance = 1e-5  # as Horizon.build's default
    if kind == "gn":
        return pycunls.GaussNewtonMinimizer(mo)
    lm = pycunls.LevenbergMarquardtMinimizerOptions()
    lm.base_options = mo
    return pycunls.LevenbergMarquardtMinimizer(lm)


KINDS = ["default", "gn", "lm"]
# Converged solves every step, or the real-time budget (1 outer x 2 inner
# iterations after a converged first solve).
MODES = [None, (1, 2)]


@pytest.mark.parametrize("real_time", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_carter_tracks_lines_closed_loop(stream, kind, real_time):
    B, N, dt, umax = 2, 20, 0.1, 20.0
    h = mpc.Horizon(mpc.Carter(wheel_radius=R_WHEEL, track_width=TRACK), steps=N, dt=dt, batch=B)
    # Trajectory b tracks the line y = 0.5 b at 1 m/s, starting from the origin.
    def reference(t0):
        ref = np.zeros((B, N + 1, 3, 3), np.float32)
        for b in range(B):
            for k in range(N + 1):
                ref[b, k] = se2((t0 + k) * dt * 1.0, 0.5 * b, 0.0)
        return ref
    h.track_pose(reference(0), weight=[1.0, 1.0, 0.5])
    h.control_effort(weight=0.01)
    h.control_rate(weight=0.05)
    h.control_bounds(-umax, umax)
    ctrl = h.build(minimizer=inner(kind), real_time=real_time)
    poses = [se2(0, 0, 0) for _ in range(B)]
    for t in range(60):
        h.pose_reference[...] = cp.asarray(reference(t + 1)[:, 1:])
        u0 = cp.asnumpy(ctrl.step(stream, pose=np.stack(poses).astype(np.float32)))
        assert np.all(np.abs(u0) <= umax + 1e-3)
        poses = [carter_plant(poses[b], u0[b], dt) for b in range(B)]
    for b in range(B):
        assert abs(poses[b][1, 2] - 0.5 * b) < 0.05, f"lateral error, trajectory {b}"
        assert abs(heading(poses[b])) < 0.05
        assert abs(poses[b][0, 2] - 60 * dt) < 0.3


@pytest.mark.parametrize("real_time", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_carter_drives_around_disk_closed_loop(stream, kind, real_time):
    N, dt, radius, margin = 30, 0.1, 0.5, 0.3
    center = np.array([2.0, 0.05])
    h = mpc.Horizon(mpc.Carter(wheel_radius=R_WHEEL, track_width=TRACK), steps=N, dt=dt)
    goal = se2(4.0, 0.0, 0.0)
    h.track_pose(np.broadcast_to(goal, (1, N + 1, 3, 3)).astype(np.float32), weight=[1.0, 1.0, 0.2])
    h.control_effort(weight=0.01)
    h.control_bounds(-30.0, 30.0)
    h.disk_obstacles(np.array([[[center[0], center[1], radius]]], np.float32), margin=margin)
    ctrl = h.build(minimizer=inner(kind), real_time=real_time)
    T, closest = se2(0, 0, 0), np.inf
    for t in range(80):
        u0 = cp.asnumpy(ctrl.step(stream, pose=T[None].astype(np.float32)))[0]
        T = carter_plant(T, u0, dt)
        closest = min(closest, np.hypot(T[0, 2] - center[0], T[1, 2] - center[1]))
    assert closest >= radius + margin - 0.02, f"clearance {closest:.3f}"
    assert np.hypot(T[0, 2] - 4.0, T[1, 2]) < 0.1, "goal not reached"


@pytest.mark.parametrize("real_time", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_car_lane_keeping_closed_loop(stream, kind, real_time):
    N, dt, L = 20, 0.1, 2.75
    h = mpc.Horizon(mpc.Car(wheelbase=L), steps=N, dt=dt)
    speed = 5.0
    def reference(t0):
        return np.stack([se2((t0 + k) * dt * speed, 0.0, 0.0) for k in range(N + 1)])[None]
    h.track_pose(reference(0), weight=[0.5, 2.0, 1.0])
    h.track_vector("speed_steer", np.tile([speed, 0.0], (1, N + 1, 1)), weight=[1.0, 0.1])
    h.control_effort(weight=0.05)
    h.control_bounds([-3.0, -0.5], [3.0, 0.5])
    h.vector_bounds("speed_steer", [0.0, -0.4], [10.0, 0.4])
    ctrl = h.build(minimizer=inner(kind), real_time=real_time)
    # Start 1 m off the lane, at 4 m/s.
    T, z = se2(0.0, 1.0, 0.0), np.array([4.0, 0.0])
    for t in range(60):
        h.pose_reference[...] = cp.asarray(reference(t + 1)[:, 1:].astype(np.float32))
        u = cp.asnumpy(ctrl.step(stream, pose=T[None].astype(np.float32),
                                 speed_steer=z[None].astype(np.float32)))[0]
        # Bounds hold to the AL tolerance scale.
        assert -3.0 - 5e-3 <= u[0] <= 3.0 + 5e-3 and abs(u[1]) <= 0.5 + 5e-3
        for _ in range(10):  # plant: fine Euler substeps of the bicycle model
            v, delta = z
            T = T @ se2_exp(dt / 10 * v, 0.0, dt / 10 * v * np.tan(delta) / L)
            z = z + dt / 10 * u
        assert abs(z[1]) <= 0.4 + 0.06  # steering state bound (plant between samples)
    assert abs(T[1, 2]) < 0.05, "lateral error"
    assert abs(z[0] - speed) < 0.1


def se3(x, y, z):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = (x, y, z)
    return T


def quadrotor_plant(p, T, v, w, f, dt, substeps=20):
    """Fine Euler integration of the quadrotor model (rotation by exact SO(3) steps)."""
    a = p.arm_length / np.sqrt(2)
    tau = np.array([a * (-f[0] + f[1] + f[2] - f[3]), a * (-f[0] + f[1] - f[2] + f[3]),
                    p.torque_coefficient * (-f[0] - f[1] + f[2] + f[3])])
    J = np.array(p.inertia, dtype=np.float64)
    T, v, w = T.astype(np.float64), v.astype(np.float64), w.astype(np.float64)
    h = dt / substeps
    for _ in range(substeps):
        R = T[:3, :3]
        acc = R[:, 2] * f.sum() / p.mass - p.linear_drag * v - np.array([0, 0, p.gravity])
        wdot = (tau - np.cross(w, J * w)) / J
        th = np.linalg.norm(w) * h
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) * h
        dR = np.eye(3) + (np.sin(th) / th if th > 1e-12 else 1.0) * K + \
            ((1 - np.cos(th)) / th**2 if th > 1e-12 else 0.5) * K @ K
        T[:3, 3] += h * v
        T[:3, :3] = R @ dR
        v = v + h * acc
        w = w + h * wdot
    return T, v, w


@pytest.mark.parametrize("real_time", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_quadrotors_fly_to_waypoints_closed_loop(stream, kind, real_time):
    B, N, dt, fmax = 8, 40, 0.025, 8.0
    p = pycunls.QuadrotorParameters()
    h = mpc.Horizon(mpc.Quadrotor(parameters=p), steps=N, dt=dt, batch=B)
    start = np.stack([se3(0, 0, 1) for _ in range(B)])
    targets = np.stack([se3(np.cos(2 * np.pi * b / B), np.sin(2 * np.pi * b / B), 1.5)
                        for b in range(B)])
    h.poses[...] = cp.asarray(np.repeat(start[:, None], N + 1, axis=1))
    h.track_pose(np.repeat(targets[:, None], N + 1, axis=1), weight=[0.5, 0.5, 0.5, 2.0, 2.0, 2.0])
    h.track_vector("velocity", np.zeros((B, N + 1, 3)), weight=0.3)
    hover = p.mass * p.gravity / 4
    h.controls[...] = hover
    h.control_effort(weight=0.05, nominal=np.full((B, N, 4), hover))
    h.control_bounds(0.0, fmax)
    ctrl = h.build(minimizer=inner(kind), real_time=real_time)
    T = [start[b].astype(np.float64) for b in range(B)]
    v = [np.zeros(3) for _ in range(B)]
    w = [np.zeros(3) for _ in range(B)]
    for t in range(120):  # 3 s
        u = cp.asnumpy(ctrl.step(stream, pose=np.stack(T).astype(np.float32),
                                 velocity=np.stack(v).astype(np.float32),
                                 rates=np.stack(w).astype(np.float32)))
        assert u.min() >= -5e-3 and u.max() <= fmax + 5e-3
        for b in range(B):
            T[b], v[b], w[b] = quadrotor_plant(p, T[b], v[b], w[b], u[b].astype(np.float64), dt)
    for b in range(B):
        assert np.linalg.norm(T[b][:3, 3] - targets[b][:3, 3]) < 0.05, f"drone {b}"
        assert np.linalg.norm(v[b]) < 0.05


@pytest.mark.parametrize("real_time", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_quadruped_stands_closed_loop(stream, kind, real_time):
    N, dt = 10, 0.02
    prm = pycunls.QuadrupedParameters()
    h = mpc.Horizon(mpc.Quadruped(parameters=prm), steps=N, dt=dt)
    base = se3(0, 0, 0.4)
    feet = np.array([[0.2, -0.15, 0], [-0.2, 0.15, 0], [0.2, 0.15, 0], [-0.2, -0.15, 0]],
                    np.float32)
    h.foot_positions[...] = cp.asarray(np.broadcast_to(feet, (1, N, 4, 3)))
    h.poses[...] = cp.asarray(base)
    h.track_pose(np.broadcast_to(base, (1, N + 1, 4, 4)), weight=5.0)
    h.track_vector("velocity", np.zeros((1, N + 1, 3)), weight=1.0)
    h.track_vector("rates", np.zeros((1, N + 1, 3)), weight=1.0)
    h.control_effort(weight=1e-3)
    ctrl = h.build(minimizer=inner(kind), real_time=real_time)
    zero = np.zeros((1, 3), np.float32)
    for t in range(5):
        F = cp.asnumpy(ctrl.step(stream, pose=base[None], velocity=zero, rates=zero))
        # The applied forces hold the weight, a quarter per leg, vertically.
        np.testing.assert_allclose(F.reshape(4, 3)[:, 2], prm.mass * prm.gravity / 4, rtol=2e-2)
        assert np.abs(F.reshape(4, 3)[:, :2]).max() < 0.5


def test_horizon_layout_and_shift(stream):
    h = mpc.Horizon(mpc.Car(wheelbase=2.75), steps=5, dt=0.1, batch=3)
    assert h.poses.shape == (3, 6, 3, 3)
    assert h.vectors["speed_steer"].shape == (3, 6, 2)
    assert h.controls.shape == (3, 5, 2)
    ctrl = h.control_bounds(-1.0, 1.0).build()
    h.controls[...] = cp.arange(5, dtype=cp.float32)[None, :, None]
    ctrl.shift()
    np.testing.assert_array_equal(cp.asnumpy(h.controls)[0, :, 0], [1, 2, 3, 4, 4])
    with pytest.raises(ValueError):
        mpc.Horizon(mpc.Carter(), steps=0, dt=0.1)
