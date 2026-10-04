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

"""Visual-inertial bundle adjustment with ImuFactorBatch.

Keyframes of a synthetic trajectory are joined by IMU factors (the raw
gyro / accelerometer samples between consecutive keyframes) and observe
landmarks through ReprojectionFactorBatch. Both factor types read the same
rig_from_world pose states. Velocities and IMU biases start at zero and are
recovered from the solve; poses and landmarks start perturbed. The data are
noiseless; the solve recovers the trajectory to about a millimeter.

Usage::

    python imu_bundle_adjustment.py [--keyframes K] [--samples-per-keyframe N]
"""

import argparse

import cupy as cp
import numpy as np

import pycunls

RATE_HZ = 200.0  # IMU sample rate
GYRO_BIAS = np.array([0.01, -0.02, 0.015])  # rad/s, constant over the run
ACCEL_BIAS = np.array([0.1, -0.05, 0.08])  # m/s^2


def so3_exp(w):
    """Rodrigues' formula."""
    theta = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if theta < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta**2 * K @ K


def simulate(num_keyframes, samples_per_keyframe, gravity, rng):
    """Ground truth: IMU samples and keyframe states of a smooth trajectory.

    The IMU is integrated with the same Euler model as the factor, so the
    keyframes are exactly consistent with the samples.
    """
    dt = 1.0 / RATE_HZ
    R, v, p = np.eye(3), np.array([0.5, 0.0, 0.0]), np.zeros(3)
    keyframes, samples = [], []
    t = 0.0
    for k in range(num_keyframes):
        keyframes.append((R.copy(), v.copy(), p.copy()))
        if k + 1 == num_keyframes:
            break
        for _ in range(samples_per_keyframe):
            w = np.array([0.3 * np.sin(1.1 * t), 0.2 * np.cos(0.7 * t), 0.4 * np.sin(0.5 * t)])
            a_world = np.array([0.4 * np.cos(0.9 * t), 0.3 * np.sin(1.3 * t), 0.1 * np.sin(t)])
            specific_force = R.T @ (a_world - gravity)  # what an accelerometer measures
            gyro_meas = (w + GYRO_BIAS).astype(np.float32).astype(np.float64)
            accel_meas = (specific_force + ACCEL_BIAS).astype(np.float32).astype(np.float64)
            samples.append(np.concatenate([gyro_meas, accel_meas, [dt]]))
            # Euler step with the measured values minus the true bias.
            acc = R @ (accel_meas - ACCEL_BIAS) + gravity
            p = p + v * dt + 0.5 * acc * dt * dt
            v = v + acc * dt
            R = R @ so3_exp((gyro_meas - GYRO_BIAS) * dt)
            t += dt
    return keyframes, np.array(samples, dtype=np.float32)


def rig_from_world(R, p):
    """4x4 rig_from_world from the rig's world pose (R, p)."""
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R.T, -R.T @ p
    return X


def landmarks_and_observations(poses, rng, per_keyframe=30):
    """Points in front of each keyframe's camera (camera = rig), observed by
    the keyframes within +-2 of it where they have positive depth."""
    points, obs, pairs = [], [], []
    for anchor, X in enumerate(poses):
        world_from_cam = np.linalg.inv(X)
        for _ in range(per_keyframe):
            pc = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(3, 6)])
            pw = world_from_cam[:3, :3] @ pc + world_from_cam[:3, 3]
            idx = len(points)
            points.append(pw)
            for k in range(max(0, anchor - 2), min(len(poses), anchor + 3)):
                c = poses[k][:3, :3] @ pw + poses[k][:3, 3]
                if c[2] > 0.5:
                    obs.append(c[:2] / c[2])
                    pairs.append((k, idx))
    return np.array(points), np.array(obs, dtype=np.float32), pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keyframes", type=int, default=10)
    parser.add_argument("--samples-per-keyframe", type=int, default=40)
    args = parser.parse_args()
    K, N = args.keyframes, args.samples_per_keyframe
    rng = np.random.default_rng(0)

    # --- Sensor model ---
    # Noise densities a few times larger than a navigation-grade spec (the
    # defaults are the EuRoC ADIS16448): with very stiff IMU rows next to
    # vision, a float32 solve converges slowly (see the guide, "Float32
    # conditioning").
    params = pycunls.ImuParameters()  # gravity (0, 0, -9.80665): world +Z up
    params.gyro_noise_density = 1e-2  # rad/s/sqrt(Hz)
    params.accel_noise_density = 1e-1  # m/s^2/sqrt(Hz)
    params.integration_noise_density = 1e-2  # m/sqrt(s)
    params.gyro_bias_random_walk = 1e-2  # rad/s^2/sqrt(Hz)
    params.accel_bias_random_walk = 1e-1  # m/s^3/sqrt(Hz)
    gravity = np.array(params.gravity, dtype=np.float64)

    keyframes, samples = simulate(K, N, gravity, rng)
    true_poses = [rig_from_world(R, p) for R, _, p in keyframes]
    true_points, observations, pairs = landmarks_and_observations(true_poses, rng)
    L = len(true_points)

    # --- States: perturbed poses and points, zero velocities and biases ---
    init_poses = []
    for k, X in enumerate(true_poses):
        Xi = X.copy()
        if k > 0:  # keyframe 0 is fixed (gauge)
            Xi[:3, 3] += rng.normal(0, 0.02, 3)
        init_poses.append(Xi)
    pose_buf = cp.asarray(np.stack(init_poses), dtype=cp.float32)
    vel_buf = cp.zeros((K, 3), dtype=cp.float32)
    bias_buf = cp.zeros((K, 6), dtype=cp.float32)
    point_buf = cp.asarray(true_points + rng.normal(0, 0.05, true_points.shape), dtype=cp.float32)

    fixed = cp.asarray([0], dtype=cp.int32)
    poses = pycunls.SE3StateBatch(pose_buf, K, fixed, 1)
    poses.set_num_active_states(K, 1)
    vels = pycunls.VectorStateBatch3(vel_buf, K)
    vels.set_num_active_states(K)
    biases = pycunls.VectorStateBatch6(bias_buf, K)
    biases.set_num_active_states(K)
    points = pycunls.VectorStateBatch3(point_buf, L)
    points.set_num_active_states(L)

    # --- IMU factors: samples of all keyframe pairs back to back, CSR offsets ---
    imu_samples = cp.asarray(samples, dtype=cp.float32)  # (num_samples, 7)
    offsets = cp.asarray(np.arange(K) * N, dtype=cp.int32)  # K - 1 pairs + 1
    imu = pycunls.ImuFactorBatch(imu_samples, offsets, len(samples), params, K - 1)
    imu.set_num_active_factors(K - 1)
    imu_ptrs = []
    for k in range(K - 1):
        for j in (k, k + 1):  # X_a, v_a, b_a, X_b, v_b, b_b
            imu_ptrs += [poses.state_device_ptr(j), vels.state_device_ptr(j),
                         biases.state_device_ptr(j)]

    # --- Reprojections (normalized image coordinates, sigma = 1e-3) on the same poses ---
    obs_buf = cp.asarray(observations, dtype=cp.float32)
    reproj = pycunls.WeightedFactorBatch(
        pycunls.ReprojectionFactorBatch(obs_buf, len(pairs)), 1e3)
    reproj.set_num_active_factors(len(pairs))
    reproj_ptrs = []
    for k, i in pairs:
        reproj_ptrs += [poses.state_device_ptr(k), points.state_device_ptr(i)]

    problem = pycunls.Problem()
    for s in (poses, vels, biases, points):
        problem.add_state_batch(s)
    problem.add_factor_batch(imu, imu_ptrs)
    problem.add_factor_batch(reproj, reproj_ptrs)

    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 30
    stream = pycunls.CudaStream()
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)

    # --- Compare with the ground truth ---
    est_poses = cp.asnumpy(pose_buf).reshape(K, 4, 4)
    est_vels, est_bias = cp.asnumpy(vel_buf), cp.asnumpy(bias_buf)
    true_vels = np.array([v for _, v, _ in keyframes])
    pos_err = max(np.linalg.norm(np.linalg.inv(est_poses[k])[:3, 3] - keyframes[k][2])
                  for k in range(K))
    print(f"{K} keyframes, {N} IMU samples each, {L} landmarks, {len(pairs)} observations")
    print(f"LM: {summary.num_iterations} iterations, cost {summary.initial_cost:.3g} -> "
          f"{summary.final_cost:.3g}")
    print(f"max position error   {pos_err:.2e} m")
    print(f"max velocity error   {np.abs(est_vels - true_vels).max():.2e} m/s")
    print(f"gyro bias  {est_bias[0, :3]}  (true {GYRO_BIAS})")
    print(f"accel bias {est_bias[0, 3:]}  (true {ACCEL_BIAS})")


if __name__ == "__main__":
    main()
