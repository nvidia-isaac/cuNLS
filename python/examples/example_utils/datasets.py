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

"""Synthetic problems for the pycunls examples (host-side NumPy data).

Every generator takes a seed and returns a small dataclass of float32 arrays.
The random draws follow a fixed order, so a seed always gives the same data.
"""

from dataclasses import dataclass

import numpy as np

from .se3 import compose_se3, compute_depth, project_normalized, se3_inverse, twist_to_se3


def _random_se3(rng, rot_scale, trans_scale):
    tw = np.zeros(6, dtype=np.float32)
    tw[:3] = rng.uniform(-rot_scale, rot_scale, 3).astype(np.float32)
    tw[3:] = rng.uniform(-trans_scale, trans_scale, 3).astype(np.float32)
    return twist_to_se3(tw)


# ── Sparse bundle adjustment ────────────────────────────────────────────────

@dataclass
class BundleAdjustmentScene:
    gt_poses: np.ndarray         # (P, 4, 4) world -> camera
    initial_poses: np.ndarray    # (P, 4, 4); pose 0 equals the ground truth (gauge)
    gt_points: np.ndarray        # (M, 3)
    initial_points: np.ndarray   # (M, 3)
    observations: np.ndarray     # (P * M, 2) normalized coords, pose-major


def bundle_adjustment_scene(num_poses=6, num_points=800, seed=1234):
    """Cameras around the origin, points visible from all of them, exact observations."""
    rng = np.random.default_rng(seed)
    gt_poses = []
    for _ in range(num_poses):
        twist = np.zeros(6, dtype=np.float32)
        twist[:3] = rng.uniform(-0.2, 0.2, 3).astype(np.float32)
        twist[3:5] = rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        twist[5] = np.array(rng.uniform(7.0, 9.0), dtype=np.float32)
        gt_poses.append(twist_to_se3(twist))

    gt_points = np.empty((num_points, 3), dtype=np.float32)
    count = 0
    while count < num_points:
        p = rng.uniform(-3.0, 3.0, 3).astype(np.float32)
        if all(compute_depth(T, p) > 1.0 for T in gt_poses):
            gt_points[count] = p
            count += 1

    initial_points = gt_points + rng.uniform(-0.35, 0.35, gt_points.shape).astype(np.float32)

    observations = np.empty((num_poses * num_points, 2), dtype=np.float32)
    for pi in range(num_poses):
        for qi in range(num_points):
            observations[pi * num_points + qi] = project_normalized(gt_poses[pi], gt_points[qi])

    initial_poses = np.stack(gt_poses)
    for i in range(1, num_poses):
        delta = np.zeros(6, dtype=np.float32)
        delta[:3] = rng.uniform(-0.02, 0.02, 3).astype(np.float32)
        delta[3:] = rng.uniform(-0.1, 0.1, 3).astype(np.float32)
        initial_poses[i] = compose_se3(twist_to_se3(delta), gt_poses[i])

    return BundleAdjustmentScene(np.stack(gt_poses), initial_poses, gt_points,
                                 initial_points, observations)


# ── Pose graph ──────────────────────────────────────────────────────────────

@dataclass
class PoseChain:
    gt_poses: np.ndarray       # (N, 4, 4)
    initial_poses: np.ndarray  # (N, 4, 4); pose 0 equals the ground truth (gauge)
    deltas: np.ndarray         # (N - 1, 4, 4) relative measurements, T_{i+1} = T_i * inv(delta_i)


def pose_chain(num_poses=201, seed=9012):
    """A chain of SE(3) poses, its relative measurements and a perturbed initial guess."""
    rng = np.random.default_rng(seed)
    anchor = _random_se3(rng, 0.3, 1.0)
    deltas = [_random_se3(rng, 0.3, 1.0) for _ in range(num_poses - 1)]
    gt = [anchor]
    for d in deltas:
        gt.append(compose_se3(gt[-1], se3_inverse(d)))
    initial = [gt[0].copy()]
    for i in range(1, num_poses):
        initial.append(compose_se3(_random_se3(rng, 0.05, 0.3), gt[i]))
    return PoseChain(np.stack(gt), np.stack(initial), np.stack(deltas))


# ── Scalar chains (custom factor / state examples) ──────────────────────────

@dataclass
class ScalarChain:
    gt: np.ndarray            # (N,) ground-truth values
    initial: np.ndarray       # (N,) noisy initial guess
    measurements: np.ndarray  # (N - 1,) consecutive differences of gt


def scalar_chain(num_states=256, seed=121314):
    """Monotonic scalars; measurements are exact differences x_{i+1} - x_i."""
    rng = np.random.default_rng(seed)
    gt = np.zeros(num_states, dtype=np.float32)
    gt[0] = 0.5
    for i in range(1, num_states):
        gt[i] = gt[i - 1] + rng.uniform(0.2, 0.6)
    measurements = np.diff(gt).astype(np.float32)
    initial = gt + rng.uniform(-0.35, 0.35, num_states).astype(np.float32)
    return ScalarChain(gt, initial, measurements)


def positive_chain(num_states=128, seed=314159):
    """Growing positive scalars; measurements are log-ratios log(x_{i+1} / x_i)."""
    rng = np.random.default_rng(seed)
    gt = np.ones(num_states, dtype=np.float32)
    gt[0] = 2.0
    for i in range(1, num_states):
        gt[i] = gt[i - 1] * rng.uniform(1.05, 1.25)
    measurements = np.log(gt[1:] / gt[:-1]).astype(np.float32)
    initial = gt * rng.uniform(0.6, 1.6, num_states).astype(np.float32)
    return ScalarChain(gt, initial, measurements)


# ── PnP with outliers ───────────────────────────────────────────────────────

@dataclass
class PnPScene:
    gt_pose: np.ndarray       # (4, 4) world -> camera
    initial_pose: np.ndarray  # (4, 4)
    points: np.ndarray        # (N, 3) world points (known, constant)
    observations: np.ndarray  # (N, 2) normalized image coordinates
    is_outlier: np.ndarray    # (N,) bool


def pnp_scene(num_points, outlier_ratio, inlier_threshold=0.01, pixel_noise=3e-3, seed=2024):
    """Noisy projections of random points in front of the camera; a fraction
    `outlier_ratio` is replaced by random image points at least four
    thresholds away from the true projection."""
    rng = np.random.default_rng(seed)
    twist = np.concatenate([rng.uniform(-0.3, 0.3, 3),
                            rng.uniform(-1.0, 1.0, 2),
                            [8.0 + rng.uniform(-1.0, 1.0)]]).astype(np.float32)
    gt_pose = twist_to_se3(twist)

    points = np.empty((num_points, 3), dtype=np.float32)
    observations = np.empty((num_points, 2), dtype=np.float32)
    is_outlier = rng.uniform(size=num_points) < outlier_ratio
    for i in range(num_points):
        p = rng.uniform(-3.0, 3.0, 3).astype(np.float32)
        while compute_depth(gt_pose, p) < 1.0:
            p = rng.uniform(-3.0, 3.0, 3).astype(np.float32)
        proj = project_normalized(gt_pose, p)
        if is_outlier[i]:
            obs = rng.uniform(-0.5, 0.5, 2)
            while np.linalg.norm(obs - proj) < 4 * inlier_threshold:
                obs = rng.uniform(-0.5, 0.5, 2)
        else:
            obs = proj + rng.normal(0.0, pixel_noise, 2)
        points[i] = p
        observations[i] = obs

    delta = np.concatenate([rng.uniform(-0.1, 0.1, 3),
                            rng.uniform(-0.3, 0.3, 3)]).astype(np.float32)
    initial_pose = compose_se3(twist_to_se3(delta), gt_pose)
    return PnPScene(gt_pose, initial_pose, points, observations, is_outlier)
