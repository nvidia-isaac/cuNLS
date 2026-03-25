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

"""Sparse Bundle Adjustment using pycunls with CuPy arrays.

This is a Python port of ``examples/sparse_bundle_adjustment/main.cpp``.
It generates a synthetic SBA problem (cameras + 3D points), adds noise,
and recovers the ground-truth using Levenberg-Marquardt.

Problem structure
-----------------
* **States**: N SE(3) camera poses (ambient=16, tangent=6) + M 3D landmark
  points (ambient=3, tangent=3).
* **Factors**: N×M reprojection factors, each connecting one pose and one
  point.  The residual is the difference between the observed normalised
  image coordinate and the projection of the point through the pose.
* **Gauge fix**: Pose 0 is held constant via ``const_ids`` to remove the
  gauge freedom (global rigid-body transform).
"""

import numpy as np
import cupy as cp

import pycunls
from se3_utils import (
    twist_to_se3, compose_se3, project_normalized, compute_depth,
)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    num_poses = 6
    num_points = 800
    num_observations = num_poses * num_points
    z_threshold = 1e-3

    rng = np.random.default_rng(1234)

    # Ground-truth poses: cameras looking roughly at the origin.
    gt_poses = []
    for _ in range(num_poses):
        twist = np.zeros(6, dtype=np.float32)
        twist[:3] = rng.uniform(-0.2, 0.2, 3).astype(np.float32)
        twist[3:5] = rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        twist[5] = np.array(rng.uniform(7.0, 9.0), dtype=np.float32)
        gt_poses.append(twist_to_se3(twist))

    # Ground-truth 3D points visible from all cameras.
    gt_points = np.empty((num_points, 3), dtype=np.float32)
    count = 0
    while count < num_points:
        p = rng.uniform(-3.0, 3.0, 3).astype(np.float32)
        if all(compute_depth(T, p) > 1.0 for T in gt_poses):
            gt_points[count] = p
            count += 1

    # Noisy initial points.
    initial_points = gt_points + rng.uniform(-0.35, 0.35, gt_points.shape).astype(np.float32)

    # Observations in normalized image coordinates.
    observations = np.empty((num_observations, 2), dtype=np.float32)
    for pi in range(num_poses):
        for qi in range(num_points):
            observations[pi * num_points + qi] = project_normalized(gt_poses[pi], gt_points[qi])

    # Perturb poses 1..N-1; pose 0 is the gauge anchor.
    initial_poses_np = np.stack(gt_poses)  # (num_poses, 4, 4)
    for i in range(1, num_poses):
        delta = np.zeros(6, dtype=np.float32)
        delta[:3] = rng.uniform(-0.02, 0.02, 3).astype(np.float32)
        delta[3:] = rng.uniform(-0.1, 0.1, 3).astype(np.float32)
        initial_poses_np[i] = compose_se3(twist_to_se3(delta), gt_poses[i])

    # ── Upload to GPU ───────────────────────────────────────────────────────
    # cuNLS expects row-major 4x4 matrices stored as 16 contiguous floats.
    poses_gpu = cp.asarray(initial_poses_np.reshape(-1).astype(np.float32))
    points_gpu = cp.asarray(initial_points.reshape(-1).astype(np.float32))
    observations_gpu = cp.asarray(observations.reshape(-1).astype(np.float32))
    const_ids_gpu = cp.array([0], dtype=cp.int32)

    # ── Build cuNLS problem ─────────────────────────────────────────────────
    cublas = pycunls.CublasHandle()
    stream = pycunls.CudaStream()

    pose_states = pycunls.SE3StateBatch(
        cublas, poses_gpu, num_poses, const_ids_gpu, 1)
    point_states = pycunls.VectorStateBatch3(points_gpu, num_points)

    reproj_factor = pycunls.ReprojectionFactorBatch(
        cublas, observations_gpu, num_observations, z_threshold)

    state_pointers = []
    for pi in range(num_poses):
        for qi in range(num_points):
            state_pointers.append(pose_states.state_block_device_ptr(pi))
            state_pointers.append(point_states.state_block_device_ptr(qi))

    problem = pycunls.Problem()
    problem.add_state_batch(pose_states)
    problem.add_state_batch(point_states)
    problem.add_factor_batch(reproj_factor, state_pointers)
    assert problem.check_consistency(), "Problem consistency check failed"

    # ── Solve ───────────────────────────────────────────────────────────────
    opts = pycunls.MinimizerOptions()
    opts.max_num_iterations = 80
    opts.state_tolerance = 1e-8
    opts.cost_tolerance = 1e-8

    lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
    lm_opts.base_options = opts
    lm_opts.initial_lambda = 1e-3

    minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
    summary = minimizer.minimize(stream, problem)

    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    # ── Report ──────────────────────────────────────────────────────────────
    optimized_points = cp.asnumpy(points_gpu).reshape(-1, 3)
    point_mse_before = float(np.mean((initial_points - gt_points) ** 2))
    point_mse_after = float(np.mean((optimized_points - gt_points) ** 2))

    print("Sparse Bundle Adjustment (pycunls)")
    print(f"  Initial cost : {summary.initial_cost:.6f}")
    print(f"  Final cost   : {summary.final_cost:.6f}")
    print(f"  Iterations   : {summary.num_iterations}")
    print(f"  Point MSE    : {point_mse_before:.6f} -> {point_mse_after:.6f}")


if __name__ == "__main__":
    main()
