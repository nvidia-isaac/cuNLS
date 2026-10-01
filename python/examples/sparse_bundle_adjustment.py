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

"""Sparse Bundle Adjustment with pycunls (Python port of
``examples/sparse_bundle_adjustment/main.cpp``).

Jointly refines N SE(3) camera poses and M 3D points from N x M reprojection
factors (normalized image coordinates). Pose 0 is held constant to fix the
gauge (the global rigid-body transform).
"""

import cupy as cp

import pycunls
from example_utils import datasets, metrics, report


def main():
    # 1. Synthetic data: ground truth, perturbed initial guess, observations.
    scene = datasets.bundle_adjustment_scene(num_poses=6, num_points=800)
    num_poses, num_points = len(scene.gt_poses), len(scene.gt_points)

    # 2. Upload to the GPU (row-major 4x4 poses, xyz points, xy observations).
    poses_gpu = cp.asarray(scene.initial_poses.reshape(-1))
    points_gpu = cp.asarray(scene.initial_points.reshape(-1))
    observations_gpu = cp.asarray(scene.observations.reshape(-1))
    const_ids_gpu = cp.array([0], dtype=cp.int32)  # pose 0 is constant

    # 3. State batches: SE(3) poses (one constant) and 3D points.
    cublas = pycunls.CublasHandle()
    pose_states = pycunls.SE3StateBatch(cublas, poses_gpu, num_poses, const_ids_gpu, 1)
    point_states = pycunls.VectorStateBatch3(points_gpu, num_points)

    # 4. One reprojection factor per (pose, point); each reads [pose, point].
    reprojection = pycunls.ReprojectionFactorBatch(observations_gpu, num_poses * num_points, 1e-3)
    state_pointers = []
    for pi in range(num_poses):
        for qi in range(num_points):
            state_pointers.append(pose_states.state_block_device_ptr(pi))
            state_pointers.append(point_states.state_block_device_ptr(qi))

    # 5. Problem.
    problem = pycunls.Problem()
    problem.add_state_batch(pose_states)
    problem.add_state_batch(point_states)
    problem.add_factor_batch(reprojection, state_pointers)
    assert problem.check_consistency(), "Problem consistency check failed"

    # 6. Solve with Levenberg-Marquardt.
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 80
    options.base_options.state_tolerance = 1e-8
    options.base_options.cost_tolerance = 1e-8
    options.initial_lambda = 1e-3
    stream = pycunls.CudaStream()
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    # 7. Results and checks.
    optimized_points = cp.asnumpy(points_gpu).reshape(-1, 3)
    mse_before = metrics.mse(scene.initial_points, scene.gt_points)
    mse_after = metrics.mse(optimized_points, scene.gt_points)
    report.print_summary("Sparse Bundle Adjustment (pycunls)", summary,
                         Point_MSE=f"{mse_before:.6f} -> {mse_after:.6f}")
    report.check(mse_after < 0.1 * mse_before, "point error did not decrease")


if __name__ == "__main__":
    main()
