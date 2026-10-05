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
factors (normalized image coordinates). Poses are the cameras' world poses
(world_from_camera, the pose convention of cuNLS). Poses 0 and 1 are held
constant to fix the gauge: the global rigid-body transform and the scale.
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
    # Gauge anchors: pose 0 fixes the frame, pose 1 the scale (unobservable from
    # reprojections alone).
    const_ids_gpu = cp.array([0, 1], dtype=cp.int32)

    # 3. State batches: SE(3) poses (two constant) and 3D points.
    #    Capacity vs. active count. A batch is constructed with its capacity: how many states (or
    #    factors) its bound device buffers hold. The capacity is fixed for the batch's lifetime;
    #    size it once for the largest problem you expect. Right after construction nothing is
    #    active: set_num_active_states / set_num_active_factors set the active count, how many of
    #    the first slots the next solve uses (a solve without it throws). The setter is host-only
    #    (no allocation, no device work) and may change the count between solves up to the capacity,
    #    which is what lets a real-time application allocate once and reuse the same buffers every
    #    frame while the problem size changes. This example solves every slot once, so each active
    #    count equals its capacity.
    poses_capacity = num_poses  # every slot solved: active = capacity
    points_capacity = num_points
    const_poses_capacity = 2  # entries of const_ids_gpu
    pose_states = pycunls.SE3StateBatch(poses_gpu, poses_capacity, const_ids_gpu,
                                        const_poses_capacity)
    point_states = pycunls.VectorStateBatch3(points_gpu, points_capacity)
    num_const_poses = 2  # active constant ids: the gauge anchors
    pose_states.set_num_active_states(num_poses, num_const_poses)  # active counts
    point_states.set_num_active_states(num_points)

    # 4. One reprojection factor per (pose, point); each reads [pose, point].
    #    Capacity (fixed, sizes the buffers) vs. active count (set per solve): see step 3.
    num_observations = num_poses * num_points
    observations_capacity = num_observations  # every slot solved
    reprojection = pycunls.ReprojectionFactorBatch(observations_gpu, observations_capacity, 1e-3)
    reprojection.set_num_active_factors(num_observations)  # active count
    state_pointers = []
    for pi in range(num_poses):
        for qi in range(num_points):
            state_pointers.append(pose_states.state_device_ptr(pi))
            state_pointers.append(point_states.state_device_ptr(qi))

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
