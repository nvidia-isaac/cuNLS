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

"""Sparse Bundle Adjustment with pycunls.ProblemBuilder (the Python version of
``examples/sparse_bundle_adjustment/main.cpp``).

Jointly refines N SE(3) camera poses and M 3D points from N x M reprojection
factors (normalized image coordinates). Poses are the cameras' world poses
(world_from_camera, the pose convention of cuNLS). Poses 0 and 1 are held
constant to fix the gauge: the global rigid-body transform and the scale.

The points are eliminated with a Schur complement, whose reduced system is
solved with the built-in supernodal Cholesky.
"""

import numpy as np

import pycunls
from example_utils import datasets, metrics, report


def main():
    # 1. Synthetic data: ground-truth cameras and points, a perturbed initial
    #    guess, and the normalized observation of every point in every camera.
    scene = datasets.bundle_adjustment_scene(num_poses=6, num_points=800)
    num_poses, num_points = len(scene.gt_poses), len(scene.gt_points)

    # 2. Two state batches: camera poses (0 and 1 constant: frame and scale
    #    gauge) and 3D points.
    pb = pycunls.ProblemBuilder()
    cameras = pb.add_states(pycunls.SE3StateBatch, scene.initial_poses,  # (N, 4, 4)
                            constant=[0, 1])
    points = pb.add_states(pycunls.VectorStateBatch3, scene.initial_points)  # (M, 3)

    # 3. One reprojection factor per observation. Its two state slots live in
    #    different batches, so connect gives one (handle, indices) pair per slot:
    #    observation k is point pt_ids[k] seen by camera cam_ids[k] (camera-major).
    cam_ids = np.repeat(np.arange(num_poses), num_points)
    pt_ids = np.tile(np.arange(num_points), num_poses)
    pb.add_factors(pycunls.ReprojectionFactorBatch, scene.observations,  # (N * M, 2)
                   connect=[(cameras, cam_ids), (points, pt_ids)])

    # 4. Solve with Levenberg-Marquardt on the Schur complement: the points are
    #    eliminated, the reduced camera system is factored. The elimination is a
    #    setting of the underlying pycunls.Problem (pb.problem; points.batch is
    #    the VectorStateBatch3).
    pb.problem.set_eliminated_state_batches([points.batch])
    result = pb.solve(max_num_iterations=80, state_tolerance=1e-8, cost_tolerance=1e-8,
                      initial_lambda=1e-3,
                      linear_system=pycunls.LinearSystemType.SchurComplement)

    # 5. Results: the solved points as NumPy, compared with the ground truth.
    optimized_points = result[points]
    mse_before = metrics.mse(scene.initial_points, scene.gt_points)
    mse_after = metrics.mse(optimized_points, scene.gt_points)
    report.print_summary("Sparse Bundle Adjustment (pycunls)", result.summary,
                         Point_MSE=f"{mse_before:.6f} -> {mse_after:.6f}")
    print(result.message)
    report.check(result.converged, result.message)
    report.check(mse_after < 0.1 * mse_before, "point error did not decrease")


if __name__ == "__main__":
    main()
