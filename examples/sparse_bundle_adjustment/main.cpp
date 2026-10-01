/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Sparse bundle adjustment: jointly refine camera poses and 3D points from
// their 2D observations. The first camera is held fixed (gauge anchor).

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "utils/datasets.h"
#include "utils/report.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;

int main() {
  try {
    const size_t num_poses = 6;
    const size_t num_points = 800;
    const size_t num_observations = num_poses * num_points;  // every camera sees every point

    // 1. Synthetic data: ground truth, perturbed initial guess, observations.
    const examples::BundleAdjustmentScene scene =
        examples::MakeBundleAdjustmentScene(num_poses, num_points);

    // 2. Upload the initial guess and the observations to the GPU.
    dvector<SE3Transform> poses(scene.initial_poses);
    dvector<Vector<3>> points(scene.initial_points);
    dvector<Vector<2>> observations(scene.observations);
    dvector<int> constant_pose_ids(std::vector<int>{0});  // camera 0 is the gauge anchor

    // 3. State batches wrap the device memory: SE(3) poses (one constant) and 3D points.
    //    Capacity vs. active count. A batch is constructed with its capacity: how many state blocks
    //    (or factors) its bound device buffers hold. The capacity is fixed for the batch's
    //    lifetime; size it once for the largest problem you expect. Right after construction
    //    nothing is active: SetNumStateBlocks / SetNumFactors set the active count, how many of the
    //    first slots the next solve uses (a solve without it throws). The setter is host-only (no
    //    allocation, no device work) and may change the count between solves up to the capacity,
    //    which is what lets a real-time application allocate once and reuse the same buffers every
    //    frame while the problem size changes. This example solves every slot once, so each active
    //    count equals its capacity.
    cunls::cuBLASHandle cublas;
    const size_t poses_capacity = num_poses;  // every slot solved: active = capacity
    const size_t points_capacity = num_points;
    const size_t const_poses_capacity = 1;  // entries of constant_pose_ids
    cunls::SE3StateBatch pose_states(cublas, reinterpret_cast<const float *>(poses.data()),
                                     poses_capacity, constant_pose_ids.data(),
                                     const_poses_capacity);
    cunls::VectorStateBatch<3> point_states(reinterpret_cast<const float *>(points.data()),
                                            points_capacity);
    const size_t num_const_poses = 1;  // active constant ids: the gauge anchor
    pose_states.SetNumStateBlocks(num_poses, num_const_poses);  // active counts
    point_states.SetNumStateBlocks(num_points);

    // 4. One reprojection factor per observation, reading [pose, point].
    //    Capacity (fixed, sizes the buffers) vs. active count (set per solve): see step 3.
    const size_t observations_capacity = num_observations;  // every slot solved
    cunls::ReprojectionFactorBatch reprojection(observations.data(), observations_capacity,
                                                /*z_threshold=*/1e-3f);
    reprojection.SetNumFactors(num_observations);  // active count
    std::vector<float *> state_pointers;
    for (size_t c = 0; c < num_poses; ++c) {
      for (size_t j = 0; j < num_points; ++j) {
        state_pointers.push_back(pose_states.StateBlockDevicePtr(c));
        state_pointers.push_back(point_states.StateBlockDevicePtr(j));
      }
    }

    // 5. The problem: states plus factors with their state pointers.
    cunls::Problem problem;
    problem.AddStateBatch(&pose_states);
    problem.AddStateBatch(&point_states);
    problem.AddFactorBatch(&reprojection, state_pointers);
    if (!problem.CheckConsistency()) {
      std::cerr << "Problem consistency check failed\n";
      return 1;
    }

    // 6. Solve with Levenberg-Marquardt; the result is written into poses / points.
    cunls::LevenbergMarquardtMinimizerOptions options;
    options.base_options.max_num_iterations = 80;
    options.base_options.state_tolerance = 1e-8f;
    options.base_options.cost_tolerance = 1e-8f;
    options.initial_lambda = 1e-3f;
    cunls::LevenbergMarquardtMinimizer minimizer(options);
    cunls::CudaStream stream;
    const cunls::MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    // 7. Read back and compare with the ground truth.
    std::vector<SE3Transform> final_poses(num_poses);
    std::vector<Vector<3>> final_points(num_points);
    poses.CopyToHost(final_poses.data(), num_poses);
    points.CopyToHost(final_points.data(), num_points);
    const float point_mse_before =
        examples::ComputeVectorMSE(scene.initial_points, scene.gt_points);
    const float point_mse_after = examples::ComputeVectorMSE(final_points, scene.gt_points);

    examples::PrintTitle("Sparse Bundle Adjustment Example");
    examples::PrintSummary(summary);
    examples::PrintChange("Point MSE", point_mse_before, point_mse_after);
    examples::PrintChange("Pose MSE", examples::ComputePoseMSE(scene.initial_poses, scene.gt_poses),
                          examples::ComputePoseMSE(final_poses, scene.gt_poses));
    return examples::QualityExitCode(summary.final_cost <= 1e-3f &&
                                     point_mse_after <= point_mse_before * 0.05f);
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
