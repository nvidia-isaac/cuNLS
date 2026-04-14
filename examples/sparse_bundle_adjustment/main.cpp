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

#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "utils/camera_utils.h"
#include "utils/se3_utils.h"
#include "utils/validation.h"

using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;
using cunls::dvector;

namespace {

// Visibility threshold used when generating synthetic points.
constexpr float kMinDepth = 1.0f;
// Reprojection factor guard threshold to avoid unstable divisions near z=0.
constexpr float kZThreshold = 1e-3f;

// Generate random camera poses biased so that the world origin is always
// in front (positive tz ~ 8). This keeps synthetic points visible.
void GenerateRandomPoses(size_t num_poses, std::vector<SE3Transform>& poses) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> rot_dist(-0.2f, 0.2f);
  std::uniform_real_distribution<float> trans_dist(-1.0f, 1.0f);

  std::vector<Vector<6>> twists(num_poses);
  for (size_t i = 0; i < num_poses; ++i) {
    twists[i][0] = rot_dist(rng);
    twists[i][1] = rot_dist(rng);
    twists[i][2] = rot_dist(rng);
    twists[i][3] = trans_dist(rng);
    twists[i][4] = trans_dist(rng);
    twists[i][5] = 8.0f + trans_dist(rng);
  }

  examples::TwistsToSE3(twists, poses);
}

}  // namespace

int main() {
  try {
    // Problem dimensions:
    // - num_poses camera states (first is fixed, rest optimized)
    // - num_points 3D landmarks (all optimized)
    // - one reprojection factor per (pose, point) pair
    const size_t num_poses = 6;
    const size_t num_points = 800;
    const size_t num_observations = num_poses * num_points;

    // Generate synthetic ground-truth camera poses.
    std::vector<SE3Transform> gt_poses;
    GenerateRandomPoses(num_poses, gt_poses);

    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> point_dist(-3.0f, 3.0f);
    std::uniform_real_distribution<float> noise_dist(-0.35f, 0.35f);

    // Generate points that are visible from every camera and create noisy
    // initial points for optimization.
    std::vector<Vector<3>> gt_points(num_points);
    std::vector<Vector<3>> initial_points(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      bool valid = false;
      while (!valid) {
        Vector<3> p;
        p[0] = point_dist(rng);
        p[1] = point_dist(rng);
        p[2] = point_dist(rng);
        valid = true;
        for (const auto& pose : gt_poses) {
          if (examples::ComputeDepth(pose, p) < kMinDepth) {
            valid = false;
            break;
          }
        }
        if (valid) {
          gt_points[i] = p;
        }
      }
      initial_points[i] = gt_points[i];
      initial_points[i][0] += noise_dist(rng);
      initial_points[i][1] += noise_dist(rng);
      initial_points[i][2] += noise_dist(rng);
    }

    // Build normalized 2D observations from ground-truth poses and points.
    std::vector<Vector<2>> observations(num_observations);
    for (size_t pose_idx = 0; pose_idx < num_poses; ++pose_idx) {
      for (size_t point_idx = 0; point_idx < num_points; ++point_idx) {
        const size_t obs_idx = pose_idx * num_points + point_idx;
        observations[obs_idx] =
            examples::ProjectNormalized(gt_poses[pose_idx], gt_points[point_idx]);
      }
    }

    // Perturb poses T_1...T_{M-1}; T_0 stays at ground truth as the gauge
    // anchor. Small perturbations keep all points at positive depth.
    const size_t num_perturbed = num_poses - 1;
    std::vector<SE3Transform> perturbations;
    examples::GenerateRandomSE3(num_perturbed, rng, perturbations, 0.02f, 0.1f);

    std::vector<SE3Transform> initial_poses(num_poses);
    initial_poses[0] = gt_poses[0];
    for (size_t i = 0; i < num_perturbed; ++i) {
      initial_poses[i + 1] = examples::ComposeSE3(perturbations[i], gt_poses[i + 1]);
    }

    // Upload all data to device.
    dvector<SE3Transform> poses_device(initial_poses);
    dvector<Vector<3>> points_device(initial_points);
    dvector<Vector<2>> observations_device(observations);

    // Mark only the first camera pose as constant (gauge anchor).
    std::vector<int> const_pose_ids = {0};
    dvector<int> const_pose_ids_device(const_pose_ids);

    const float* poses_ptr =
        reinterpret_cast<const float*>(poses_device.data());
    const float* points_ptr =
        reinterpret_cast<const float*>(points_device.data());

    // Build state batches and factor batch.
    cunls::cuBLASHandle cublas_handle;
    cunls::SE3StateBatch pose_states(cublas_handle, poses_ptr, num_poses,
                                     const_pose_ids_device.data(), 1);
    cunls::VectorStateBatch<3> point_states(points_ptr, num_points);

    cunls::ReprojectionFactorBatch reproj_factor(observations_device.data(),
                                                 num_observations, kZThreshold);

    // Flatten factor connectivity:
    // for each observation, provide [pose_ptr, point_ptr].
    std::vector<float*> state_pointers;
    state_pointers.reserve(2 * num_observations);
    for (size_t pose_idx = 0; pose_idx < num_poses; ++pose_idx) {
      for (size_t point_idx = 0; point_idx < num_points; ++point_idx) {
        state_pointers.push_back(pose_states.StateBlockDevicePtr(pose_idx));
        state_pointers.push_back(point_states.StateBlockDevicePtr(point_idx));
      }
    }

    // Assemble and validate problem.
    cunls::Problem problem;
    problem.AddStateBatch(&pose_states);
    problem.AddStateBatch(&point_states);
    problem.AddFactorBatch(&reproj_factor, state_pointers);
    if (!problem.CheckConsistency()) {
      std::cerr << "Problem consistency check failed\n";
      return 1;
    }

    // LM configuration.
    cunls::MinimizerOptions options;
    options.max_num_iterations = 80;
    options.state_tolerance = 1e-8f;
    options.cost_tolerance = 1e-8f;

    cunls::LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    lm_options.initial_lambda = 1e-3f;
    cunls::LevenbergMarquardtMinimizer minimizer(lm_options);

    // Run solver and synchronize before reading back results.
    cunls::CudaStream stream;
    const auto summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    // Download optimized poses and points, compare against ground truth.
    std::vector<SE3Transform> optimized_poses(num_poses);
    poses_device.CopyToHost(optimized_poses.data(), num_poses);

    std::vector<Vector<3>> optimized_points(num_points);
    points_device.CopyToHost(optimized_points.data(), num_points);

    const float initial_point_mse =
        examples::ComputeVectorMSE(initial_points, gt_points);
    const float final_point_mse =
        examples::ComputeVectorMSE(optimized_points, gt_points);
    const float initial_pose_mse =
        examples::ComputePoseMSE(initial_poses, gt_poses);
    const float final_pose_mse =
        examples::ComputePoseMSE(optimized_poses, gt_poses);

    std::cout << "Sparse Bundle Adjustment Example\n";
    std::cout << "  Initial cost: " << summary.initial_cost << "\n";
    std::cout << "  Final cost:   " << summary.final_cost << "\n";
    std::cout << "  Iterations:   " << summary.num_iterations << "\n";
    std::cout << "  Point MSE:    " << initial_point_mse << " -> "
              << final_point_mse << "\n";
    std::cout << "  Pose MSE:     " << initial_pose_mse << " -> "
              << final_pose_mse << "\n";

    if (summary.final_cost > 1e-3f || final_point_mse > initial_point_mse * 0.05f) {
      std::cerr << "Optimization quality check failed.\n";
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
