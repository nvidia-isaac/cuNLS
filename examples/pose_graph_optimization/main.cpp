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
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "utils/se3_utils.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;

int main() {
  try {
    // A chain of poses connected by between constraints:
    //   T_0 -> T_1 -> T_2 -> ... -> T_{N-1}
    // The first pose T_0 is held fixed as the gauge anchor.
    const size_t num_poses = 201;
    const size_t num_constraints = num_poses - 1;

    std::mt19937 rng(9012);

    // Synthetic problem ingredients:
    // - A random anchor pose T_0
    // - Random relative transforms (deltas) between consecutive poses
    std::vector<SE3Transform> anchor_pose;
    std::vector<SE3Transform> deltas;
    examples::GenerateRandomSE3(1, rng, anchor_pose);
    examples::GenerateRandomSE3(num_constraints, rng, deltas);

    // Build ground-truth chain that satisfies every constraint:
    // delta * T_i^{-1} * T_{i+1} = I  =>  T_{i+1} = T_i * delta^{-1}
    std::vector<SE3Transform> gt_poses(num_poses);
    gt_poses[0] = anchor_pose[0];
    for (size_t i = 0; i < num_constraints; ++i) {
      gt_poses[i + 1] =
          examples::ComposeSE3(gt_poses[i], examples::InverseSE3(deltas[i]));
    }

    // Disturb all poses except the fixed anchor T_0. Small perturbations keep
    // the initial estimate within the convergence basin of the SE(3) log map.
    std::vector<SE3Transform> disturbance;
    examples::GenerateRandomSE3(num_constraints, rng, disturbance, 0.05f, 0.3f);
    std::vector<SE3Transform> initial_poses(num_poses);
    initial_poses[0] = gt_poses[0];
    for (size_t i = 0; i < num_constraints; ++i) {
      initial_poses[i + 1] =
          examples::ComposeSE3(disturbance[i], gt_poses[i + 1]);
    }

    // Copy host data to GPU.
    dvector<SE3Transform> poses_device(initial_poses);
    dvector<SE3Transform> deltas_device(deltas);

    // Mark the first pose T_0 as constant (gauge anchor).
    std::vector<int> const_ids = {0};
    dvector<int> const_ids_device(const_ids);

    const float* poses_ptr =
        reinterpret_cast<const float*>(poses_device.data());

    // Single state batch for the entire chain, with T_0 fixed.
    cunls::cuBLASHandle cublas_handle;
    cunls::SE3StateBatch pose_states(cublas_handle, poses_ptr, num_poses,
                                     const_ids_device.data(), 1);

    // Build SE(3) between factor batch for consecutive constraints.
    cunls::SE3BetweenFactorBatch between_factor(cublas_handle, deltas_device.data(),
                                                num_constraints);

    // Flatten factor-to-state connectivity:
    // [T_0, T_1, T_1, T_2, ..., T_{N-2}, T_{N-1}]
    std::vector<float*> state_pointers;
    state_pointers.reserve(2 * num_constraints);
    for (size_t i = 0; i < num_constraints; ++i) {
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i));
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i + 1));
    }

    // Assemble optimization problem.
    cunls::Problem problem;
    problem.AddStateBatch(&pose_states);
    problem.AddFactorBatch(&between_factor, state_pointers);
    if (!problem.CheckConsistency()) {
      std::cerr << "Problem consistency check failed\n";
      return 1;
    }

    // LM configuration.
    cunls::MinimizerOptions options;
    options.max_num_iterations = 60;
    options.state_tolerance = 1e-8f;
    options.cost_tolerance = 1e-8f;

    cunls::LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    lm_options.initial_lambda = 1e-3f;
    cunls::LevenbergMarquardtMinimizer minimizer(lm_options);

    // Solve and synchronize before reading back result states.
    cunls::CudaStream stream;
    const auto summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    // Download optimized poses and evaluate chain constraint satisfaction.
    std::vector<SE3Transform> optimized_poses(num_poses);
    poses_device.CopyToHost(optimized_poses.data(), num_poses);

    const float initial_error =
        examples::ChainConstraintError(initial_poses, deltas);
    const float final_error =
        examples::ChainConstraintError(optimized_poses, deltas);

    std::cout << "Pose Graph Optimization Example (Chain)\n";
    std::cout << "  Num poses:              " << num_poses << "\n";
    std::cout << "  Num constraints:        " << num_constraints << "\n";
    std::cout << "  Initial cost:           " << summary.initial_cost << "\n";
    std::cout << "  Final cost:             " << summary.final_cost << "\n";
    std::cout << "  Iterations:             " << summary.num_iterations << "\n";
    std::cout << "  Constraint MSE:         " << initial_error << " -> "
              << final_error << "\n";

    if (summary.final_cost > 1e-2f || final_error > initial_error * 0.05f) {
      std::cerr << "Optimization quality check failed.\n";
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
