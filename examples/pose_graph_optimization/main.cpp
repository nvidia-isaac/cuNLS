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

// Pose graph optimization: recover a chain of SE(3) poses
// T_0 -> T_1 -> ... -> T_{N-1} from relative-transform measurements between
// consecutive poses. T_0 is held fixed (gauge anchor).

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/between_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "utils/datasets.h"
#include "utils/report.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;

int main() {
  try {
    const size_t num_poses = 201;
    const size_t num_constraints = num_poses - 1;

    // 1. Synthetic data: ground-truth chain, disturbed initial guess, and the
    //    relative transforms (deltas) between consecutive poses.
    const examples::PoseChainScene scene = examples::MakePoseChainScene(num_poses);

    // 2. Upload the initial guess and the measurements to the GPU.
    dvector<SE3Transform> poses(scene.initial_poses);
    dvector<SE3Transform> deltas(scene.deltas);
    dvector<int> constant_pose_ids(std::vector<int>{0});  // T_0 is the gauge anchor

    // 3. One state batch for the whole chain.
    //    Capacity vs. active count. A batch is constructed with its capacity: how many states (or
    //    factors) its bound device buffers hold. The capacity is fixed for the batch's lifetime;
    //    size it once for the largest problem you expect. Right after construction nothing is
    //    active: SetNumActiveStates / SetNumActiveFactors set the active count, how many of the
    //    first slots the next solve uses (a solve without it throws). The setter is host-only (no
    //    allocation, no device work) and may change the count between solves up to the capacity,
    //    which is what lets a real-time application allocate once and reuse the same buffers every
    //    frame while the problem size changes. This example solves every slot once, so each active
    //    count equals its capacity.
    const size_t poses_capacity = num_poses;  // every slot solved: active = capacity
    const size_t const_poses_capacity = 1;    // entries of constant_pose_ids
    cunls::SE3StateBatch pose_states(reinterpret_cast<const float *>(poses.data()), poses_capacity,
                                     constant_pose_ids.data(), const_poses_capacity);
    const size_t num_const_poses = 1;  // active constant ids: the gauge anchor
    pose_states.SetNumActiveStates(num_poses, num_const_poses);  // active counts

    // 4. Between factors; the manifold (SE(3)) is deduced from the deltas' type.
    //    Factor i reads [T_i, T_{i+1}]. Capacity (fixed, sizes the buffers) vs.
    //    active count (set per solve): see step 3.
    const size_t constraints_capacity = num_constraints;  // every slot solved
    cunls::BetweenFactorBatch between(deltas.data(), constraints_capacity);
    between.SetNumActiveFactors(num_constraints);  // active count
    std::vector<float *> state_pointers;
    for (size_t i = 0; i < num_constraints; ++i) {
      state_pointers.push_back(pose_states.StateDevicePtr(i));
      state_pointers.push_back(pose_states.StateDevicePtr(i + 1));
    }

    // 5. The problem.
    cunls::Problem problem;
    problem.AddStateBatch(&pose_states);
    problem.AddFactorBatch(&between, state_pointers);
    if (!problem.CheckConsistency()) {
      std::cerr << "Problem consistency check failed\n";
      return 1;
    }

    // 6. Solve with Levenberg-Marquardt; the result is written into poses.
    cunls::LevenbergMarquardtMinimizerOptions options;
    options.base_options.max_num_iterations = 60;
    options.base_options.state_tolerance = 1e-8f;
    options.base_options.cost_tolerance = 1e-8f;
    options.initial_lambda = 1e-3f;
    cunls::LevenbergMarquardtMinimizer minimizer(options);
    cunls::CudaStream stream;
    const cunls::MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    // 7. Read back and measure how well the chain satisfies its constraints.
    std::vector<SE3Transform> final_poses(num_poses);
    poses.CopyToHost(final_poses.data(), num_poses);
    const float error_before = examples::ChainConstraintError(scene.initial_poses, scene.deltas);
    const float error_after = examples::ChainConstraintError(final_poses, scene.deltas);

    examples::PrintTitle("Pose Graph Optimization Example (Chain)");
    examples::PrintValue("Num poses", num_poses);
    examples::PrintValue("Num constraints", num_constraints);
    examples::PrintSummary(summary);
    examples::PrintChange("Constraint MSE", error_before, error_after);
    return examples::QualityExitCode(summary.final_cost <= 1e-2f &&
                                     error_after <= error_before * 0.05f);
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
