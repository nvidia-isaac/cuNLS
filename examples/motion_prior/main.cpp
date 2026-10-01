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

// Constant-velocity motion prior: a chain of SE(3) poses T_i and body
// velocities v_i tied together by ConstantVelocityInformationSE3FactorBatch,
// the constant-velocity factor weighted by its closed-form process-noise
// information Q(dt)^-1. T_0 and v_0 are held fixed (gauge anchors), which
// makes the chain fully determined.

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/information/motion_prior_information.h"
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
    const size_t num_poses = 101;
    const size_t num_factors = num_poses - 1;
    const float dt = 0.1f;

    // 1. Synthetic data: an exact constant-velocity trajectory, then disturbed.
    const examples::ConstantVelocityScene scene =
        examples::MakeConstantVelocityScene(num_poses, dt);

    // 2. Upload the initial guess and the time steps to the GPU.
    dvector<SE3Transform> poses(scene.initial_poses);
    dvector<Vector<6>> velocities(scene.initial_velocities);
    dvector<float> dts(std::vector<float>(num_factors, dt));
    dvector<int> constant_ids(std::vector<int>{0});  // T_0 and v_0 are gauge anchors

    // 3. State batches: SE(3) poses and 6D body velocities.
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
    cunls::CudaStream stream;
    const size_t poses_capacity = num_poses;  // every slot solved: active = capacity
    const size_t const_capacity = 1;          // entries of constant_ids
    cunls::SE3StateBatch pose_states(cublas, reinterpret_cast<const float *>(poses.data()),
                                     poses_capacity, constant_ids.data(), const_capacity);
    cunls::VectorStateBatch<6> velocity_states(reinterpret_cast<const float *>(velocities.data()),
                                               poses_capacity, constant_ids.data(), const_capacity);
    const size_t num_const = 1;                           // active constant ids: T_0 / v_0
    pose_states.SetNumStateBlocks(num_poses, num_const);  // active counts
    velocity_states.SetNumStateBlocks(num_poses, num_const);

    // 4. The motion prior. Qc is the continuous-time process-noise PSD per
    //    tangent DOF (rotation x/y/z, translation x/y/z): smaller values trust
    //    constant velocity more. Factor i reads [T_i, T_{i+1}, v_i, v_{i+1}].
    //    Capacity (fixed, sizes the buffers) vs. active count (set per solve):
    //    see step 3.
    const std::vector<float> qc = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    dvector<float> qc_device(qc);
    const size_t factors_capacity = num_factors;  // dts holds this many; every slot solved
    cunls::ConstantVelocityInformationSE3FactorBatch motion_prior(
        cublas, stream.GetStream(), dts.data(), qc_device.data(), factors_capacity);
    motion_prior.SetNumFactors(num_factors);  // active count
    std::vector<float *> state_pointers;
    for (size_t i = 0; i < num_factors; ++i) {
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i));
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i + 1));
      state_pointers.push_back(velocity_states.StateBlockDevicePtr(i));
      state_pointers.push_back(velocity_states.StateBlockDevicePtr(i + 1));
    }

    // 5. The problem.
    cunls::Problem problem;
    problem.AddStateBatch(&pose_states);
    problem.AddStateBatch(&velocity_states);
    problem.AddFactorBatch(&motion_prior, state_pointers);
    if (!problem.CheckConsistency()) {
      std::cerr << "Problem consistency check failed\n";
      return 1;
    }

    // 6. Solve with Levenberg-Marquardt.
    cunls::LevenbergMarquardtMinimizerOptions options;
    options.base_options.max_num_iterations = 100;
    options.base_options.state_tolerance = 1e-8f;
    options.base_options.cost_tolerance = 1e-8f;
    options.initial_lambda = 1e-3f;
    cunls::LevenbergMarquardtMinimizer minimizer(options);
    const cunls::MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    // 7. Read back and compare with the ground truth.
    std::vector<SE3Transform> final_poses(num_poses);
    std::vector<Vector<6>> final_velocities(num_poses);
    poses.CopyToHost(final_poses.data(), num_poses);
    velocities.CopyToHost(final_velocities.data(), num_poses);
    const float pose_before = examples::ComputePoseMSE(scene.initial_poses, scene.gt_poses);
    const float pose_after = examples::ComputePoseMSE(final_poses, scene.gt_poses);
    const float vel_before =
        examples::ComputeVectorMSE(scene.initial_velocities, scene.gt_velocities);
    const float vel_after = examples::ComputeVectorMSE(final_velocities, scene.gt_velocities);

    examples::PrintTitle("Motion Prior Example (Constant-Velocity SE(3) Chain, Q(dt)^-1-weighted)");
    examples::PrintValue("Num poses", num_poses);
    examples::PrintValue("Num CV factors", num_factors);
    examples::PrintValue("dt", dt);
    examples::PrintValue("Qc (rot, trans)", examples::Str("[", qc[0], ", ", qc[3], "]"));
    examples::PrintSummary(summary);
    examples::PrintChange("Pose MSE", pose_before, pose_after);
    examples::PrintChange("Velocity MSE", vel_before, vel_after);
    return examples::QualityExitCode(summary.final_cost <= 1e-2f &&
                                     pose_after <= pose_before * 0.05f &&
                                     vel_after <= vel_before * 0.05f);
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
