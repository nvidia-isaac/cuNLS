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
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/information/motion_prior_information.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "utils/se3_utils.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;

namespace {

// Applies the SE(3) left Jacobian J_l(twist) to a vector on the host, using
// the GPU math library. J_l transports a body-frame velocity from one pose
// to the next under the constant-velocity model, i.e. v_{i+1} = J_l(step_twist) * v_i.
Vector<6> ApplyLeftJacobianSE3(const Vector<6> &twist, const Vector<6> &v) {
  dvector<Vector<6>> twist_dev({twist});
  dvector<cunls::Matrix<6>> jl_dev(1);
  cunls::CudaStream stream;
  cunls::ComputeJacobianLeftSE3(stream.GetStream(),
                                reinterpret_cast<const float *>(twist_dev.data()), 6, 6, 36, 1,
                                reinterpret_cast<float *>(jl_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  cunls::Matrix<6> jl;
  jl_dev.CopyToHost(&jl, 1);

  Vector<6> out{};
  for (int i = 0; i < 6; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < 6; ++j) acc += jl[i * 6 + j] * v[j];
    out[i] = acc;
  }
  return out;
}

}  // namespace

int main() {
  try {
    // A chain of poses under a constant body-velocity motion prior:
    //   (T_0, v_0) -> (T_1, v_1) -> ... -> (T_{N-1}, v_{N-1})
    // Both pose T_0 and velocity v_0 are held fixed as gauge anchors: this
    // matches the number of free unknowns ((2N-1)*6 minus the anchor's 12)
    // to the number of constraints ((N-1)*12) from the between-consecutive
    // ConstantVelocitySE3FactorBatch factors, so the chain is fully (not
    // just relatively) determined.
    const size_t num_poses = 101;
    const size_t num_factors = num_poses - 1;
    const float dt = 0.1f;

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> rot(-0.15f, 0.15f);
    std::uniform_real_distribution<float> trans(-0.5f, 0.5f);

    // Ground-truth trajectory: pick an anchor pose and velocity, then
    // integrate forward exactly along the constant-velocity model, so every
    // consecutive pair satisfies the factor's residual with zero error.
    std::vector<SE3Transform> anchor_pose;
    examples::GenerateRandomSE3(1, rng, anchor_pose);
    Vector<6> anchor_vel{rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), trans(rng)};

    std::vector<SE3Transform> gt_poses(num_poses);
    std::vector<Vector<6>> gt_vels(num_poses);
    gt_poses[0] = anchor_pose[0];
    gt_vels[0] = anchor_vel;
    for (size_t i = 0; i < num_factors; ++i) {
      Vector<6> step_twist;
      for (int d = 0; d < 6; ++d) step_twist[d] = dt * gt_vels[i][d];

      std::vector<Vector<6>> step_twist_vec = {step_twist};
      std::vector<SE3Transform> step_pose;
      examples::TwistsToSE3(step_twist_vec, step_pose);

      gt_poses[i + 1] = examples::ComposeSE3(gt_poses[i], step_pose[0]);
      // Constant body velocity: v_{i+1} transported back to frame i via
      // J_l^{-1} equals v_i, i.e. v_{i+1} = J_l(step_twist) * v_i.
      gt_vels[i + 1] = ApplyLeftJacobianSE3(step_twist, gt_vels[i]);
    }

    // Disturb every pose/velocity except the anchor to give the optimizer
    // something to do.
    std::vector<SE3Transform> pose_disturbance;
    examples::GenerateRandomSE3(num_factors, rng, pose_disturbance, 0.05f, 0.2f);
    std::uniform_real_distribution<float> vel_noise(-0.1f, 0.1f);

    std::vector<SE3Transform> initial_poses(num_poses);
    std::vector<Vector<6>> initial_vels(num_poses);
    initial_poses[0] = gt_poses[0];
    initial_vels[0] = gt_vels[0];
    for (size_t i = 0; i < num_factors; ++i) {
      initial_poses[i + 1] = examples::ComposeSE3(pose_disturbance[i], gt_poses[i + 1]);
      for (int d = 0; d < 6; ++d) initial_vels[i + 1][d] = gt_vels[i + 1][d] + vel_noise(rng);
    }

    // Copy host data to GPU.
    dvector<SE3Transform> poses_device(initial_poses);
    dvector<Vector<6>> vels_device(initial_vels);
    std::vector<float> dt_values(num_factors, dt);
    dvector<float> dt_device(dt_values);

    // Anchor T_0 and v_0 as constant (gauge fix).
    std::vector<int> const_ids = {0};
    dvector<int> const_ids_device(const_ids);

    cunls::cuBLASHandle cublas_handle;
    cunls::CudaStream stream;
    cunls::SE3StateBatch pose_states(cublas_handle,
                                     reinterpret_cast<const float *>(poses_device.data()),
                                     num_poses, const_ids_device.data(), 1);
    cunls::VectorStateBatch<6> vel_states(reinterpret_cast<const float *>(vels_device.data()),
                                          num_poses, const_ids_device.data(), 1);

    // Continuous-time process-noise PSD Qc (one entry per SE(3) tangent DOF:
    // rotation x/y/z, then translation x/y/z). This encodes how much the
    // constant-velocity assumption is trusted to drift per unit time; smaller
    // values mean a tighter prior (more confidence in constant velocity),
    // larger values mean a looser one. Fusing it via
    // ConstantVelocityInformationSE3FactorBatch turns the plain (unweighted)
    // ConstantVelocitySE3FactorBatch residual/Jacobian into the paper's
    // properly-scaled Q(dt)^-1-weighted one, with no extra work at the call
    // site beyond providing Qc.
    const std::vector<float> qc_diag = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    dvector<float> qc_device(qc_diag);

    cunls::ConstantVelocityInformationSE3FactorBatch motion_prior(
        cublas_handle, stream.GetStream(), dt_device.data(), qc_device.data(), num_factors);

    // Flatten factor-to-state connectivity: [T_i, T_i+1, v_i, v_i+1] per
    // factor.
    std::vector<float *> state_pointers;
    state_pointers.reserve(4 * num_factors);
    for (size_t i = 0; i < num_factors; ++i) {
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i));
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i + 1));
      state_pointers.push_back(vel_states.StateBlockDevicePtr(i));
      state_pointers.push_back(vel_states.StateBlockDevicePtr(i + 1));
    }

    cunls::Problem problem;
    problem.AddStateBatch(&pose_states);
    problem.AddStateBatch(&vel_states);
    problem.AddFactorBatch(&motion_prior, state_pointers);
    if (!problem.CheckConsistency()) {
      std::cerr << "Problem consistency check failed\n";
      return 1;
    }

    cunls::MinimizerOptions options;
    options.max_num_iterations = 100;
    options.state_tolerance = 1e-8f;
    options.cost_tolerance = 1e-8f;

    cunls::LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    lm_options.initial_lambda = 1e-3f;
    cunls::LevenbergMarquardtMinimizer minimizer(lm_options);

    const auto summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    std::vector<SE3Transform> optimized_poses(num_poses);
    std::vector<Vector<6>> optimized_vels(num_poses);
    poses_device.CopyToHost(optimized_poses.data(), num_poses);
    vels_device.CopyToHost(optimized_vels.data(), num_poses);

    const float initial_pose_mse = examples::ComputePoseMSE(initial_poses, gt_poses);
    const float final_pose_mse = examples::ComputePoseMSE(optimized_poses, gt_poses);
    const float initial_vel_mse = examples::ComputeVectorMSE<6>(initial_vels, gt_vels);
    const float final_vel_mse = examples::ComputeVectorMSE<6>(optimized_vels, gt_vels);

    std::cout << "Motion Prior Example (Constant-Velocity SE(3) Chain, Q(dt)^-1-weighted)\n";
    std::cout << "  Num poses:              " << num_poses << "\n";
    std::cout << "  Num CV factors:         " << num_factors << "\n";
    std::cout << "  dt:                     " << dt << "\n";
    std::cout << "  Qc (rot, trans):        [" << qc_diag[0] << ", " << qc_diag[3] << "]\n";
    std::cout << "  Initial cost:           " << summary.initial_cost << "\n";
    std::cout << "  Final cost:             " << summary.final_cost << "\n";
    std::cout << "  Iterations:             " << summary.num_iterations << "\n";
    std::cout << "  Pose MSE:               " << initial_pose_mse << " -> " << final_pose_mse
              << "\n";
    std::cout << "  Velocity MSE:           " << initial_vel_mse << " -> " << final_vel_mse << "\n";

    if (summary.final_cost > 1e-2f || final_pose_mse > initial_pose_mse * 0.05f ||
        final_vel_mse > initial_vel_mse * 0.05f) {
      std::cerr << "Optimization quality check failed.\n";
      return 2;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
