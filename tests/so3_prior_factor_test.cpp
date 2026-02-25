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

/**
 * @file so3_prior_factor_test.cpp
 * @brief Tests for SO3PriorFactorBatch with Levenberg-Marquardt
 * convergence.
 *
 * Generates random SO(3) rotations as targets, perturbs them, and verifies that
 * LM optimization with the SO3 prior factor converges to the targets.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/so3_prior_factor_batch.h"
#include "cunls/math/lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/so3_state_batch.h"

namespace cunls {

/**
 * @brief Test fixture for SO3PriorFactorBatch with LM optimization.
 *
 * Generates random SO(3) target rotations via the exponential map, then creates
 * perturbed initial rotations by applying a small random twist. Verifies that
 * LM optimization converges the perturbed rotations to the targets.
 */
class SO3PriorCostTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> rotation_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> perturbation_dist(-0.3f, 0.3f);

    // Generate random target twists and perturbation twists
    hvector<Vector<3>> target_twists(num_rotations_);
    hvector<Vector<3>> perturbation_twists(num_rotations_);

    for (size_t i = 0; i < num_rotations_; i++) {
      target_twists[i][0] = rotation_dist(rng);
      target_twists[i][1] = rotation_dist(rng);
      target_twists[i][2] = rotation_dist(rng);

      perturbation_twists[i][0] = perturbation_dist(rng);
      perturbation_twists[i][1] = perturbation_dist(rng);
      perturbation_twists[i][2] = perturbation_dist(rng);
    }

    // Convert twists to rotation matrices on device
    CudaStream stream;
    constexpr size_t twist_stride = 3;
    constexpr size_t rotation_pitch = 3;
    constexpr size_t rotation_stride = 9;

    dvector<Vector<3>> target_twists_device(target_twists);
    dvector<Vector<3>> perturbation_twists_device(perturbation_twists);

    target_rotations_device_ = DeviceVector<Matrix<3>>(num_rotations_);
    initial_rotations_device_ = DeviceVector<Matrix<3>>(num_rotations_);

    // Compute R_target = Exp(target_twist)
    ComputeExpSO3(stream.GetStream(),
                  reinterpret_cast<const float*>(target_twists_device.data()),
                  twist_stride, rotation_pitch, rotation_stride, num_rotations_,
                  reinterpret_cast<float*>(target_rotations_device_.data()));

    // Compute R_perturbation = Exp(perturbation_twist)
    dvector<Matrix<3>> perturbation_rotations(num_rotations_);
    ComputeExpSO3(
        stream.GetStream(),
        reinterpret_cast<const float*>(perturbation_twists_device.data()),
        twist_stride, rotation_pitch, rotation_stride, num_rotations_,
        reinterpret_cast<float*>(perturbation_rotations.data()));

    // Compute R_initial = R_target * R_perturbation using cuBLAS
    auto handle = cublas_handle_.GetHandle(stream.GetStream());
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    constexpr int mat_size = 3;
    constexpr int stride = 9;

    THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
        reinterpret_cast<const float*>(perturbation_rotations.data()), mat_size,
        stride, reinterpret_cast<const float*>(target_rotations_device_.data()),
        mat_size, stride, &beta,
        reinterpret_cast<float*>(initial_rotations_device_.data()), mat_size,
        stride, num_rotations_));

    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  const size_t num_rotations_ = 10000;
  const uint32_t fixed_seed_ = 42;
  DeviceVector<Matrix<3>> target_rotations_device_;
  DeviceVector<Matrix<3>> initial_rotations_device_;
  cuBLASHandle cublas_handle_;

  profiler::Domain profiler_domain_{"SO3PriorCostTest"};
};

/**
 * @brief Tests that LM optimization converges for SO(3) prior cost.
 *
 * Each SO(3) rotation is initialized with a perturbation and optimized to
 * match a target rotation using SO3PriorFactorBatch.
 * Minimizes Log(R_target^T * R_current) over the SO(3) manifold.
 */
TEST_F(SO3PriorCostTest, LMConvergence) {
  auto test_range = profiler_domain_.CreateDomainRange("LMConvergence");

  // Create SO(3) state batch from perturbed initial rotations
  const float* state_data_ptr =
      reinterpret_cast<const float*>(initial_rotations_device_.data());
  SO3StateBatch state_batch(cublas_handle_, state_data_ptr, num_rotations_);

  // Create SO(3) prior factor batch with target rotations
  SO3PriorFactorBatch factor_batch(cublas_handle_,
                                          target_rotations_device_.data(),
                                          num_rotations_);

  // Collect state pointers
  std::vector<float*> state_pointers;
  state_pointers.reserve(num_rotations_);
  for (size_t i = 0; i < num_rotations_; i++) {
    state_pointers.push_back(state_batch.StateBlockDevicePtr(i));
  }

  // Build problem
  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  // Configure LM minimizer
  MinimizerOptions options;
  options.max_num_iterations = 50;
  options.state_tolerance = 1e-8f;
  options.cost_tolerance = 1e-8f;

  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-3f;

  // Run optimization
  CudaStream stream;
  LevenbergMarquardtMinimizer minimizer(lm_options);
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Verify cost decreased significantly
  EXPECT_LT(summary.final_cost, 1e-4f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GE(summary.num_iterations, 0u);

  // Copy optimized and target rotations to host for verification
  hvector<Matrix<3>> optimized(num_rotations_);
  hvector<Matrix<3>> targets(num_rotations_);
  const float* opt_ptr = state_batch.StateBlockDevicePtr(0);
  THROW_ON_CUDA_ERROR(cudaMemcpy(optimized.data(), opt_ptr,
                                 num_rotations_ * sizeof(Matrix<3>),
                                 cudaMemcpyDeviceToHost));
  target_rotations_device_.CopyToHost(targets.data(), num_rotations_);

  constexpr float tolerance = 1e-2f;
  for (size_t i = 0; i < num_rotations_; i++) {
    for (size_t j = 0; j < 9; j++) {
      ASSERT_NEAR(optimized[i][j], targets[i][j], tolerance)
          << "Mismatch at rotation " << i << ", element " << j;
    }
  }
}

}  // namespace cunls
