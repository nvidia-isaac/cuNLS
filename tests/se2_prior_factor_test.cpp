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
 * @file se2_prior_factor_test.cpp
 * @brief Tests for SE2StateBatch and SE2PriorFactorBatch with
 * Levenberg-Marquardt convergence.
 *
 * Validates that SE(2) state blocks have correct dimensions and that
 * perturbed SE(2) transforms can be optimized to match targets using the
 * SE2 prior factor and LM minimizer.
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
#include "cunls/factor/se2_prior_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"

namespace cunls {

/**
 * @brief Test fixture for SE2StateBatch and SE2PriorFactorBatch.
 *
 * Creates target SE(2) transformation matrices from random (x, y, theta) poses
 * and perturbed initial transforms, then verifies dimension properties and
 * that LM optimization converges.
 */
class SE2PriorCostTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> angle_dist(
        -static_cast<float>(M_PI), static_cast<float>(M_PI));
    std::uniform_real_distribution<float> translation_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> perturbation_angle_dist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> perturbation_trans_dist(-0.3f, 0.3f);

    target_transforms_.resize(num_transforms_);
    initial_transforms_.resize(num_transforms_);

    for (size_t i = 0; i < num_transforms_; i++) {
      float target_theta = angle_dist(rng);
      float target_x = translation_dist(rng);
      float target_y = translation_dist(rng);

      float perturbed_theta = target_theta + perturbation_angle_dist(rng);
      float perturbed_x = target_x + perturbation_trans_dist(rng);
      float perturbed_y = target_y + perturbation_trans_dist(rng);

      // Target transform: 3x3 row-major
      // [cos(theta), -sin(theta), tx]
      // [sin(theta),  cos(theta), ty]
      // [    0,           0,       1]
      float ct = cosf(target_theta);
      float st = sinf(target_theta);
      target_transforms_[i] = {ct, -st, target_x,
                                st,  ct, target_y,
                                0.0f, 0.0f, 1.0f};

      // Perturbed initial transform
      float cp = cosf(perturbed_theta);
      float sp = sinf(perturbed_theta);
      initial_transforms_[i] = {cp, -sp, perturbed_x,
                                 sp,  cp, perturbed_y,
                                 0.0f, 0.0f, 1.0f};
    }
  }

  const size_t num_transforms_ = 10000;
  const uint32_t fixed_seed_ = 42;
  std::vector<Matrix<3>> target_transforms_;
  std::vector<Matrix<3>> initial_transforms_;

  cuBLASHandle cublas_handle_;

  profiler::Domain profiler_domain_{"SE2PriorCostTest"};
};

/**
 * @brief Tests that TangentSize returns the correct dimension (3).
 *
 * The tangent space of SE(2) has dimension 3: 2 for translation and 1 for
 * rotation.
 */
TEST_F(SE2PriorCostTest, TangentSize) {
  dvector<Matrix<3>> transforms_device(initial_transforms_);
  const float* data_ptr =
      reinterpret_cast<const float*>(transforms_device.data());
  SE2StateBatch se2_states(cublas_handle_, data_ptr, num_transforms_);
  ASSERT_EQ(se2_states.TangentSize(), 3);
}

/**
 * @brief Tests that AmbientSize returns the correct dimension (9).
 *
 * The ambient space is a 3x3 homogeneous transformation matrix,
 * stored as 9 floats in row-major order.
 */
TEST_F(SE2PriorCostTest, AmbientSize) {
  dvector<Matrix<3>> transforms_device(initial_transforms_);
  const float* data_ptr =
      reinterpret_cast<const float*>(transforms_device.data());
  SE2StateBatch se2_states(cublas_handle_, data_ptr, num_transforms_);
  ASSERT_EQ(se2_states.AmbientSize(), 9);
}

/**
 * @brief Tests that NumStateBlocks returns the correct count.
 */
TEST_F(SE2PriorCostTest, NumStateBlocks) {
  dvector<Matrix<3>> transforms_device(initial_transforms_);
  const float* data_ptr =
      reinterpret_cast<const float*>(transforms_device.data());
  SE2StateBatch se2_states(cublas_handle_, data_ptr, num_transforms_);
  ASSERT_EQ(se2_states.NumStateBlocks(), num_transforms_);
}

/**
 * @brief Tests that LM optimization converges for SE(2) prior cost.
 *
 * Each SE(2) transform is initialized with a perturbed pose and optimized
 * to match a target transform using SE2PriorFactorBatch.
 * Minimizes Log(T_target^{-1} * T_current) over the SE(2) manifold.
 */
TEST_F(SE2PriorCostTest, LMConvergence) {
  auto test_range = profiler_domain_.CreateDomainRange("LMConvergence");

  // Copy data to device
  dvector<Matrix<3>> transforms_device(initial_transforms_);
  dvector<Matrix<3>> targets_device(target_transforms_);

  // Create SE(2) state batch
  const float* state_data_ptr =
      reinterpret_cast<const float*>(transforms_device.data());
  SE2StateBatch state_batch(cublas_handle_, state_data_ptr, num_transforms_);

  // Create SE(2) prior factor batch
  SE2PriorFactorBatch factor_batch(cublas_handle_, targets_device.data(),
                                          num_transforms_);

  // Collect state pointers
  std::vector<float*> state_pointers;
  state_pointers.reserve(num_transforms_);
  for (size_t i = 0; i < num_transforms_; i++) {
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
  EXPECT_GT(summary.num_iterations, 0u);

  // Verify optimized transforms match targets
  std::vector<Matrix<3>> optimized(num_transforms_);
  const float* opt_ptr = state_batch.StateBlockDevicePtr(0);
  THROW_ON_CUDA_ERROR(cudaMemcpy(optimized.data(), opt_ptr,
                                  num_transforms_ * sizeof(Matrix<3>),
                                  cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < num_transforms_; i++) {
    for (size_t j = 0; j < 9; j++) {
      ASSERT_NEAR(optimized[i][j], target_transforms_[i][j], 1e-2f)
          << "Mismatch at transform " << i << ", element " << j;
    }
  }
}

}  // namespace cunls
