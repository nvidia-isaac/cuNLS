/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * @file so2_prior_factor_test.cpp
 * @brief Tests for SO2PriorFactorBatch with Levenberg-Marquardt convergence.
 *
 * Validates that SO(2) state blocks can be optimized to match target rotations
 * using the SO2 prior factor and LM minimizer.
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
#include "cunls/factor/so2_prior_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/so2_state_batch.h"

namespace cunls {

/**
 * @brief Test fixture for SO2PriorFactorBatch with LM optimization.
 *
 * Creates target SO(2) rotation matrices from random angles and perturbed
 * initial rotations, then verifies that LM optimization converges.
 */
class SO2PriorCostTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> angle_dist(
        -static_cast<float>(M_PI), static_cast<float>(M_PI));
    std::uniform_real_distribution<float> perturbation_dist(-0.3f, 0.3f);

    target_rotations_.resize(num_rotations_);
    initial_rotations_.resize(num_rotations_);

    for (size_t i = 0; i < num_rotations_; i++) {
      float target_angle = angle_dist(rng);
      float perturbed_angle = target_angle + perturbation_dist(rng);

      // Target rotation: [[cos, -sin], [sin, cos]] in row-major order
      float ct = cosf(target_angle);
      float st = sinf(target_angle);
      target_rotations_[i] = {ct, -st, st, ct};

      // Perturbed initial rotation
      float cp = cosf(perturbed_angle);
      float sp = sinf(perturbed_angle);
      initial_rotations_[i] = {cp, -sp, sp, cp};
    }
  }

  const size_t num_rotations_ = 10000;
  const uint32_t fixed_seed_ = 42;
  std::vector<Matrix<2>> target_rotations_;
  std::vector<Matrix<2>> initial_rotations_;
  cuBLASHandle cublas_handle_;

  profiler::Domain profiler_domain_{"SO2PriorCostTest"};
};

/**
 * @brief Tests that TangentSize returns the correct dimension (1).
 */
TEST_F(SO2PriorCostTest, TangentSize) {
  dvector<Matrix<2>> rotations_device(initial_rotations_);
  const float* data_ptr =
      reinterpret_cast<const float*>(rotations_device.data());
  SO2StateBatch so2_states(cublas_handle_, data_ptr, num_rotations_);
  ASSERT_EQ(so2_states.TangentSize(), 1);
}

/**
 * @brief Tests that AmbientSize returns the correct dimension (4).
 */
TEST_F(SO2PriorCostTest, AmbientSize) {
  dvector<Matrix<2>> rotations_device(initial_rotations_);
  const float* data_ptr =
      reinterpret_cast<const float*>(rotations_device.data());
  SO2StateBatch so2_states(cublas_handle_, data_ptr, num_rotations_);
  ASSERT_EQ(so2_states.AmbientSize(), 4);
}

/**
 * @brief Tests that NumStateBlocks returns the correct count.
 */
TEST_F(SO2PriorCostTest, NumStateBlocks) {
  dvector<Matrix<2>> rotations_device(initial_rotations_);
  const float* data_ptr =
      reinterpret_cast<const float*>(rotations_device.data());
  SO2StateBatch so2_states(cublas_handle_, data_ptr, num_rotations_);
  ASSERT_EQ(so2_states.NumStateBlocks(), num_rotations_);
}

/**
 * @brief Tests that LM optimization converges for SO(2) prior cost.
 *
 * Each SO(2) rotation is initialized with a perturbed angle and optimized
 * to match a target rotation using SO2PriorFactorBatch.
 * Minimizes the angular error Log(R_target^T * R_current) over the SO(2) manifold.
 */
TEST_F(SO2PriorCostTest, LMConvergence) {
  auto test_range = profiler_domain_.CreateDomainRange("LMConvergence");

  // Copy data to device
  dvector<Matrix<2>> rotations_device(initial_rotations_);
  dvector<Matrix<2>> targets_device(target_rotations_);

  // Create SO(2) state batch
  const float* state_data_ptr =
      reinterpret_cast<const float*>(rotations_device.data());
  SO2StateBatch state_batch(cublas_handle_, state_data_ptr, num_rotations_);

  // Create SO(2) prior factor batch
  SO2PriorFactorBatch factor_batch(targets_device.data(),
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
  EXPECT_GT(summary.num_iterations, 0u);

  // Verify optimized rotations match targets
  std::vector<Matrix<2>> optimized(num_rotations_);
  const float* opt_ptr = state_batch.StateBlockDevicePtr(0);
  THROW_ON_CUDA_ERROR(cudaMemcpy(optimized.data(), opt_ptr,
                                  num_rotations_ * sizeof(Matrix<2>),
                                  cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < num_rotations_; i++) {
    for (size_t j = 0; j < 4; j++) {
      ASSERT_NEAR(optimized[i][j], target_rotations_[i][j], 1e-3f)
          << "Mismatch at rotation " << i << ", element " << j;
    }
  }
}

}  // namespace cunls
