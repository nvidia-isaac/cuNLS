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
 * @file se3_prior_factor_test.cpp
 * @brief Tests for SE3PriorFactorBatch with Levenberg-Marquardt convergence.
 *
 * Generates random SE(3) transforms as targets, perturbs them, and verifies that
 * LM optimization with the SE3 prior factor converges to the targets.
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
#include "cunls/factor/se3_prior_factor_batch.h"
#include "cunls/math/lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"

namespace cunls {

/**
 * @brief Test fixture for SE3PriorFactorBatch with LM optimization.
 *
 * Generates random SE(3) target transforms via the exponential map, then
 * creates perturbed initial transforms by applying a small random twist.
 * Verifies that LM optimization converges the perturbed transforms to targets.
 */
class SE3PriorCostTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> rotation_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> translation_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> perturbation_rot_dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> perturbation_trans_dist(-0.3f, 0.3f);

    // Generate random target twists and perturbation twists
    hvector<Vector<6>> target_twists(num_transforms_);
    hvector<Vector<6>> perturbation_twists(num_transforms_);

    for (size_t i = 0; i < num_transforms_; i++) {
      target_twists[i][0] = rotation_dist(rng);
      target_twists[i][1] = rotation_dist(rng);
      target_twists[i][2] = rotation_dist(rng);
      target_twists[i][3] = translation_dist(rng);
      target_twists[i][4] = translation_dist(rng);
      target_twists[i][5] = translation_dist(rng);

      perturbation_twists[i][0] = perturbation_rot_dist(rng);
      perturbation_twists[i][1] = perturbation_rot_dist(rng);
      perturbation_twists[i][2] = perturbation_rot_dist(rng);
      perturbation_twists[i][3] = perturbation_trans_dist(rng);
      perturbation_twists[i][4] = perturbation_trans_dist(rng);
      perturbation_twists[i][5] = perturbation_trans_dist(rng);
    }

    // Convert twists to SE(3) transforms on device
    CudaStream stream;
    constexpr size_t twist_stride = 6;
    constexpr size_t transform_pitch = 4;
    constexpr size_t transform_stride = 16;

    dvector<Vector<6>> target_twists_device(target_twists);
    dvector<Vector<6>> perturbation_twists_device(perturbation_twists);

    target_transforms_device_ = DeviceVector<SE3Transform>(num_transforms_);
    initial_transforms_device_ = DeviceVector<SE3Transform>(num_transforms_);

    // Compute T_target = Exp(target_twist)
    ComputeExpSE3(stream.GetStream(),
                  reinterpret_cast<const float*>(target_twists_device.data()),
                  twist_stride, transform_pitch, transform_stride,
                  num_transforms_,
                  reinterpret_cast<float*>(target_transforms_device_.data()));

    // Compute T_perturbation = Exp(perturbation_twist)
    dvector<SE3Transform> perturbation_transforms(num_transforms_);
    ComputeExpSE3(
        stream.GetStream(),
        reinterpret_cast<const float*>(perturbation_twists_device.data()),
        twist_stride, transform_pitch, transform_stride, num_transforms_,
        reinterpret_cast<float*>(perturbation_transforms.data()));

    // Compute T_initial = T_target * T_perturbation using cuBLAS
    cuBLASHandle cublas_handle;
    auto handle = cublas_handle.GetHandle(stream.GetStream());
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    constexpr int mat_size = 4;
    constexpr int stride = 16;

    THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
        reinterpret_cast<const float*>(perturbation_transforms.data()),
        mat_size, stride,
        reinterpret_cast<const float*>(target_transforms_device_.data()),
        mat_size, stride, &beta,
        reinterpret_cast<float*>(initial_transforms_device_.data()), mat_size,
        stride, num_transforms_));

    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  const size_t num_transforms_ = 10000;
  const uint32_t fixed_seed_ = 42;
  DeviceVector<SE3Transform> target_transforms_device_;
  DeviceVector<SE3Transform> initial_transforms_device_;
  cuBLASHandle cublas_handle_;

  profiler::Domain profiler_domain_{"SE3PriorCostTest"};
};

/**
 * @brief Tests that LM optimization converges for SE(3) prior cost.
 *
 * Each SE(3) transform is initialized with a perturbation and optimized to
 * match a target transform using SE3PriorFactorBatch.
 * Minimizes Log(T_target^{-1} * T_current) over the SE(3) manifold.
 */
TEST_F(SE3PriorCostTest, LMConvergence) {
  auto test_range = profiler_domain_.CreateDomainRange("LMConvergence");

  // Create SE(3) state batch from perturbed initial transforms
  const float* params_ptr =
      reinterpret_cast<const float*>(initial_transforms_device_.data());
  SE3StateBatch state_batch(cublas_handle_, params_ptr, num_transforms_);

  // Create SE(3) prior factor batch with target transforms
  SE3PriorFactorBatch factor_batch(cublas_handle_,
                                          target_transforms_device_.data(),
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
  EXPECT_LT(summary.final_cost, 1e-2f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  // Copy optimized and target transforms to host for verification
  hvector<SE3Transform> optimized(num_transforms_);
  hvector<SE3Transform> targets(num_transforms_);
  const float* opt_ptr = state_batch.StateBlockDevicePtr(0);
  THROW_ON_CUDA_ERROR(cudaMemcpy(optimized.data(), opt_ptr,
                                  num_transforms_ * sizeof(SE3Transform),
                                  cudaMemcpyDeviceToHost));
  target_transforms_device_.CopyToHost(targets.data(), num_transforms_);

  constexpr float tolerance = 1e-2f;
  for (size_t i = 0; i < num_transforms_; i++) {
    for (size_t j = 0; j < 16; j++) {
      ASSERT_NEAR(optimized[i][j], targets[i][j], tolerance)
          << "Mismatch at transform " << i << ", element " << j;
    }
  }
}

}  // namespace cunls
