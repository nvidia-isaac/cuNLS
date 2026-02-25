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
 * @file se3_between_factor_batch_test.cpp
 * @brief Unit tests for SE3BetweenFactorBatch and its InformationFactorBatch wrapper.
 *
 * Verifies that Evaluate runs without errors for SE3 between-pose factors
 * with and without Jacobians, and with information matrix weighting.
 */

#include <gtest/gtest.h>

#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/information_factor_batch.h"
#include "cunls/math/lie_math.h"
#include "cunls/common/profiler.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Test fixture for SE3BetweenFactorBatch unit tests.
 */
class SE3BetweenFactorBatchTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> rotation_dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> translation_dist(-1.0f, 1.0f);

    size_t num_poses = num_factors_ + 1;

    // Generate random twists for pose_left and pose_right
    std::vector<Vector<6>> twists(num_poses);

    for (size_t i = 0; i < num_poses; i++) {
      Vector<6>& twist = twists[i];

      // Initialize random SE3 twists (rotation + translation)
      twist[0] = rotation_dist(rng);     // rotation x
      twist[1] = rotation_dist(rng);     // rotation y
      twist[2] = rotation_dist(rng);     // rotation z
      twist[3] = translation_dist(rng);  // translation x
      twist[4] = translation_dist(rng);  // translation y
      twist[5] = translation_dist(rng);  // translation z
    }

    // Copy twists to device
    DeviceVector<Vector<6>> twists_device(twists);

    // Allocate device memory for transforms
    poses_device_.resize(twists.size());

    // Use ComputeExpSE3 to convert twists to SE3 transforms
    CudaStream stream;
    constexpr size_t twist_stride = 6;
    constexpr size_t transform_pitch = 4;    // row-major 4x4 matrix
    constexpr size_t transform_stride = 16;  // 16 floats per transform

    const float* twists_ptr = reinterpret_cast<const float*>(
        twists_device.data());
    float* poses_ptr = reinterpret_cast<float*>(
        poses_device_.data());

    ComputeExpSE3(stream.GetStream(), twists_ptr, twist_stride, transform_pitch,
           transform_stride, num_poses, poses_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    // Allocate device memory for state pointers
    // We need an array of const float* pointers on the device
    std::vector<const float*> state_ptrs_host(2 * num_factors_);

    for (size_t i = 0; i < num_factors_; i++) {
      state_ptrs_host[2 * i] = reinterpret_cast<const float*>(
          poses_device_.data() + (i + 1));
      state_ptrs_host[2 * i + 1] = reinterpret_cast<const float*>(
          poses_device_.data() + i);
    }

    // Copy state pointers to device
    state_ptrs_device_ = DeviceVector<const float*>(state_ptrs_host);

    // Allocate device memory for residuals (6 floats per factor)
    residuals_device_.resize(6 * num_factors_);

    // Allocate device memory for jacobians
    // Each factor has 6 residuals and 2 state blocks of dimension 6 each
    // Jacobian layout: [J_left (6x6), J_right (6x6)] per factor
    jacobians_device_.resize(12 * 6 * num_factors_);

    // Initialize jacobians to zero
    THROW_ON_CUDA_ERROR(cudaMemset(jacobians_device_.data(), 0,
                                   jacobians_device_.size() * sizeof(float)));

    SE3Transform identity;
    identity.fill(0.0f);
    identity[0] = 1.0f;
    identity[5] = 1.0f;
    identity[10] = 1.0f;
    identity[15] = 1.0f;

    // Create pose deltas (identity matrices) in device memory
    std::vector<SE3Transform> pose_deltas_host(num_factors_, identity);
    pose_deltas_device_ = DeviceVector<SE3Transform>(pose_deltas_host);
  }

  const size_t num_factors_ =
      10000;                        ///< Number of factors in the batch
  const uint32_t fixed_seed_ = 42;  ///< Fixed seed for random number generation
  DeviceVector<SE3Transform> poses_device_;  ///< Device storage for SE(3) poses
  DeviceVector<const float*>
      state_ptrs_device_;  ///< Device storage for state block pointers
  DeviceVector<float> residuals_device_;  ///< Device storage for residuals (6 floats
                                     ///< per factor)
  DeviceVector<float> jacobians_device_;  ///< Device storage for Jacobians (12x6
                                     ///< floats per factor)

  DeviceVector<SE3Transform>
      pose_deltas_device_;  ///< Device storage for pose deltas (identity matrices)
  cuBLASHandle cublas_handle_;

  profiler::Domain profiler_domain_{"SE3BetweenFactorBatchTest"};
};

/**
 * @brief Tests that Evaluate executes without errors.
 *
 * This test verifies that the Evaluate method can be called successfully
 * with valid inputs and completes without throwing exceptions or CUDA errors.
 */
TEST_F(SE3BetweenFactorBatchTest, Evaluate) {
  auto test_range = this->profiler_domain_.CreateDomainRange("Evaluate");
  SE3BetweenFactorBatch factor_batch(cublas_handle_,
                                             pose_deltas_device_.data(),
                                             num_factors_);
  CudaStream stream;

  float* residuals_ptr = residuals_device_.data();
  float* jacobians_ptr = jacobians_device_.data();
  const float* const* state_ptrs =
      reinterpret_cast<const float* const*>(state_ptrs_device_.data());

  // Test that Evaluate executes without errors
  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    factor_batch.Evaluate(residuals_ptr, jacobians_ptr, state_ptrs,
                           stream.GetStream());

    // Synchronize to ensure execution completes
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate");
    factor_batch.Evaluate(residuals_ptr, jacobians_ptr, state_ptrs,
                           stream.GetStream());

    // Synchronize to ensure execution completes
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }
}

/**
 * @brief Tests that Evaluate executes without errors when jacobians is nullptr.
 *
 * This test verifies that the Evaluate method can be called successfully
 * with jacobians set to nullptr (when jacobians are not needed).
 */
TEST_F(SE3BetweenFactorBatchTest, EvaluateWithoutJacobians) {
  auto test_range =
      this->profiler_domain_.CreateDomainRange("EvaluateWithoutJacobians");
  SE3BetweenFactorBatch factor_batch(cublas_handle_,
                                             pose_deltas_device_.data(),
                                             num_factors_);
  CudaStream stream;

  float* residuals_ptr = residuals_device_.data();
  const float* const* state_ptrs =
      reinterpret_cast<const float* const*>(state_ptrs_device_.data());

  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    // Test that Evaluate executes without errors when jacobians is nullptr
    factor_batch.Evaluate(residuals_ptr, nullptr, state_ptrs,
                           stream.GetStream());

    // Synchronize to ensure execution completes
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate");
    // Test that Evaluate executes without errors when jacobians is nullptr
    factor_batch.Evaluate(residuals_ptr, nullptr, state_ptrs,
                           stream.GetStream());

    // Synchronize to ensure execution completes
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }
}

/**
 * @brief Tests that InformationFactorBatch wrapper executes without
 * errors.
 *
 * This test verifies that the InformationFactorBatch wrapper around
 * SE3BetweenFactorBatch can be called successfully with valid inputs
 * and completes without throwing exceptions or CUDA errors.
 */
TEST_F(SE3BetweenFactorBatchTest, InformationFactorBatch) {
  auto test_range = this->profiler_domain_.CreateDomainRange("Evaluate");

  std::vector<Matrix<6>> sqrt_information_matrices_host(num_factors_);
  for (size_t i = 0; i < num_factors_; i++) {
    sqrt_information_matrices_host[i].fill(0.0f);
    for (int j = 0; j < 6; j++) {
      sqrt_information_matrices_host[i][j * 6 + j] = 1.0f;
    }
  }
  DeviceVector<Matrix<6>> sqrt_information_matrices_device(sqrt_information_matrices_host);

  InformationFactorBatch<SE3BetweenFactorBatch> factor_batch(
      cublas_handle_, sqrt_information_matrices_device.data(), num_factors_,
      pose_deltas_device_.data(), num_factors_);
  CudaStream stream;

  float* residuals_ptr = residuals_device_.data();
  float* jacobians_ptr = jacobians_device_.data();
  const float* const* state_ptrs =
      reinterpret_cast<const float* const*>(state_ptrs_device_.data());

  // Test that Evaluate executes without errors
  {
    // WARMUP
    auto range = this->profiler_domain_.CreateDomainRange("Warmup");
    factor_batch.Evaluate(residuals_ptr, jacobians_ptr, state_ptrs,
                           stream.GetStream());

    // Synchronize to ensure execution completes
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate");
    factor_batch.Evaluate(residuals_ptr, jacobians_ptr, state_ptrs,
                           stream.GetStream());

    // Synchronize to ensure execution completes
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }
}

}  // namespace cunls
