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

/** @file prior_vector_prior_factor_test.cpp
 *  @brief Tests for PriorVectorFactorBatch residual and Jacobian evaluation.
 */

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

/**
 * @brief Test fixture for prior vector factor evaluation.
 *
 * Typed by vector dimension; evaluates residuals and Jacobians of
 * PriorVectorFactorBatch against expected analytic results.
 *
 * @tparam VectorSize Compile-time vector dimension.
 */
template <class VectorSize>
class PriorVectorCostTest : public ::testing::Test {
 public:
  static constexpr int kDim = VectorSize::size;

  /** @brief Allocates device buffers for residuals and Jacobians. */
  void SetUp() override {
    const int residual_size = num_vectors_ * VectorSize::size;
    residuals_.resize(residual_size);
    jacobians_.resize(residual_size * VectorSize::size);
  }

  const size_t num_vectors_ = 100;
  DeviceVector<float> residuals_;
  DeviceVector<float> jacobians_;

  profiler::Domain profiler_domain_{"PriorVectorCostTest"};
};

typedef ::testing::Types<test_utils::Size<1>, test_utils::Size<2>,
                         test_utils::Size<3>, test_utils::Size<4>>
    VectorSizes;
TYPED_TEST_CASE(PriorVectorCostTest, VectorSizes);

/** @brief Verifies that residual evaluation produces correct (state - observation) values. */
TYPED_TEST(PriorVectorCostTest, Residual) {
  auto test_range = this->profiler_domain_.CreateDomainRange("ResidualTest");
  CudaStream stream;

  auto seq_vecs =
      test_utils::MakeSequentialVectors<TestFixture::kDim>(this->num_vectors_);
  test_utils::VectorStateData<TestFixture::kDim> states_data(seq_vecs);
  auto& states = states_data.get();
  auto state_pointers = test_utils::CollectStatePointersDevice(states);
  auto obs_vecs = test_utils::MakeConstantVectors<TestFixture::kDim>(
      this->num_vectors_, 1.f);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data(
      obs_vecs);
  auto& factor_batch = factor_data.get();

  float* residuals_ptr = this->residuals_.data();
  float** state_ptrs = state_pointers.data();
  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate Residual");
    factor_batch.Evaluate(residuals_ptr, nullptr, state_ptrs, stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  const int size = TypeParam::size;

  // Copy results back to host for verification
  std::vector<float> host_residuals(this->residuals_.size());
  this->residuals_.CopyToHost(host_residuals.data(), host_residuals.size());

  for (size_t i = 0; i < this->num_vectors_; i++) {
    float x = static_cast<float>(i);
    for (size_t j = 0; j < size; j++) {
      ASSERT_NEAR(host_residuals[size * i + j], x - 1.f, 1e-5);
    }
  }
}

/** @brief Verifies that StateBlockSizes() reports the correct single block size. */
TYPED_TEST(PriorVectorCostTest, StateBlockSizes) {
  auto obs_vecs = test_utils::MakeConstantVectors<TestFixture::kDim>(
      this->num_vectors_, 1.f);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data(
      obs_vecs);
  auto& factor_batch = factor_data.get();

  auto state_block_sizes = factor_batch.StateBlockSizes();
  ASSERT_EQ(state_block_sizes.size(), 1);
  ASSERT_EQ(state_block_sizes[0], TypeParam::size);
}

/** @brief Verifies that Jacobian evaluation produces the expected identity matrix per block. */
TYPED_TEST(PriorVectorCostTest, Jacobian) {
  auto test_range = this->profiler_domain_.CreateDomainRange("JacobianTest");
  CudaStream stream;

  auto seq_vecs =
      test_utils::MakeSequentialVectors<TestFixture::kDim>(this->num_vectors_);
  test_utils::VectorStateData<TestFixture::kDim> states_data(seq_vecs);
  auto& states = states_data.get();
  auto state_pointers = test_utils::CollectStatePointersDevice(states);
  auto obs_vecs = test_utils::MakeConstantVectors<TestFixture::kDim>(
      this->num_vectors_, 1.f);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data(
      obs_vecs);
  auto& factor_batch = factor_data.get();

  float* jac_ptr = this->jacobians_.data();
  float** state_ptrs = state_pointers.data();
  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate Jacobian");
    factor_batch.Evaluate(nullptr, jac_ptr, state_ptrs, stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }
  const int size = TypeParam::size;

  // Copy results back to host for verification
  std::vector<float> host_jacobians(this->jacobians_.size());
  this->jacobians_.CopyToHost(host_jacobians.data(), host_jacobians.size());

  for (size_t k = 0; k < this->num_vectors_; k++) {
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        float value = i == j ? 1.f : 0;
        ASSERT_NEAR(host_jacobians[(size * k + i) * size + j], value, 1e-5);
      }
    }
  }
}

}  // namespace cunls
