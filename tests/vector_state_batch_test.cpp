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
 * @file vector_state_batch_test.cpp
 * @brief Unit tests for VectorStateBatch (Plus, TangentSize, AmbientSize, NumStateBlocks).
 *
 * Validates vector state batch operations across multiple vector
 * dimensionalities using typed tests (1, 2, 3, 4).
 */

#include "cunls/state/vector_state_batch.h"

#include <gtest/gtest.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "tests/utils.h"

namespace cunls {

/**
 * @brief Typed test fixture for VectorStateBatch tests across multiple dimensions.
 *
 * Initializes device vectors with sequential values and unit deltas for
 * verifying Plus, TangentSize, AmbientSize, and NumStateBlocks.
 */
template <class VectorSize>
class VectorStateBatchTest : public ::testing::Test {
 public:
  static constexpr int kDim = VectorSize::size;

  void SetUp() override {
    const size_t total_floats = num_vectors_ * kDim;

    // Initialize host vectors first, then copy to device
    std::vector<float> host_vectors(total_floats);
    std::vector<float> host_deltas(total_floats);
    for (size_t i = 0; i < num_vectors_; i++) {
      float x = static_cast<float>(i);
      for (int j = 0; j < kDim; j++) {
        host_vectors[i * kDim + j] = x;
        host_deltas[i * kDim + j] = 1.f;
      }
    }
    vectors_ = DeviceVector<float>(host_vectors);
    deltas_ = DeviceVector<float>(host_deltas);
    results_ = DeviceVector<float>(total_floats);
  }

  const size_t num_vectors_ = 100;
  DeviceVector<float> deltas_;
  DeviceVector<float> results_;
  DeviceVector<float> vectors_;

  profiler::Domain profiler_domain_{"VectorStateBatchTest"};
};

typedef ::testing::Types<test_utils::Size<1>, test_utils::Size<2>,
                         test_utils::Size<3>, test_utils::Size<4>>
    VectorSizes;
TYPED_TEST_CASE(VectorStateBatchTest, VectorSizes);

/** @brief Tests that Plus correctly adds delta vectors to state vectors. */
TYPED_TEST(VectorStateBatchTest, Plus) {
  auto test_range = this->profiler_domain_.CreateDomainRange("PlusTest");
  constexpr int kDim = TypeParam::size;
  const float* data_ptr = this->vectors_.data();
  VectorStateBatch<kDim> vector_states(data_ptr, this->num_vectors_);

  const float* deltas_ptr = this->deltas_.data();
  float* results_ptr = this->results_.data();

  CudaStream stream;
  {
    auto range = this->profiler_domain_.CreateDomainRange("Plus");
    vector_states.Plus(data_ptr, deltas_ptr, results_ptr,
                       stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results to host for verification
  std::vector<float> results_host(this->num_vectors_ * kDim);
  this->results_.CopyToHost(results_host.data(), results_host.size());

  for (size_t i = 0; i < this->num_vectors_; i++) {
    float x = static_cast<float>(i);
    for (int j = 0; j < kDim; j++) {
      ASSERT_NEAR(x + 1.f, results_host[i * kDim + j], 1e-4);
    }
  }
}

/** @brief Tests that TangentSize returns the correct vector dimension. */
TYPED_TEST(VectorStateBatchTest, TangentSize) {
  constexpr int kDim = TypeParam::size;
  const float* data_ptr = this->vectors_.data();
  VectorStateBatch<kDim> vector_states(data_ptr, this->num_vectors_);
  auto test_range = this->profiler_domain_.CreateDomainRange("TangentSize");
  ASSERT_EQ(vector_states.TangentSize(), kDim);
}

/** @brief Tests that AmbientSize returns the correct vector dimension. */
TYPED_TEST(VectorStateBatchTest, AmbientSize) {
  constexpr int kDim = TypeParam::size;
  const float* data_ptr = this->vectors_.data();
  VectorStateBatch<kDim> vector_states(data_ptr, this->num_vectors_);
  auto test_range = this->profiler_domain_.CreateDomainRange("AmbientSize");
  ASSERT_EQ(vector_states.AmbientSize(), kDim);
}

/** @brief Tests that NumStateBlocks returns the correct batch count. */
TYPED_TEST(VectorStateBatchTest, NumStateBlocks) {
  constexpr int kDim = TypeParam::size;
  const float* data_ptr = this->vectors_.data();
  VectorStateBatch<kDim> vector_states(data_ptr, this->num_vectors_);
  auto test_range =
      this->profiler_domain_.CreateDomainRange("NumStateBlocks");
  ASSERT_EQ(vector_states.NumStateBlocks(), this->num_vectors_);
}

}  // namespace cunls
