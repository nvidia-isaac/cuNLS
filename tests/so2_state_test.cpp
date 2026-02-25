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
 * @file so2_state_test.cpp
 * @brief Unit tests for SO2StateBatch dimensions and block count.
 *
 * Validates that TangentSize, AmbientSize, and NumStateBlocks return
 * the correct values for SO(2) state batches.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/profiler.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Test fixture for SO2StateBatch unit tests.
 *
 * Sets up test data including initial rotation matrices (identity)
 * and random delta values (angles) for testing Plus operations.
 */
class SO2StateBatchTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::vector<Matrix<2>> host_rotations(num_rotations_);
    std::vector<Vector<1>> host_deltas(num_rotations_);
    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> angle_dist(-0.1f, 0.1f);

    // Initialize rotations as identity matrices (row-major)
    for (size_t i = 0; i < num_rotations_; i++) {
      Matrix<2>& R = host_rotations[i];
      // Set to identity matrix: [[1, 0], [0, 1]]
      R.fill(0.0f);
      R[0] = 1.0f;  // [0,0]
      R[3] = 1.0f;  // [1,1]

      // Initialize delta as random angle
      Vector<1>& delta = host_deltas[i];
      delta[0] = angle_dist(rng);
    }

    // Copy to device
    rotations_ = DeviceVector<Matrix<2>>(host_rotations);
    deltas_ = DeviceVector<Vector<1>>(host_deltas);
  }

  const size_t num_rotations_ = 10000;
  const uint32_t fixed_seed_ = 42;
  DeviceVector<Matrix<2>> rotations_;
  DeviceVector<Vector<1>> deltas_;

  profiler::Domain profiler_domain_{"SO2StateBatchTest"};
};

/**
 * @brief Tests that TangentSize returns the correct dimension (1).
 *
 * The tangent space of SO(2) has dimension 1: a scalar angle in radians.
 */
TEST_F(SO2StateBatchTest, TangentSize) {
  cuBLASHandle cublas_handle;
  const float* data_ptr = reinterpret_cast<const float*>(
      this->rotations_.data());
  SO2StateBatch so2_states(cublas_handle, data_ptr, this->num_rotations_);
  auto test_range = this->profiler_domain_.CreateDomainRange("TangentSize");
  ASSERT_EQ(so2_states.TangentSize(), 1);
}

/**
 * @brief Tests that AmbientSize returns the correct dimension (4).
 *
 * The ambient space is a 2x2 rotation matrix,
 * stored as 4 floats in row-major order.
 */
TEST_F(SO2StateBatchTest, AmbientSize) {
  cuBLASHandle cublas_handle;
  const float* data_ptr = reinterpret_cast<const float*>(
      this->rotations_.data());
  SO2StateBatch so2_states(cublas_handle, data_ptr, this->num_rotations_);
  auto test_range = this->profiler_domain_.CreateDomainRange("AmbientSize");
  ASSERT_EQ(so2_states.AmbientSize(), 4);
}

/**
 * @brief Tests that NumStateBlocks returns the correct count.
 *
 * Verifies that the batch correctly reports the number of state blocks
 * (rotation matrices) it contains.
 */
TEST_F(SO2StateBatchTest, NumStateBlocks) {
  cuBLASHandle cublas_handle;
  const float* data_ptr = reinterpret_cast<const float*>(
      this->rotations_.data());
  SO2StateBatch so2_states(cublas_handle, data_ptr, this->num_rotations_);
  auto test_range =
      this->profiler_domain_.CreateDomainRange("NumStateBlocks");
  ASSERT_EQ(so2_states.NumStateBlocks(), this->num_rotations_);
}

}  // namespace cunls
