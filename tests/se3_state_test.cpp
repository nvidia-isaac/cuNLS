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
 * @file se3_state_test.cpp
 * @brief Unit tests for SE3StateBatch dimensions and block count.
 *
 * Validates that TangentSize, AmbientSize, and NumStateBlocks return
 * the correct values for SE(3) state batches.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/profiler.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Test fixture for SE3StateBatch unit tests.
 *
 * Sets up test data including initial transformation matrices (identity)
 * and random delta vectors (twists) for testing Plus operations.
 */
class SE3StateBatchTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::vector<SE3Transform> host_transforms(num_transforms_);
    std::vector<Vector<6>> host_deltas(num_transforms_);
    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> rotation_dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> translation_dist(-1.0f, 1.0f);

    // Initialize transforms as identity matrices (row-major)
    for (size_t i = 0; i < num_transforms_; i++) {
      SE3Transform& T = host_transforms[i];
      // Set to identity matrix
      T.fill(0.0f);
      T[0] = 1.0f;   // [0,0]
      T[5] = 1.0f;   // [1,1]
      T[10] = 1.0f;  // [2,2]
      T[15] = 1.0f;  // [3,3]

      // Initialize delta as random twist (rotation + translation)
      Vector<6>& delta = host_deltas[i];
      delta[0] = rotation_dist(rng);     // rotation x
      delta[1] = rotation_dist(rng);     // rotation y
      delta[2] = rotation_dist(rng);     // rotation z
      delta[3] = translation_dist(rng);  // translation x
      delta[4] = translation_dist(rng);  // translation y
      delta[5] = translation_dist(rng);  // translation z
    }

    // Copy to device
    transforms_ = DeviceVector<SE3Transform>(host_transforms);
    deltas_ = DeviceVector<Vector<6>>(host_deltas);
  }

  const size_t num_transforms_ = 10000;
  const uint32_t fixed_seed_ = 42;
  DeviceVector<SE3Transform> transforms_;
  DeviceVector<Vector<6>> deltas_;

  profiler::Domain profiler_domain_{"SE3StateBatchTest"};
};

/**
 * @brief Tests that TangentSize returns the correct dimension (6).
 *
 * The tangent space of SE(3) has dimension 6: 3 for rotation (SO(3))
 * and 3 for translation.
 */
TEST_F(SE3StateBatchTest, TangentSize) {
  cuBLASHandle cublas_handle;
  const float* data_ptr = reinterpret_cast<const float*>(
      this->transforms_.data());
  SE3StateBatch se3_states(cublas_handle, data_ptr, this->num_transforms_);
  auto test_range = this->profiler_domain_.CreateDomainRange("TangentSize");
  ASSERT_EQ(se3_states.TangentSize(), 6);
}

/**
 * @brief Tests that AmbientSize returns the correct dimension (16).
 *
 * The ambient space is a 4x4 homogeneous transformation matrix,
 * stored as 16 floats in row-major order.
 */
TEST_F(SE3StateBatchTest, AmbientSize) {
  cuBLASHandle cublas_handle;
  const float* data_ptr = reinterpret_cast<const float*>(
      this->transforms_.data());
  SE3StateBatch se3_states(cublas_handle, data_ptr, this->num_transforms_);
  auto test_range = this->profiler_domain_.CreateDomainRange("AmbientSize");
  ASSERT_EQ(se3_states.AmbientSize(), 16);
}

/**
 * @brief Tests that NumStateBlocks returns the correct count.
 *
 * Verifies that the batch correctly reports the number of state blocks
 * (transformation matrices) it contains.
 */
TEST_F(SE3StateBatchTest, NumStateBlocks) {
  cuBLASHandle cublas_handle;
  const float* data_ptr = reinterpret_cast<const float*>(
      this->transforms_.data());
  SE3StateBatch se3_states(cublas_handle, data_ptr, this->num_transforms_);
  auto test_range =
      this->profiler_domain_.CreateDomainRange("NumStateBlocks");
  ASSERT_EQ(se3_states.NumStateBlocks(), this->num_transforms_);
}

}  // namespace cunls
