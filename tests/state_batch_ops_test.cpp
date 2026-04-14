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

/** @file state_batch_ops_test.cpp
 *  @brief Tests for state batch operations (mapping and Plus update).
 */

#include "cunls/state/state_batch_ops.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <numeric>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

/**
 * @brief Test fixture for StateBatchOps mapping and Plus operations.
 *
 * Inherits from StateBatchOps to test internal mapping and update logic
 * across multiple state batches of different dimensions.
 */
class StateBatchOpsTest : public StateBatchOps, public ::testing::Test {
protected:
  /**
   * @brief Verifies that state values are correct after a Plus update.
   *
   * Constant states should remain 0; non-constant should be updated to 1.
   *
   * @tparam Dim Dimension of each vector state block.
   * @param state_batch The state batch to verify.
   */
  template <int Dim>
  void TestStateValues(const VectorStateBatch<Dim> &state_batch) {
    auto values = test_utils::CopyStateToHost(state_batch);

    for (size_t i = 0; i < values.size(); i++) {
      float result = i < num_const_states ? 0 : 1.f;
      for (auto x : values[i]) {
        ASSERT_NEAR(x, result, 1e-5);
      }
    }
  }

  const size_t num_vectors_ = 1000000;

  const size_t num_const_states = 10;

  profiler::Domain profiler_domain_{"StateBatchOpsTest"};
};

/** @brief Verifies that Preprocess builds the correct reduced-state index
 * mapping. */
TEST_F(StateBatchOpsTest, Map) {
  auto test_range = this->profiler_domain_.CreateDomainRange("Map");
  // Check that the mapping in correct
  CudaStream stream;

  auto const_ids = test_utils::MakeSequentialIds(this->num_const_states);
  test_utils::VectorStateData<2> points_2d(
      test_utils::MakeZeroVectors<2>(this->num_vectors_), const_ids);
  test_utils::VectorStateData<3> points_3d(
      test_utils::MakeZeroVectors<3>(this->num_vectors_), const_ids);
  test_utils::VectorStateData<4> points_4d(
      test_utils::MakeZeroVectors<4>(this->num_vectors_), const_ids);
  test_utils::VectorStateData<5> points_5d(
      test_utils::MakeZeroVectors<5>(this->num_vectors_), const_ids);

  // Create the set on state batches
  std::vector<StateBatch *> state_batches = {points_2d.ptr(), points_3d.ptr(),
                                             points_4d.ptr(), points_5d.ptr()};

  // Build the mapping
  {
    auto range = this->profiler_domain_.CreateDomainRange("Preprocess");
    this->Preprocess(stream.GetStream(), state_batches);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Calculate ground truth for the mapping
  size_t num_reduced_states = 0;
  std::vector<int> gt_map;
  size_t N = 0;
  for (auto pb : state_batches) {
    size_t tangent_size = pb->TangentSize();
    size_t num_state_blocks = pb->NumStateBlocks();
    size_t num_const_blocks = pb->NumConstStateBlocks();

    num_reduced_states += tangent_size * (num_state_blocks - num_const_blocks);

    for (int i = num_const_blocks * tangent_size;
         i < num_state_blocks * tangent_size; i++) {
      gt_map.push_back(N + i);
    }
    N += num_state_blocks * tangent_size;
  }

  // Check the number of reduced states is correct
  ASSERT_EQ(this->NumReducedStates(), num_reduced_states);
  std::vector<int> predicted_map(this->map_.size());
  this->map_.CopyToHost(predicted_map.data(), predicted_map.size());

  // Check the mapping is right
  for (int i = 0; i < num_reduced_states; i++) {
    ASSERT_EQ(gt_map[i], predicted_map[i]);
  }
}

/** @brief Verifies that the Plus operation correctly applies a delta to
 * non-constant states. */
TEST_F(StateBatchOpsTest, Plus) {
  auto test_range = this->profiler_domain_.CreateDomainRange("PlusTest");
  // Check the Plus operation works right
  CudaStream stream;

  auto const_ids = test_utils::MakeSequentialIds(this->num_const_states);
  test_utils::VectorStateData<2> points_2d(
      test_utils::MakeZeroVectors<2>(this->num_vectors_), const_ids);
  test_utils::VectorStateData<3> points_3d(
      test_utils::MakeZeroVectors<3>(this->num_vectors_), const_ids);
  test_utils::VectorStateData<4> points_4d(
      test_utils::MakeZeroVectors<4>(this->num_vectors_), const_ids);
  test_utils::VectorStateData<5> points_5d(
      test_utils::MakeZeroVectors<5>(this->num_vectors_), const_ids);

  // Create the set on state batches
  std::vector<StateBatch *> state_batches = {points_2d.ptr(), points_3d.ptr(),
                                             points_4d.ptr(), points_5d.ptr()};

  // Collect the pointers for state batches
  std::vector<const float *> input_pointers;
  std::vector<float *> output_pointers;
  for (auto pbatch : state_batches) {
    auto ptr = pbatch->StateBlockDevicePtr(0);
    input_pointers.push_back(ptr);
    output_pointers.push_back(ptr);
  }

  // Build the mapping
  this->Preprocess(stream.GetStream(), state_batches);

  DeviceVector<float> delta(this->NumReducedStates());
  std::vector<float> host_delta(this->NumReducedStates(), 1.);
  delta.CopyFromHost(host_delta.data(), host_delta.size());

  // Run Plus
  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate Plus");
    this->Plus(stream.GetStream(), input_pointers, delta, output_pointers);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Check the Plus is correct
  this->TestStateValues(points_2d.get());
  this->TestStateValues(points_3d.get());
  this->TestStateValues(points_4d.get());
  this->TestStateValues(points_5d.get());
}
} // namespace cunls
