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

/** @file jacobian_ops_test.cpp
 *  @brief Tests for Jacobian sparse structure construction on GPU.
 */

#include "cunls/minimizer/minimizer_state.h"

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <random>

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
 * @brief Test fixture for Jacobian operations with typed vector sizes.
 *
 * Sets up a randomly-sized optimization problem with constant and non-constant
 * states, then verifies GPU-built triplet sparse structures are consistent
 * across two MinimizerState builds.
 *
 * @tparam VectorSize Compile-time vector dimension.
 */
template <class VectorSize>
class JacobianOpsTest : public ::testing::Test {
 public:
  static constexpr int kDim = VectorSize::size;

  /** @brief Generates a randomly-sized optimization problem with ~20% constant states. */
  void SetUp() override {
    unsigned int fixed_seed = 0;
    std::mt19937 gen(fixed_seed);

    std::uniform_int_distribution<size_t> states_distrib(1000, 100000);
    std::uniform_int_distribution<size_t> cost_batches_distrib(1, 10);
    num_vectors = states_distrib(gen);
    num_factor_batches = cost_batches_distrib(gen);

    std::uniform_real_distribution<float> unit_distr(0, 1);
    for (size_t i = 0; i < num_vectors; i++) {
      // 20% of the states are constant
      if (unit_distr(gen) > 0.8) {
        constant_state_ids.push_back(i);
      }
    }
  }

  size_t num_vectors;
  size_t num_factor_batches;
  std::vector<int> constant_state_ids;

  profiler::Domain profiler_domain_{"JacobianOpsTest"};
};

typedef ::testing::Types<test_utils::Size<1>, test_utils::Size<2>,
                         test_utils::Size<3>, test_utils::Size<4>>
    VectorSizes;
TYPED_TEST_CASE(JacobianOpsTest, VectorSizes);

/** @brief Verifies that two GPU triplet builds match (idempotent structure). */
TYPED_TEST(JacobianOpsTest, BuildTripletSparseStructure) {
  auto test_range = this->profiler_domain_.CreateDomainRange(
      "BuildTripletSparseStructureTest");
  // Prepare inputs
  auto seq_vecs =
      test_utils::MakeSequentialVectors<TestFixture::kDim>(this->num_vectors);
  test_utils::VectorStateData<TestFixture::kDim> state_data(
      seq_vecs, this->constant_state_ids);
  auto& vector_states = state_data.get();
  auto device_pointers =
      test_utils::CollectStatePointers(vector_states);
  auto obs_vecs = test_utils::MakeConstantVectors<TestFixture::kDim>(
      this->num_vectors, 1.f);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data(
      obs_vecs);
  auto& factor_batch = factor_data.get();

  // Build the optimization problem
  Problem problem;
  problem.AddStateBatch(&vector_states);

  for (size_t i = 0; i < this->num_factor_batches; i++) {
    problem.AddFactorBatch(&factor_batch, device_pointers);
  }

  ASSERT_TRUE(problem.CheckConsistency());

  CudaStream stream;

  TripletSparseStructure structure_a;
  TripletSparseStructure structure_b;
  {
    auto range =
        this->profiler_domain_.CreateDomainRange("BuildTripletSparseStructure");
    MinimizerState ms_a;
    ms_a.BuildTripletSparseStructure(stream.GetStream(), problem, structure_a);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    MinimizerState ms_b;
    ms_b.BuildTripletSparseStructure(stream.GetStream(), problem, structure_b);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  ASSERT_EQ(structure_a.row_ids.size(), structure_b.row_ids.size());
  ASSERT_EQ(structure_a.col_ids.size(), structure_b.col_ids.size());

  std::vector<int> row_ids(structure_a.row_ids.size());
  structure_a.row_ids.CopyToHost(row_ids.data(), row_ids.size());
  std::vector<int> row_ids_b(structure_b.row_ids.size());
  structure_b.row_ids.CopyToHost(row_ids_b.data(), row_ids_b.size());
  for (size_t i = 0; i < structure_a.row_ids.size(); i++) {
    ASSERT_EQ(row_ids[i], row_ids_b[i]);
  }

  std::vector<int> col_ids(structure_a.col_ids.size());
  structure_a.col_ids.CopyToHost(col_ids.data(), col_ids.size());
  std::vector<int> col_ids_b(structure_b.col_ids.size());
  structure_b.col_ids.CopyToHost(col_ids_b.data(), col_ids_b.size());
  for (size_t i = 0; i < structure_a.col_ids.size(); i++) {
    ASSERT_EQ(col_ids[i], col_ids_b[i]);
  }
}
}  // namespace cunls
