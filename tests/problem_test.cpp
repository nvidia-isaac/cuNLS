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

/** @file problem_test.cpp
 *  @brief Tests for Problem consistency checking with various state/cost
 * configurations.
 */

#include "cunls/minimizer/problem.h"

#include <gtest/gtest.h>

#include <memory>

#include "cunls/common/device_vector.h"
#include "cunls/common/log.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

/**
 * @brief Test fixture for Problem consistency validation.
 *
 * Typed by vector dimension; verifies that Problem::CheckConsistency
 * correctly detects valid and invalid problem configurations.
 *
 * @tparam VectorSize Compile-time vector dimension.
 */
template <class VectorSize>
class ProblemConsistencyTest : public ::testing::Test {
public:
  static constexpr int kDim = VectorSize::size;

  const size_t num_vectors = 100;
  profiler::Domain profiler_domain_{"ProblemConsistencyTest"};
};

typedef ::testing::Types<test_utils::Size<1>, test_utils::Size<2>,
                         test_utils::Size<3>, test_utils::Size<4>>
    VectorSizes;
TYPED_TEST_CASE(ProblemConsistencyTest, VectorSizes);

/** @brief Verifies that a problem is inconsistent without states and consistent
 * once added. */
TYPED_TEST(ProblemConsistencyTest, CheckConsistencySimple) {
  auto seq_vecs =
      test_utils::MakeSequentialVectors<TestFixture::kDim>(this->num_vectors);
  test_utils::VectorStateData<TestFixture::kDim> state_data(seq_vecs);
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);
  auto obs_vecs = test_utils::MakeConstantVectors<TestFixture::kDim>(
      this->num_vectors, 1.f);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data(obs_vecs);
  auto &factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, device_pointers);
  ASSERT_FALSE(problem.CheckConsistency());

  problem.AddStateBatch(&vector_states);

  auto test_range =
      this->profiler_domain_.CreateDomainRange("CheckConsistencySimple");
  ASSERT_TRUE(problem.CheckConsistency());
}

/** @brief Verifies consistency with one state batch and multiple factor
 * batches. */
TYPED_TEST(ProblemConsistencyTest, CheckConsistencyOneToMany) {
  auto seq_vecs =
      test_utils::MakeSequentialVectors<TestFixture::kDim>(this->num_vectors);
  test_utils::VectorStateData<TestFixture::kDim> state_data(seq_vecs);
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);
  auto obs_vecs = test_utils::MakeConstantVectors<TestFixture::kDim>(
      this->num_vectors, 1.f);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data_1(obs_vecs);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data_2(obs_vecs);
  auto &factor_batch_1 = factor_data_1.get();
  auto &factor_batch_2 = factor_data_2.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch_1, device_pointers);
  problem.AddFactorBatch(&factor_batch_2, device_pointers);
  ASSERT_FALSE(problem.CheckConsistency());

  problem.AddStateBatch(&vector_states);
  ASSERT_TRUE(problem.CheckConsistency());

  auto obs_vecs_3 = test_utils::MakeConstantVectors<TestFixture::kDim>(1, 1.f);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data_3(obs_vecs_3);
  auto &factor_batch_3 = factor_data_3.get();
  problem.AddFactorBatch(&factor_batch_3, {nullptr});

  auto test_range =
      this->profiler_domain_.CreateDomainRange("CheckConsistencyOneToMany");
  ASSERT_FALSE(problem.CheckConsistency());
}

/** @brief Verifies consistency with multiple state batches linked to one factor
 * batch. */
TYPED_TEST(ProblemConsistencyTest, CheckConsistencyManyToOne) {
  auto seq_vecs =
      test_utils::MakeSequentialVectors<TestFixture::kDim>(this->num_vectors);
  test_utils::VectorStateData<TestFixture::kDim> state_data_1(seq_vecs);
  test_utils::VectorStateData<TestFixture::kDim> state_data_2(seq_vecs);
  auto &state_batch_1 = state_data_1.get();
  auto &state_batch_2 = state_data_2.get();

  std::vector<float *> state_pointers;
  for (size_t i = 0; i < this->num_vectors; i++) {
    state_pointers.push_back(state_batch_1.StateBlockDevicePtr(i));
    state_pointers.push_back(state_batch_2.StateBlockDevicePtr(i));
  }
  auto obs_vecs = test_utils::MakeConstantVectors<TestFixture::kDim>(
      this->num_vectors * 2, 1.f);
  test_utils::PriorFactorData<TestFixture::kDim> factor_data(obs_vecs);
  auto &factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, state_pointers);
  ASSERT_FALSE(problem.CheckConsistency());

  problem.AddStateBatch(&state_batch_1);
  ASSERT_FALSE(problem.CheckConsistency());

  problem.AddStateBatch(&state_batch_2);
  auto test_range =
      this->profiler_domain_.CreateDomainRange("CheckConsistencyManyToOne");
  ASSERT_TRUE(problem.CheckConsistency());
}

} // namespace cunls
