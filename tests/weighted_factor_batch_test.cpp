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

/** @file weighted_factor_batch_test.cpp
 *  @brief Tests for WeightedFactorBatch convergence with Levenberg-Marquardt.
 *
 *  Verifies that wrapping a PriorVectorFactorBatch in a WeightedFactorBatch
 *  (both uniform and per-factor weights) still converges to the correct
 *  solution under LM optimization.
 */

#include "cunls/factor/weighted_factor_batch.h"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

template <class TestParam>
class WeightedFactorBatchTest : public ::testing::Test {
 public:
  static constexpr int kDim = TestParam::vector_size;
  using StatesType = VectorStateBatch<kDim>;
  using VectorType = Vector<kDim>;
  using StateData = test_utils::VectorStateData<kDim>;

  void SetUp() override {
    observations_.resize(num_vectors_);
    state_values_.resize(num_vectors_);
    for (size_t i = 0; i < num_vectors_; i++) {
      float x = static_cast<float>(i);
      observations_[i].fill(x - 1.f);
      state_values_[i].fill(x);
    }

    test_utils::ConfigureTestSolver(TestParam::solver_id, minimizer_options_);
  }

  void CheckConvergence(const StatesType &states) {
    size_t num_blocks = states.NumActiveStates();
    auto ptr = reinterpret_cast<const VectorType *>(states.StateDevicePtr(0));

    std::vector<VectorType> host_states(num_blocks);
    THROW_ON_CUDA_ERROR(cudaMemcpy(host_states.data(), ptr, num_blocks * sizeof(VectorType),
                                   cudaMemcpyDeviceToHost));

    ASSERT_EQ(host_states.size(), observations_.size());
    for (size_t i = 0; i < num_blocks; i++) {
      const auto &obs = observations_[i];
      const auto &state_vals = host_states[i];
      for (size_t j = 0; j < TestParam::vector_size; j++) {
        ASSERT_NEAR(obs[j], state_vals[j], 1e-3);
      }
    }
  }

  const size_t num_vectors_ = 10000;

  std::vector<VectorType> observations_;
  std::vector<VectorType> state_values_;

  MinimizerOptions minimizer_options_{.disable_safety_checks = false};
};

template <int VectorSize, int SolverId>
struct TestParam {
  static constexpr int vector_size = VectorSize;
  static constexpr int solver_id = SolverId;
};

// The wrapper scales rows; it does not depend on the vector size or the linear
// solver (both covered elsewhere): one of each.
typedef ::testing::Types<TestParam<3, test_utils::kBlockSparseCholeskySolverId>> WeightedTestParams;
TYPED_TEST_SUITE(WeightedFactorBatchTest, WeightedTestParams);

/**
 * @brief LM convergence with a uniform scalar weight.
 *
 * Wraps PriorVectorFactorBatch in WeightedFactorBatch(weight=2.0) and
 * verifies that LM converges to the same optimum as the unweighted case.
 */
TYPED_TEST(WeightedFactorBatchTest, UniformWeightLM) {
  using FactorType = PriorVectorFactorBatch<TestFixture::kDim>;
  using WeightedType = WeightedFactorBatch<FactorType>;

  typename TestFixture::StateData state_data(this->state_values_);
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);

  DeviceVector<typename TestFixture::VectorType> obs_device(this->observations_);

  const float weight = 2.0f;
  WeightedType weighted_factor(weight, obs_device.data(), this->num_vectors_);
  weighted_factor.SetNumActiveFactors(weighted_factor.Capacity());

  Problem problem;
  problem.AddFactorBatch(&weighted_factor, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = this->minimizer_options_;
  LevenbergMarquardtMinimizer minimizer(lm_options);
  auto summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  ASSERT_TRUE(std::isfinite(summary.initial_cost));
  ASSERT_TRUE(std::isfinite(summary.final_cost));
  ASSERT_LE(summary.final_cost, summary.initial_cost);

  this->CheckConvergence(vector_states);
}

/**
 * @brief LM convergence with per-factor weights.
 *
 * Provides a distinct weight for each factor and verifies that LM still
 * converges to the correct optimum.
 */
TYPED_TEST(WeightedFactorBatchTest, PerFactorWeightLM) {
  using FactorType = PriorVectorFactorBatch<TestFixture::kDim>;
  using WeightedType = WeightedFactorBatch<FactorType>;

  typename TestFixture::StateData state_data(this->state_values_);
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);

  DeviceVector<typename TestFixture::VectorType> obs_device(this->observations_);

  std::vector<float> host_weights(this->num_vectors_);
  for (size_t i = 0; i < this->num_vectors_; i++) {
    host_weights[i] = 0.5f + static_cast<float>(i % 10) * 0.1f;
  }
  DeviceVector<float> weights_device(host_weights);

  WeightedType weighted_factor(weights_device.data(), this->num_vectors_, obs_device.data(),
                               this->num_vectors_);
  weighted_factor.SetNumActiveFactors(weighted_factor.Capacity());

  Problem problem;
  problem.AddFactorBatch(&weighted_factor, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = this->minimizer_options_;
  LevenbergMarquardtMinimizer minimizer(lm_options);
  auto summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  ASSERT_TRUE(std::isfinite(summary.initial_cost));
  ASSERT_TRUE(std::isfinite(summary.final_cost));
  ASSERT_LE(summary.final_cost, summary.initial_cost);

  this->CheckConvergence(vector_states);
}

/**
 * @brief Values: the weighted residuals and Jacobians are the wrapped factor's
 * multiplied by the weight of each item's factor (uniform and per-factor).
 */
TEST(WeightedFactorBatchValues, ScalesResidualsAndJacobians) {
  constexpr int kDim = 3;
  constexpr size_t kN = 37;
  using FactorType = PriorVectorFactorBatch<kDim>;
  std::vector<Vector<kDim>> obs(kN), states(kN);
  std::vector<float> weights(kN);
  for (size_t i = 0; i < kN; i++) {
    for (int k = 0; k < kDim; k++) {
      obs[i][k] = 0.1f * static_cast<float>(i) - static_cast<float>(k);
      states[i][k] = 0.3f * static_cast<float>(k) - 0.05f * static_cast<float>(i);
    }
    weights[i] = 0.5f + 0.25f * static_cast<float>(i % 7);
  }
  DeviceVector<Vector<kDim>> obs_device(obs), states_device(states);
  DeviceVector<float> weights_device(weights);
  std::vector<const float *> pointers(kN);
  for (size_t i = 0; i < kN; i++) {
    pointers[i] = reinterpret_cast<const float *>(states_device.data() + i);
  }
  DeviceVector<const float *> pointers_device(pointers);

  // Residuals and Jacobians (kDim x kDim per item) of `factor`.
  auto evaluate = [&](FactorBatch &factor) {
    factor.SetNumActiveFactors(kN);
    DeviceVector<float> r(kN * kDim), j(kN * kDim * kDim);
    CudaStream stream;
    EXPECT_TRUE(factor.Evaluate(r.data(), j.data(), pointers_device.data(), stream.GetStream()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    std::vector<float> hr(r.size()), hj(j.size());
    r.CopyToHost(hr.data(), hr.size());
    j.CopyToHost(hj.data(), hj.size());
    return std::make_pair(hr, hj);
  };
  FactorType plain(obs_device.data(), kN);
  WeightedFactorBatch<FactorType> uniform(2.5f, obs_device.data(), kN);
  WeightedFactorBatch<FactorType> per_factor(weights_device.data(), kN, obs_device.data(), kN);
  const auto ref = evaluate(plain);
  const auto u = evaluate(uniform);
  const auto p = evaluate(per_factor);
  for (size_t i = 0; i < kN; i++) {
    for (int k = 0; k < kDim; k++) {
      const size_t at = i * kDim + k;
      EXPECT_FLOAT_EQ(u.first[at], 2.5f * ref.first[at]) << "uniform residual " << at;
      EXPECT_FLOAT_EQ(p.first[at], weights[i] * ref.first[at]) << "per-factor residual " << at;
    }
    for (int k = 0; k < kDim * kDim; k++) {
      const size_t at = i * kDim * kDim + k;
      EXPECT_FLOAT_EQ(u.second[at], 2.5f * ref.second[at]) << "uniform Jacobian " << at;
      EXPECT_FLOAT_EQ(p.second[at], weights[i] * ref.second[at]) << "per-factor Jacobian " << at;
    }
  }
}

/**
 * @brief Constructor validation: num_weights must match inner NumActiveFactors().
 */
TEST(WeightedFactorBatchValidation, NumWeightsMismatchThrows) {
  using FactorType = PriorVectorFactorBatch<2>;
  using WeightedType = WeightedFactorBatch<FactorType>;

  std::vector<Vector<2>> obs_host = {{1.f, 2.f}, {3.f, 4.f}};
  DeviceVector<Vector<2>> obs_device(obs_host);
  std::vector<float> host_weights = {1.f, 2.f, 3.f};
  DeviceVector<float> weights_device(host_weights);

  ASSERT_THROW(WeightedType(weights_device.data(), /*num_weights=*/5, obs_device.data(), size_t{2}),
               std::invalid_argument);
}

/**
 * @brief Constructor validation: null per-factor weights should throw.
 */
TEST(WeightedFactorBatchValidation, NullPerFactorWeightsThrows) {
  using FactorType = PriorVectorFactorBatch<2>;
  using WeightedType = WeightedFactorBatch<FactorType>;

  std::vector<Vector<2>> obs_host = {{1.f, 2.f}, {3.f, 4.f}};
  DeviceVector<Vector<2>> obs_device(obs_host);

  ASSERT_THROW(WeightedType(static_cast<const float *>(nullptr), 2, obs_device.data(), size_t{2}),
               std::invalid_argument);
}

}  // namespace cunls
