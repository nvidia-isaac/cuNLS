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

/** @file gauss_newton_test.cpp
 *  @brief Tests for Gauss-Newton and Levenberg-Marquardt minimizer convergence.
 */

#include <gtest/gtest.h>

#include <memory>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

/**
 * @brief Test fixture for Gauss-Newton and Levenberg-Marquardt minimizers.
 *
 * Provides a parameterized test framework that tests minimizers with different
 * vector dimensions (1D, 2D, 3D, 4D). The test creates a simple least squares
 * problem where states are initialized away from their optimal values
 * (observations), and verifies that the minimizer converges to the correct
 * solution.
 *
 * @tparam TestParam Template parameter specifying the test parameters.
 */
template <class TestParam>
class GaussNewtonMinimizerTest : public ::testing::Test {
 public:
  static constexpr int kDim = TestParam::vector_size;
  using StatesType = VectorStateBatch<kDim>;
  using VectorType = Vector<kDim>;
  using StateData = test_utils::VectorStateData<kDim>;
  using FactorData = test_utils::PriorFactorData<kDim>;

  /**
   * @brief Sets up test data before each test.
   *
   * Initializes observations and state values. States are initialized
   * one unit away from their optimal values (observations), creating a simple
   * optimization problem where the minimizer should converge states to
   * match observations.
   */
  void SetUp() override {
    observations_.resize(num_vectors_);
    state_values_.resize(num_vectors_);
    for (size_t i = 0; i < num_vectors_; i++) {
      float x = static_cast<float>(i);
      // Observations are the target values
      observations_[i].fill(x - 1.f);
      // States start one unit away from observations
      state_values_[i].fill(x);
    }

    minimizer_options_.sparse_linear_solver_type =
        SparseLinearSolverType::cuDSS;

    cuDSSLinearSolverOptions cudss_solver_options = {
        .mode = static_cast<cuDSSLinearSolverMode>(TestParam::solver_id),
        .nthreads = 1,
        .threading_lib_path = "",
    };
    minimizer_options_.sparse_linear_solver_config = {.cudss_solver_options =
                                                          cudss_solver_options};
  }

  /**
   * @brief Verifies that optimization converged to the correct solution.
   *
   * Checks that optimized states match their corresponding observations
   * (within tolerance). For constant states, verifies they were not
   * modified during optimization.
   *
   * @param states Optimized state batch to verify.
   */
  void CheckConvergence(const StatesType& states,
                        const std::vector<int>& const_state_ids = {}) {
    size_t num_blocks = states.NumStateBlocks();
    auto ptr = reinterpret_cast<const VectorType*>(
        states.StateBlockDevicePtr(0));

    std::vector<VectorType> host_states(num_blocks);
    THROW_ON_CUDA_ERROR(cudaMemcpy(host_states.data(), ptr,
                                    num_blocks * sizeof(VectorType),
                                    cudaMemcpyDeviceToHost));

    ASSERT_EQ(host_states.size(), observations_.size());
    for (size_t i = 0; i < num_blocks; i++) {
      const auto& obs = observations_[i];
      const auto& state_vals = host_states[i];
      auto const_state_it =
          std::find(const_state_ids.begin(), const_state_ids.end(), i);
      if (const_state_it != const_state_ids.end()) {
        // State is constant, verify it wasn't changed during optimization
        float x = static_cast<float>(i);
        for (size_t j = 0; j < TestParam::vector_size; j++) {
          ASSERT_NEAR(state_vals[j], x, 1e-3);
        }
      } else {
        for (size_t j = 0; j < TestParam::vector_size; j++) {
          ASSERT_NEAR(obs[j], state_vals[j], 1e-3);
        }
      }
    }
  }

  const size_t num_vectors_ = 10000;  ///< Number of state blocks in test.

  std::vector<VectorType> observations_;  ///< Target values for optimization.
  std::vector<VectorType> state_values_;  ///< Initial state values.

  MinimizerOptions minimizer_options_;

  profiler::Domain profiler_domain_{
      "GaussNewtonMinimizerTest"};  ///< Profiling domain.
};

/**
 * @brief Helper struct for parameterized test dimensions.
 *
 * @tparam Value Vector dimension (1, 2, 3, or 4).
 */
template <int VectorSize, int SolverId>
struct TestParam {
  static constexpr int vector_size = VectorSize;
  static constexpr int solver_id = SolverId;
};

/** @brief Test types: 1D, 2D, 3D, and 4D vectors. */
typedef ::testing::Types<TestParam<1, 0>, TestParam<2, 0>, TestParam<3, 0>,
                         TestParam<4, 0>, TestParam<1, 1>, TestParam<2, 1>,
                         TestParam<3, 1>, TestParam<4, 1>>
    TestParams;
TYPED_TEST_CASE(GaussNewtonMinimizerTest, TestParams);

/**
 * @brief Tests basic Gauss-Newton optimization.
 *
 * Creates a simple least squares problem and verifies that Gauss-Newton
 * converges to the correct solution.
 */
TYPED_TEST(GaussNewtonMinimizerTest, SimpleGN) {
  auto test_range = this->profiler_domain_.CreateDomainRange("SimpleGNTest");
  typename TestFixture::StateData state_data(this->state_values_);
  auto& vector_states = state_data.get();
  auto device_pointers =
      test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto& factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  GaussNewtonMinimizer minimizer(this->minimizer_options_);
  auto range = this->profiler_domain_.CreateDomainRange("GN Minimize");
  minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  this->CheckConvergence(vector_states);
}

/**
 * @brief Tests Gauss-Newton with constant (fixed) states.
 *
 * Verifies that the optimizer correctly handles constant states by
 * not modifying them during optimization while still optimizing other
 * states.
 */
TYPED_TEST(GaussNewtonMinimizerTest, GNWithConstantStates) {
  auto test_range =
      this->profiler_domain_.CreateDomainRange("GNWithConstantStates");
  std::vector<int> const_state_ids = {0, 9, 99, 999};
  typename TestFixture::StateData state_data(this->state_values_,
                                             const_state_ids);
  auto& vector_states = state_data.get();
  auto device_pointers =
      test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto& factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  GaussNewtonMinimizer minimizer(this->minimizer_options_);
  auto range = this->profiler_domain_.CreateDomainRange("GN Minimize");
  minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  this->CheckConvergence(vector_states, const_state_ids);
}

/**
 * @brief Tests basic Levenberg-Marquardt optimization.
 *
 * Creates a simple least squares problem and verifies that Levenberg-Marquardt
 * converges to the correct solution. LM should be more robust than GN for
 * ill-conditioned problems.
 */
TYPED_TEST(GaussNewtonMinimizerTest, SimpleLM) {
  auto test_range = this->profiler_domain_.CreateDomainRange("SimpleLM");
  typename TestFixture::StateData state_data(this->state_values_);
  auto& vector_states = state_data.get();
  auto device_pointers =
      test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto& factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = this->minimizer_options_;
  LevenbergMarquardtMinimizer minimizer(lm_options);
  auto range = this->profiler_domain_.CreateDomainRange("LM Minimize");
  minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  this->CheckConvergence(vector_states);
}

/**
 * @brief Tests Levenberg-Marquardt with constant (fixed) states.
 *
 * Verifies that LM correctly handles constant states by not modifying them
 * during optimization while still optimizing other states.
 */
TYPED_TEST(GaussNewtonMinimizerTest, LMWithConstantStates) {
  auto test_range =
      this->profiler_domain_.CreateDomainRange("LMWithConstantStates");
  std::vector<int> const_state_ids = {0, 9, 99, 999};
  typename TestFixture::StateData state_data(this->state_values_,
                                             const_state_ids);
  auto& vector_states = state_data.get();
  auto device_pointers =
      test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto& factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = this->minimizer_options_;
  LevenbergMarquardtMinimizer minimizer(lm_options);
  auto range = this->profiler_domain_.CreateDomainRange("LM Minimize");
  minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  this->CheckConvergence(vector_states, const_state_ids);
}

}  // namespace cunls
