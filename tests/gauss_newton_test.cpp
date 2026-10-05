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

/** @file gauss_newton_test.cpp
 *  @brief Tests for Gauss-Newton and Levenberg-Marquardt minimizer convergence.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/sized_factor_batch.h"
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

    test_utils::ConfigureTestSolver(TestParam::solver_id, minimizer_options_);
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
  void CheckConvergence(const StatesType &states, const std::vector<int> &const_state_ids = {}) {
    size_t num_blocks = states.NumActiveStates();
    auto ptr = reinterpret_cast<const VectorType *>(states.StateDevicePtr(0));

    std::vector<VectorType> host_states(num_blocks);
    THROW_ON_CUDA_ERROR(cudaMemcpy(host_states.data(), ptr, num_blocks * sizeof(VectorType),
                                   cudaMemcpyDeviceToHost));

    ASSERT_EQ(host_states.size(), observations_.size());
    for (size_t i = 0; i < num_blocks; i++) {
      const auto &obs = observations_[i];
      const auto &state_vals = host_states[i];
      auto const_state_it = std::find(const_state_ids.begin(), const_state_ids.end(), i);
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

  const size_t num_vectors_ = 10000;  ///< Number of states in test.

  std::vector<VectorType> observations_;  ///< Target values for optimization.
  std::vector<VectorType> state_values_;  ///< Initial state values.

  MinimizerOptions minimizer_options_{.disable_safety_checks = false};

  profiler::Domain profiler_domain_{"GaussNewtonMinimizerTest"};  ///< Profiling domain.
};

/**
 * @brief Helper struct for parameterized test dimensions.
 *
 * @tparam VectorSize Vector dimension (1, 2, 3, or 4).
 * @tparam SolverId Solver id, see test_utils::ConfigureTestSolver.
 */
template <int VectorSize, int SolverId>
struct TestParam {
  static constexpr int vector_size = VectorSize;
  static constexpr int solver_id = SolverId;
};

constexpr int kPCG = test_utils::kBlockSparsePCGSolverId;

/** @brief Test types: 1D, 2D, 3D, and 4D vectors for each available solver. */
typedef ::testing::Types<
#ifdef CUNLS_ENABLE_CUDSS
    TestParam<1, 0>, TestParam<2, 0>, TestParam<3, 0>, TestParam<4, 0>, TestParam<1, 1>,
    TestParam<2, 1>, TestParam<3, 1>, TestParam<4, 1>,
#endif
    TestParam<1, kPCG>, TestParam<2, kPCG>, TestParam<3, kPCG>, TestParam<4, kPCG>>
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
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto &factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  GaussNewtonMinimizer minimizer(this->minimizer_options_);
  auto range = this->profiler_domain_.CreateDomainRange("GN Minimize");
  // User-owned pool path: pre-attach before Minimize (minimizer skips
  // auto-attach).
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
  auto test_range = this->profiler_domain_.CreateDomainRange("GNWithConstantStates");
  std::vector<int> const_state_ids = {0, 9, 99, 999};
  typename TestFixture::StateData state_data(this->state_values_, const_state_ids);
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto &factor_batch = factor_data.get();

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
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto &factor_batch = factor_data.get();

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
  auto test_range = this->profiler_domain_.CreateDomainRange("LMWithConstantStates");
  std::vector<int> const_state_ids = {0, 9, 99, 999};
  typename TestFixture::StateData state_data(this->state_values_, const_state_ids);
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto &factor_batch = factor_data.get();

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

/**
 * @brief Levenberg-Marquardt with Hessian-diagonal column scaling.
 */
TYPED_TEST(GaussNewtonMinimizerTest, LMColumnScalingHessianDiagonal) {
  auto test_range = this->profiler_domain_.CreateDomainRange("LMColumnScalingHessianDiagonal");
  typename TestFixture::StateData state_data(this->state_values_);
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto &factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = this->minimizer_options_;
  lm_options.base_options.column_scaling = ColumnScaling::HessianDiagonal;
  LevenbergMarquardtMinimizer minimizer(lm_options);
  minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  this->CheckConvergence(vector_states);
}

/**
 * @brief Gauss-Newton with column scaling (shared MinimizerOptions path).
 */
TYPED_TEST(GaussNewtonMinimizerTest, GNColumnScalingHessianDiagonal) {
  auto test_range = this->profiler_domain_.CreateDomainRange("GNColumnScalingHessianDiagonal");
  typename TestFixture::StateData state_data(this->state_values_);
  auto &vector_states = state_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);
  typename TestFixture::FactorData factor_data(this->observations_);
  auto &factor_batch = factor_data.get();

  Problem problem;
  problem.AddFactorBatch(&factor_batch, device_pointers);
  problem.AddStateBatch(&vector_states);

  CudaStream stream;
  MinimizerOptions opts = this->minimizer_options_;
  opts.column_scaling = ColumnScaling::HessianDiagonal;
  GaussNewtonMinimizer minimizer(opts);
  minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  this->CheckConvergence(vector_states);
}

/**
 * @brief Two Minimize calls on the same problem with the same minimizer
 * instance should yield identical summaries after resetting the state
 * (regression for persistent lhs/rhs and MinimizerState buffers).
 */
TEST(MinimizeBufferReuse, GaussNewtonTwiceIdenticalSummaries) {
  constexpr size_t n = 32;
  std::vector<Vector<1>> observations(n);
  std::vector<Vector<1>> state_values(n);
  for (size_t i = 0; i < n; ++i) {
    observations[i][0] = static_cast<float>(i) - 1.f;
    state_values[i][0] = static_cast<float>(i);
  }
  test_utils::VectorStateData<1> state_data(state_values);
  auto &vector_states = state_data.get();
  test_utils::PriorFactorData<1> factor_data(observations);
  auto &factor_batch = factor_data.get();
  auto device_pointers = test_utils::CollectStatePointers(vector_states);

  Problem problem;
  problem.AddStateBatch(&vector_states);
  problem.AddFactorBatch(&factor_batch, device_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerOptions opts;
  test_utils::ConfigureTestSolver(test_utils::kDefaultTestSolverId, opts);
  opts.disable_safety_checks = false;

  CudaStream stream;
  GaussNewtonMinimizer minimizer(opts);

  float *state_base = vector_states.StateDevicePtr(0);
  const size_t num_floats = n * 1;
  std::vector<float> initial_host(num_floats);
  THROW_ON_CUDA_ERROR(cudaMemcpy(initial_host.data(), state_base, num_floats * sizeof(float),
                                 cudaMemcpyDeviceToHost));

  MinimizerSummary s1 = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  THROW_ON_CUDA_ERROR(cudaMemcpy(state_base, initial_host.data(), num_floats * sizeof(float),
                                 cudaMemcpyHostToDevice));

  MinimizerSummary s2 = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_EQ(s1.num_iterations, s2.num_iterations);
  EXPECT_NEAR(s1.final_cost, s2.final_cost, 1e-4f);
  EXPECT_NEAR(s1.initial_cost, s2.initial_cost, 1e-4f);
}

/**
 * @brief Reusing a minimizer after the number of state batches decreases must
 * not retain stale state vectors from the previous problem.
 *
 * Regression test for a two-batch to one-batch topology transition. Before the
 * fix, MinimizerState::CreateStates only grew its outer vector. UpdateStates
 * consequently passed two state pointers to StateBatchOps configured for one
 * batch, causing an assertion failure in debug builds and an out-of-bounds
 * access in release builds.
 */
TEST(MinimizeBufferReuse, GaussNewtonStateBatchCountDecreases) {
  constexpr size_t n = 32;
  const auto observations = test_utils::MakeZeroVectors<1>(n);
  const auto initial_states = test_utils::MakeConstantVectors<1>(n, 1.0f);

  MinimizerOptions opts;
  test_utils::ConfigureTestSolver(test_utils::kDefaultTestSolverId, opts);
  opts.disable_safety_checks = false;

  CudaStream stream;
  GaussNewtonMinimizer minimizer(opts);

  {
    test_utils::VectorStateData<1> first_state_data(initial_states);
    test_utils::VectorStateData<1> second_state_data(initial_states);
    test_utils::PriorFactorData<1> first_factor_data(observations);
    test_utils::PriorFactorData<1> second_factor_data(observations);

    Problem two_batch_problem;
    two_batch_problem.AddStateBatch(first_state_data.ptr());
    two_batch_problem.AddStateBatch(second_state_data.ptr());
    two_batch_problem.AddFactorBatch(&first_factor_data.get(),
                                     test_utils::CollectStatePointers(first_state_data.get()));
    two_batch_problem.AddFactorBatch(&second_factor_data.get(),
                                     test_utils::CollectStatePointers(second_state_data.get()));
    ASSERT_TRUE(two_batch_problem.CheckConsistency());

    minimizer.Minimize(stream.GetStream(), two_batch_problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    test_utils::VectorStateData<1> state_data(initial_states);
    test_utils::PriorFactorData<1> factor_data(observations);

    Problem one_batch_problem;
    one_batch_problem.AddStateBatch(state_data.ptr());
    one_batch_problem.AddFactorBatch(&factor_data.get(),
                                     test_utils::CollectStatePointers(state_data.get()));
    ASSERT_TRUE(one_batch_problem.CheckConsistency());

    EXPECT_NO_THROW(minimizer.Minimize(stream.GetStream(), one_batch_problem));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }
}

}  // namespace cunls

namespace cunls {
namespace {

/**
 * r = x - 5 on a 1-D state, but NaN beyond x = 2: the undamped step from x = 0
 * lands where the cost is not finite.
 */
class NanBeyondTwoFactor : public SizedFactorBatch<1, 1> {
 public:
  NanBeyondTwoFactor() : SizedFactorBatch(1) {}
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *, size_t num_factor_ids) const override {
    const size_t n = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
    for (size_t i = 0; i < n; ++i) {
      const float *state = nullptr;
      float x = 0.f;
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&state, state_pointers + i, sizeof(float *),
                                          cudaMemcpyDeviceToHost, stream));
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
      THROW_ON_CUDA_ERROR(
          cudaMemcpyAsync(&x, state, sizeof(float), cudaMemcpyDeviceToHost, stream));
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
      const float r = x > 2.f ? std::nanf("") : x - 5.f, j = 1.f;
      THROW_ON_CUDA_ERROR(
          cudaMemcpyAsync(residuals + i, &r, sizeof(float), cudaMemcpyHostToDevice, stream));
      if (jacobians != nullptr) {
        THROW_ON_CUDA_ERROR(
            cudaMemcpyAsync(jacobians + i, &j, sizeof(float), cudaMemcpyHostToDevice, stream));
      }
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    }
    return true;
  }
};

}  // namespace

// A trial step to a non-finite cost is rejected (and the damping raised), by
// both minimizers, with and without a subproblem partition; it was accepted
// by Levenberg-Marquardt, whose test rho < threshold is false for NaN.
// A factor whose Evaluate reports failure stops the solve with an exception
// instead of optimizing on unwritten residuals.
class FailingFactorBatch : public SizedFactorBatch<1, 1> {
 public:
  explicit FailingFactorBatch(size_t capacity) : SizedFactorBatch(capacity) {}
  bool Evaluate(float *, float *, float const *const *, cudaStream_t, const int *,
                size_t) const override {
    return false;
  }
};

TEST(FailingFactor, MinimizersThrow) {
  CudaStream stream;
  dvector<float> x(std::vector<float>{1.f});
  VectorStateBatch<1> states(x.data(), 1);
  states.SetNumActiveStates(1);
  FailingFactorBatch failing(1);
  failing.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&failing, {states.StateDevicePtr(0)});
  MinimizerOptions options;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
  GaussNewtonMinimizer gn(options);
  EXPECT_THROW(gn.Minimize(stream.GetStream(), problem), std::runtime_error);
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  LevenbergMarquardtMinimizer lm(lm_options);
  EXPECT_THROW(lm.Minimize(stream.GetStream(), problem), std::runtime_error);
}

TEST(NonFiniteTrialStep, IsRejected) {
  for (int kind = 0; kind < 4; ++kind) {
    CudaStream stream;
    dvector<float> x(std::vector<float>{0.f});
    VectorStateBatch<1> states(x.data(), 1);
    states.SetNumActiveStates(1);
    NanBeyondTwoFactor factor;
    factor.SetNumActiveFactors(1);
    Problem problem;
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&factor, std::vector<float *>{states.StateDevicePtr(0)});
    dvector<int> ids(std::vector<int>{0});
    if (kind >= 2) problem.SetProblemPartition(2, {ids.data()});  // 2nd subproblem empty
    MinimizerOptions options;
    options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
    MinimizerSummary summary;
    if (kind % 2 == 0) {
      LevenbergMarquardtMinimizerOptions lm;
      lm.base_options = options;
      summary = LevenbergMarquardtMinimizer(lm).Minimize(stream.GetStream(), problem);
    } else {
      summary = GaussNewtonMinimizer(options).Minimize(stream.GetStream(), problem);
    }
    std::vector<float> host(1);
    x.CopyToHost(host.data(), 1);
    EXPECT_TRUE(std::isfinite(summary.final_cost)) << "case " << kind;
    EXPECT_LE(summary.final_cost, summary.initial_cost) << "case " << kind;
    EXPECT_LE(host[0], 2.f) << "case " << kind;
  }
}

// On a linear problem the LM model is exact, so the gain ratio rho is 1 and
// every step is "very successful": lambda shrinks each iteration and LM turns
// into Gauss-Newton. Priors give H = I, so from lambda_0 = 10 the step is
// -g / (1 + lambda) and twelve iterations reach the optimum to ~1e-20 of the
// initial cost. A model decrease overestimated 2x (rho = 0.5 < 0.75) keeps
// lambda at 10 and leaves ~10% of the cost. Single problem and subproblems.
TEST(LevenbergMarquardt, LinearProblemShrinksDamping) {
  for (bool partitioned : {false, true}) {
    CudaStream stream;
    constexpr int kN = 8;
    std::vector<Vector<3>> targets(kN);
    for (int i = 0; i < kN; ++i) targets[i] = {float(i), -2.f * i, 0.5f};
    dvector<Vector<3>> x(std::vector<Vector<3>>(kN, Vector<3>{0.f, 0.f, 0.f}));
    dvector<Vector<3>> t(targets);
    VectorStateBatch<3> states(reinterpret_cast<const float *>(x.data()), kN);
    states.SetNumActiveStates(kN);
    PriorVectorFactorBatch<3> priors(t.data(), kN);
    priors.SetNumActiveFactors(kN);
    std::vector<float *> ptrs;
    for (int i = 0; i < kN; ++i) ptrs.push_back(states.StateDevicePtr(i));
    Problem problem;
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&priors, ptrs);
    std::vector<int> ids(kN);
    for (int i = 0; i < kN; ++i) ids[i] = i % 2;
    dvector<int> d_ids(ids);
    if (partitioned) problem.SetProblemPartition(2, {d_ids.data()});
    LevenbergMarquardtMinimizerOptions options;
    options.base_options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
    options.base_options.max_num_iterations = 12;
    options.base_options.state_tolerance = 0.f;
    options.base_options.cost_tolerance = 0.f;
    options.relative_reduction_tolerance = 0.f;
    options.initial_lambda = 10.f;
    const MinimizerSummary summary =
        LevenbergMarquardtMinimizer(options).Minimize(stream.GetStream(), problem);
    EXPECT_LT(summary.final_cost, 1e-6f * summary.initial_cost) << "partitioned " << partitioned;
  }
}

// An initial damping of 0 is valid (Gauss-Newton steps), and a rejected step
// still escalates from lambda_min: from x = 0 the undamped step lands where
// the cost is NaN, so lambda must grow (to >= 1.5) before a step is taken.
// lambda_min itself must be positive.
TEST(LevenbergMarquardt, ZeroInitialLambdaEscalatesFromLambdaMin) {
  for (bool partitioned : {false, true}) {
    CudaStream stream;
    dvector<float> x(std::vector<float>{0.f});
    VectorStateBatch<1> states(x.data(), 1);
    states.SetNumActiveStates(1);
    NanBeyondTwoFactor factor;
    factor.SetNumActiveFactors(1);
    Problem problem;
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&factor, std::vector<float *>{states.StateDevicePtr(0)});
    dvector<int> ids(std::vector<int>{0});
    if (partitioned) problem.SetProblemPartition(2, {ids.data()});
    LevenbergMarquardtMinimizerOptions options;
    options.base_options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
    options.initial_lambda = 0.f;
    const MinimizerSummary summary =
        LevenbergMarquardtMinimizer(options).Minimize(stream.GetStream(), problem);
    EXPECT_LT(summary.final_cost, summary.initial_cost) << "partitioned " << partitioned;
  }
  LevenbergMarquardtMinimizerOptions bad;
  bad.lambda_min = 0.f;
  EXPECT_THROW(LevenbergMarquardtMinimizer{bad}, std::invalid_argument);
  LevenbergMarquardtMinimizerOptions inverted;
  inverted.lambda_min = 1.f;
  inverted.lambda_max = 0.5f;
  EXPECT_THROW(LevenbergMarquardtMinimizer{inverted}, std::invalid_argument);
  inverted.lambda_max = 1.f;  // equal bounds: a fixed damping
  EXPECT_NO_THROW(LevenbergMarquardtMinimizer{inverted});
}

}  // namespace cunls
