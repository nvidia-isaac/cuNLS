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

// MinimizerSummary::termination_reason, num_problems_by_reason, message and
// Minimizer::SubproblemTerminationReasons(): every way a solve can stop is
// reported, per subproblem with a partition.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

/** n 2-vectors pulled to (1, 2) by one prior each; optionally one subproblem per vector. */
struct PriorProblem {
  DeviceVector<float> states, observations;
  DeviceVector<int> constant_ids, problem_ids;
  VectorStateBatch<2> state_batch;
  PriorVectorFactorBatch<2> priors;
  Problem problem;

  PriorProblem(const std::vector<float> &initial, bool partition, bool all_constant = false)
      : states(initial),
        observations(ObservationsFor(initial.size() / 2)),
        constant_ids(Iota(initial.size() / 2)),
        problem_ids(Iota(initial.size() / 2)),
        state_batch(states.data(), initial.size() / 2, constant_ids.data(), initial.size() / 2),
        priors(reinterpret_cast<const Vector<2> *>(observations.data()), initial.size() / 2) {
    const size_t n = initial.size() / 2;
    state_batch.SetNumActiveStates(n, all_constant ? n : 0);
    priors.SetNumActiveFactors(n);
    problem.AddStateBatch(&state_batch);
    std::vector<float *> pointers;
    for (size_t i = 0; i < n; ++i) pointers.push_back(states.data() + 2 * i);
    problem.AddFactorBatch(&priors, pointers);
    if (partition) problem.SetProblemPartition(n, {problem_ids.data()});
  }

  static std::vector<float> ObservationsFor(size_t n) {
    std::vector<float> out;
    for (size_t i = 0; i < n; ++i) out.insert(out.end(), {1.f, 2.f});
    return out;
  }
  static std::vector<int> Iota(size_t n) {
    std::vector<int> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<int>(i);
    return out;
  }
};

MinimizerOptions Options() {
  MinimizerOptions options;
  options.sparse_linear_solver_type = SparseLinearSolverType::BlockSparseCholesky;
  return options;
}

size_t Count(const MinimizerSummary &summary, TerminationReason reason) {
  return summary.num_problems_by_reason.at(static_cast<int>(reason));
}

TEST(TerminationTest, GaussNewtonConvergesOnALinearProblem) {
  CudaStream stream;
  PriorProblem p({5.f, -3.f, 0.f, 0.f}, false);
  GaussNewtonMinimizer minimizer(Options());
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), p.problem);
  EXPECT_TRUE(summary.Converged()) << summary.message;
  EXPECT_TRUE(IsConvergence(summary.termination_reason));
  EXPECT_EQ(summary.num_problems_by_reason.size(), static_cast<size_t>(kNumTerminationReasons));
  EXPECT_EQ(Count(summary, summary.termination_reason), 1u);
  EXPECT_EQ(summary.num_free_parameters, 4u);
  EXPECT_GE(summary.solve_time_ms, 0.0);
  EXPECT_NE(summary.message.find("Converged"), std::string::npos) << summary.message;
}

TEST(TerminationTest, IterationCapIsNotConvergence) {
  CudaStream stream;
  PriorProblem p({50.f, -30.f}, false);
  LevenbergMarquardtMinimizerOptions options;
  options.base_options = Options();
  options.base_options.max_num_iterations = 1;
  options.initial_lambda = 10.f;  // a damped first step: not at the minimum yet
  LevenbergMarquardtMinimizer minimizer(options);
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), p.problem);
  EXPECT_EQ(summary.termination_reason, TerminationReason::MaxIterations) << summary.message;
  EXPECT_FALSE(summary.Converged());
  EXPECT_NE(summary.message.find("max_num_iterations"), std::string::npos) << summary.message;
}

TEST(TerminationTest, NonFiniteInitialCostIsReported) {
  CudaStream stream;
  PriorProblem p({kNaN, 0.f}, false);
  GaussNewtonMinimizer minimizer(Options());
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), p.problem);
  EXPECT_EQ(summary.termination_reason, TerminationReason::NonFiniteCost) << summary.message;
  EXPECT_FALSE(summary.Converged());
  EXPECT_EQ(summary.num_iterations, 0u);
  EXPECT_NE(summary.message.find("not finite"), std::string::npos) << summary.message;
}

TEST(TerminationTest, AllConstantStatesHaveNothingToOptimize) {
  CudaStream stream;
  PriorProblem p({5.f, -3.f}, false, /*all_constant=*/true);
  GaussNewtonMinimizer minimizer(Options());
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), p.problem);
  EXPECT_EQ(summary.termination_reason, TerminationReason::NoFreeStates);
  EXPECT_TRUE(summary.Converged());
  EXPECT_EQ(summary.num_free_parameters, 0u);
}

TEST(TerminationTest, PartitionCountsEverySubproblemAndReportsTheWorst) {
  CudaStream stream;
  // Subproblem 1 starts at NaN; 0 and 2 converge; 3 starts at the minimum.
  PriorProblem p({5.f, -3.f, kNaN, 0.f, 7.f, 7.f, 1.f, 2.f}, true);
  GaussNewtonMinimizer minimizer(Options());
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), p.problem);
  EXPECT_EQ(summary.termination_reason, TerminationReason::NonFiniteCost) << summary.message;
  EXPECT_EQ(Count(summary, TerminationReason::NonFiniteCost), 1u);
  size_t converged = 0;
  for (int r = 0; r < kNumTerminationReasons; ++r) {
    if (IsConvergence(static_cast<TerminationReason>(r)))
      converged += summary.num_problems_by_reason[r];
  }
  EXPECT_EQ(converged, 3u) << summary.message;
  EXPECT_NE(summary.message.find("1 of 4 subproblems did not converge"), std::string::npos)
      << summary.message;

  ASSERT_EQ(minimizer.NumSubproblems(), 4u);
  std::vector<int> reasons(4);
  THROW_ON_CUDA_ERROR(cudaMemcpy(reasons.data(), minimizer.SubproblemTerminationReasons(),
                                 4 * sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(reasons[1], static_cast<int>(TerminationReason::NonFiniteCost))
      << reasons[0] << " " << reasons[2] << " " << summary.message;
  EXPECT_EQ(reasons[3], static_cast<int>(TerminationReason::CostTolerance));
  EXPECT_TRUE(IsConvergence(static_cast<TerminationReason>(reasons[0])));
  EXPECT_TRUE(IsConvergence(static_cast<TerminationReason>(reasons[2])));
}

TEST(TerminationTest, RealTimeSolvesDoNotDetermineTheReason) {
  CudaStream stream;
  PriorProblem p({5.f, -3.f}, false);
  GaussNewtonMinimizer minimizer(Options());
  internal::InnerSolve inner;
  inner.max_num_iterations = 2;
  inner.fixed_iterations = true;
  const MinimizerSummary summary =
      internal::InnerMinimize(minimizer, stream.GetStream(), p.problem, inner);
  EXPECT_EQ(summary.termination_reason, TerminationReason::NotDetermined);
  EXPECT_TRUE(summary.num_problems_by_reason.empty());
  EXPECT_FALSE(summary.Converged());
}

TEST(TerminationTest, RealTimeAugmentedLagrangianWithoutConstraintsIsNotDetermined) {
  CudaStream stream;
  PriorProblem p({50.f, -30.f}, false);
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = Options();
  lm.initial_lambda = 1e4f;  // one heavily damped step: far from the minimum
  LevenbergMarquardtMinimizer inner(lm);
  AugmentedLagrangianMinimizerOptions options;
  options.real_time = true;
  options.inner_iterations = 1;
  AugmentedLagrangianMinimizer al(inner, options);
  const AugmentedLagrangianMinimizerSummary summary = al.Minimize(stream.GetStream(), p.problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kNotDetermined) << summary.message;
  EXPECT_FALSE(summary.Converged());
  EXPECT_EQ(summary.num_converged, 0u);

  // The same call without real_time establishes the outcome.
  options.real_time = false;
  al.SetOptions(options);
  const AugmentedLagrangianMinimizerSummary full = al.Minimize(stream.GetStream(), p.problem);
  EXPECT_TRUE(full.Converged()) << full.message;
  EXPECT_EQ(full.num_converged, 1u);
}

TEST(TerminationTest, AllConstantStatesCountEverySubproblem) {
  CudaStream stream;
  PriorProblem p({5.f, -3.f, 1.f, 1.f, 0.f, 0.f}, true, /*all_constant=*/true);
  GaussNewtonMinimizer minimizer(Options());
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), p.problem);
  EXPECT_EQ(Count(summary, TerminationReason::NoFreeStates), 3u);
  AugmentedLagrangianMinimizer al(minimizer);
  const AugmentedLagrangianMinimizerSummary wrapped = al.Minimize(stream.GetStream(), p.problem);
  EXPECT_TRUE(wrapped.Converged()) << wrapped.message;
  EXPECT_EQ(wrapped.num_converged, 3u);
}

TEST(TerminationTest, EveryReasonHasAName) {
  for (int r = 0; r < kNumTerminationReasons; ++r) {
    EXPECT_STRNE(ToString(static_cast<TerminationReason>(r)), "Unknown");
  }
}

}  // namespace
}  // namespace cunls
