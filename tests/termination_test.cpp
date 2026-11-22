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
// Minimizer::SubproblemTerminationReasons(): how a solve stopped is reported,
// per subproblem with a partition.

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/constraint_factor_batch.h"
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

// MinimizerOptions::real_time: exactly max_num_iterations iterations and no
// host synchronization, so nothing is determined; after a first call that sets
// the structure up, a call can be captured in a CUDA graph (a read-back or a
// synchronization would fail the capture).
TEST(TerminationTest, RealTimeSolvesAreGraphCapturable) {
  for (const bool levenberg_marquardt : {false, true}) {
    SCOPED_TRACE(levenberg_marquardt ? "Levenberg-Marquardt" : "Gauss-Newton");
    CudaStream stream;
    const std::vector<float> start = {5.f, -3.f};
    PriorProblem p(start, false);
    MinimizerOptions options = Options();
    options.max_num_iterations = 2;
    options.real_time = true;
    options.reuse_structure = true;
    GaussNewtonMinimizer gn(options);
    LevenbergMarquardtMinimizerOptions lm;
    lm.base_options = options;
    LevenbergMarquardtMinimizer lm_minimizer(lm);
    Minimizer &minimizer = levenberg_marquardt ? static_cast<Minimizer &>(lm_minimizer) : gn;
    const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), p.problem);
    EXPECT_EQ(summary.termination_reason, TerminationReason::NotDetermined);
    EXPECT_TRUE(summary.num_problems_by_reason.empty());
    EXPECT_EQ(summary.num_iterations, 2u);
    EXPECT_TRUE(std::isnan(summary.final_cost));
    EXPECT_FALSE(summary.Converged());

    p.states.CopyFromHost(start.data(), start.size());
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(stream.GetStream(), cudaStreamCaptureModeThreadLocal),
              cudaSuccess);
    minimizer.Minimize(stream.GetStream(), p.problem);
    ASSERT_EQ(cudaStreamEndCapture(stream.GetStream(), &graph), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(exec, stream.GetStream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream.GetStream()), cudaSuccess);
    std::vector<float> x(2);
    p.states.CopyToHost(x.data(), x.size());
    EXPECT_NEAR(x[0], 1.f, 1e-2f);
    EXPECT_NEAR(x[1], 2.f, 1e-2f);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
}

/**
 * Begins a capture on `stream`, expects `call` to throw std::invalid_argument
 * naming `fix`, and expects the capture to end normally (nothing was enqueued).
 */
void ExpectThrowInCapture(CudaStream &stream, const std::function<void()> &call, const char *fix) {
  ASSERT_EQ(cudaStreamBeginCapture(stream.GetStream(), cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  try {
    call();
    ADD_FAILURE() << "expected std::invalid_argument naming: " << fix;
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find(fix), std::string::npos) << e.what();
  }
  cudaGraph_t graph = nullptr;
  EXPECT_EQ(cudaStreamEndCapture(stream.GetStream(), &graph), cudaSuccess);
  if (graph != nullptr) cudaGraphDestroy(graph);
}

// A captured call that would synchronize throws before it enqueues anything,
// so the capture ends normally and the stream keeps working: a first real-time
// call, a real-time call without reuse_structure and a converging call. The fix
// the message names (one call before the capture) makes the next call
// capturable.
TEST(TerminationTest, CapturedCallsThatWouldSynchronizeThrow) {
  CudaStream stream;
  const std::vector<float> start = {5.f, -3.f};
  PriorProblem p(start, false);
  MinimizerOptions options = Options();
  options.max_num_iterations = 2;
  options.real_time = true;
  options.reuse_structure = true;
  GaussNewtonMinimizer real_time(options);
  options.reuse_structure = false;
  GaussNewtonMinimizer without_reuse(options);
  options.real_time = false;
  options.reuse_structure = true;
  GaussNewtonMinimizer converging(options);

  ExpectThrowInCapture(
      stream, [&] { real_time.Minimize(stream.GetStream(), p.problem); },
      "run one Minimize() call on the problem before the capture");
  without_reuse.Minimize(stream.GetStream(), p.problem);
  ExpectThrowInCapture(
      stream, [&] { without_reuse.Minimize(stream.GetStream(), p.problem); },
      "set MinimizerOptions::reuse_structure");
  converging.Minimize(stream.GetStream(), p.problem);
  ExpectThrowInCapture(
      stream, [&] { converging.Minimize(stream.GetStream(), p.problem); },
      "set MinimizerOptions::real_time");

  // The fix: one call, then the captured call runs.
  real_time.Minimize(stream.GetStream(), p.problem);
  p.states.CopyFromHost(start.data(), start.size());
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(cudaStreamBeginCapture(stream.GetStream(), cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  real_time.Minimize(stream.GetStream(), p.problem);
  ASSERT_EQ(cudaStreamEndCapture(stream.GetStream(), &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, stream.GetStream()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.GetStream()), cudaSuccess);
  std::vector<float> x(2);
  p.states.CopyToHost(x.data(), x.size());
  EXPECT_NEAR(x[0], 1.f, 1e-2f);
  EXPECT_NEAR(x[1], 2.f, 1e-2f);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
}

// The augmented Lagrangian refuses a first captured call too, and any captured
// call with constraint batches (its status is read back at the end).
TEST(TerminationTest, CapturedAugmentedLagrangianCallsThatWouldSynchronizeThrow) {
  CudaStream stream;
  PriorProblem p({5.f, -3.f}, false);
  AugmentedLagrangianMinimizerOptions options;
  options.base_options = Options();
  options.base_options.real_time = true;
  options.base_options.reuse_structure = true;
  options.max_outer_iterations = 1;
  AugmentedLagrangianMinimizer al(options);
  ExpectThrowInCapture(
      stream, [&] { al.Minimize(stream.GetStream(), p.problem); },
      "run one Minimize() call on the problem before the capture");

  ConstraintFactorBatch constraint(&p.priors, ConstraintKind::kEquality);
  Problem constrained;
  constrained.AddStateBatch(&p.state_batch);
  constrained.AddFactorBatch(&constraint, std::vector<float *>{p.states.data()});
  al.Minimize(stream.GetStream(), constrained);
  ExpectThrowInCapture(
      stream, [&] { al.Minimize(stream.GetStream(), constrained); }, "1 constraint batches");
}

TEST(TerminationTest, RealTimeAugmentedLagrangianWithoutConstraintsIsNotDetermined) {
  CudaStream stream;
  PriorProblem p({50.f, -30.f}, false);
  AugmentedLagrangianMinimizerOptions options;
  options.base_options = Options();
  options.initial_lambda = 1e4f;  // one heavily damped step: far from the minimum
  options.base_options.real_time = true;
  options.base_options.max_num_iterations = 1;
  options.max_outer_iterations = 1;
  AugmentedLagrangianMinimizer al(options);
  const AugmentedLagrangianMinimizerSummary summary = al.Minimize(stream.GetStream(), p.problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kNotDetermined) << summary.message;
  EXPECT_FALSE(summary.Converged());
  EXPECT_EQ(summary.num_converged, 0u);

  // A converging call (the default budget) establishes the outcome.
  options.base_options = Options();
  options.max_outer_iterations = AugmentedLagrangianMinimizerOptions().max_outer_iterations;
  al.SetOptions(options);
  const AugmentedLagrangianMinimizerSummary full = al.Minimize(stream.GetStream(), p.problem);
  EXPECT_TRUE(full.Converged()) << full.message;
  EXPECT_EQ(full.num_converged, 1u);
}

TEST(TerminationTest, AllConstantStatesCountEverySubproblem) {
  CudaStream stream;
  PriorProblem p({5.f, -3.f, 1.f, 1.f, 0.f, 0.f}, true, /*all_constant=*/true);
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = Options();
  LevenbergMarquardtMinimizer minimizer(lm);
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), p.problem);
  EXPECT_EQ(Count(summary, TerminationReason::NoFreeStates), 3u);
  AugmentedLagrangianMinimizerOptions al_options;
  al_options.base_options = Options();
  AugmentedLagrangianMinimizer al(al_options);
  const AugmentedLagrangianMinimizerSummary wrapped = al.Minimize(stream.GetStream(), p.problem);
  EXPECT_TRUE(wrapped.Converged()) << wrapped.message;
  EXPECT_EQ(wrapped.num_converged, 3u);
}

// A solver reused on an all-constant problem with more subproblems reports one
// NoFreeStates reason per subproblem of that problem (not the earlier call's
// table); in real time that call reads nothing back and can be captured.
TEST(TerminationTest, AllConstantProblemAfterAnotherOne) {
  CudaStream stream;
  AugmentedLagrangianMinimizerOptions options;
  options.base_options = Options();
  AugmentedLagrangianMinimizer al(options);
  PriorProblem first({5.f, -3.f}, false);
  ASSERT_TRUE(al.Minimize(stream.GetStream(), first.problem).Converged());
  PriorProblem constant({5.f, -3.f, 1.f, 1.f, 0.f, 0.f}, true, /*all_constant=*/true);
  const auto summary = al.Minimize(stream.GetStream(), constant.problem);
  EXPECT_TRUE(summary.Converged()) << summary.message;
  ASSERT_EQ(al.NumSubproblems(), 3u);
  std::vector<int> reasons(3);
  THROW_ON_CUDA_ERROR(cudaMemcpy(reasons.data(), al.SubproblemTerminationReasons(), 3 * sizeof(int),
                                 cudaMemcpyDeviceToHost));
  for (int r : reasons) EXPECT_EQ(r, static_cast<int>(TerminationReason::NoFreeStates));

  MinimizerOptions real_time = Options();
  real_time.real_time = true;
  real_time.reuse_structure = true;
  GaussNewtonMinimizer gn(real_time);
  gn.Minimize(stream.GetStream(), constant.problem);
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamBeginCapture(stream.GetStream(), cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  const MinimizerSummary captured = gn.Minimize(stream.GetStream(), constant.problem);
  ASSERT_EQ(cudaStreamEndCapture(stream.GetStream(), &graph), cudaSuccess);
  EXPECT_EQ(captured.termination_reason, TerminationReason::NoFreeStates);
  EXPECT_TRUE(std::isnan(captured.final_cost));
  cudaGraphDestroy(graph);
}

// A problem lives on the device of its states. Minimize() runs there and
// rejects a call made from another device or with a stream of another device
// (needs two GPUs).
TEST(TerminationTest, MinimizeRunsOnTheProblemsDevice) {
  int current = -1, count = 0;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  {
    PriorProblem p({5.f, -3.f}, false);
    EXPECT_EQ(p.problem.Device(), current);
  }
  if (count < 2) GTEST_SKIP() << "needs two GPUs";
  const int other = current == 0 ? 1 : 0;
  ASSERT_EQ(cudaSetDevice(other), cudaSuccess);
  {
    PriorProblem p({5.f, -3.f}, false);  // states on `other`
    ASSERT_EQ(cudaSetDevice(current), cudaSuccess);
    CudaStream stream;  // on `current`
    LevenbergMarquardtMinimizer minimizer;
    EXPECT_THROW(minimizer.Minimize(stream.GetStream(), p.problem), std::invalid_argument);
    ASSERT_EQ(cudaSetDevice(other), cudaSuccess);
    EXPECT_THROW(minimizer.Minimize(stream.GetStream(), p.problem), std::invalid_argument);
    CudaStream other_stream;
    EXPECT_TRUE(minimizer.Minimize(other_stream.GetStream(), p.problem).Converged());
  }
  ASSERT_EQ(cudaSetDevice(current), cudaSuccess);
}

TEST(TerminationTest, EveryReasonHasAName) {
  for (int r = 0; r < kNumTerminationReasons; ++r) {
    EXPECT_STRNE(ToString(static_cast<TerminationReason>(r)), "Unknown");
  }
}

}  // namespace
}  // namespace cunls
