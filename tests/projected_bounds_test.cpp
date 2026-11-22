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

// Box bounds on vector states (VectorStateBatch::SetBounds), enforced by
// AugmentedLagrangianMinimizer (bounds held in its Levenberg-Marquardt inner
// solves): coupled
// box-constrained least squares against a float64 reference (single problem
// and a batch of subproblems), a constant state outside its bounds stays as
// it is, an infeasible initial guess is projected, and the minimizers alone
// reject a bounded problem.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

/** Augmented Lagrangian options whose inner solves run to convergence. */
AugmentedLagrangianMinimizerOptions Options(JacobianMode jacobians = JacobianMode::kAnalytic) {
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = test_utils::ConvergedSolveOptions();
  lm.base_options.jacobian_mode = jacobians;
  // One-sided differences: a perturbation clamped at an upper bound would give
  // a zero derivative there (see NumericJacobiansMatchReference).
  lm.base_options.numeric_diff_options.method = NumericDiffOptions::Method::kForward;
  lm.relative_reduction_tolerance = 1e-12f;
  return test_utils::WithInnerSettings(lm);
}

/**
 * Chains of n scalars: ½ Σ (x_i - t_i)² + ½ w² Σ (x_{i+1} - x_i)² with
 * lo <= x <= hi and the constant states held at their values. Projected
 * coordinate descent (exact per coordinate) in float64.
 */
std::vector<double> Reference(int num_chains, int n, double w, const std::vector<float> &t,
                              const std::vector<float> &lo, const std::vector<float> &hi,
                              const std::vector<float> &x0, const std::vector<bool> &constant) {
  std::vector<double> x(x0.begin(), x0.end());
  for (int sweep = 0; sweep < 20000; ++sweep) {
    for (int c = 0; c < num_chains; ++c) {
      for (int k = 0; k < n; ++k) {
        const int i = c * n + k;
        if (constant[i]) continue;
        double num = t[i], den = 1.0;
        if (k > 0) num += w * w * x[i - 1], den += w * w;
        if (k + 1 < n) num += w * w * x[i + 1], den += w * w;
        x[i] =
            std::min(std::max(num / den, static_cast<double>(lo[i])), static_cast<double>(hi[i]));
      }
    }
  }
  return x;
}

/**
 * Coupled chains with bounds, a constant state outside the box and an
 * infeasible initial guess, solved through the augmented Lagrangian; checks
 * the result against the float64 reference.
 */
void SolveCoupledChains(int num_chains, JacobianMode jacobians, double tolerance) {
  constexpr int n = 8;
  constexpr float w = 2.f;
  const int total = num_chains * n;
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.f, 2.5f);
  std::vector<float> t(total), lo(total, -1.f), hi(total, 1.f), x0(total, 0.f);
  for (float &v : t) v = normal(rng);
  lo[3] = -kInf;  // one-sided components
  hi[5] = kInf;
  x0[2] = 5.f;  // infeasible initial guess: projected before the first iteration
  // State 0 of chain 0 is constant and outside the box: it must not move.
  std::vector<bool> constant(total, false);
  constant[0] = true;
  x0[0] = 3.f;
  const std::vector<int> const_ids = {0};

  CudaStream stream;
  dvector<float> x(x0), d_t(t), d_lo(lo), d_hi(hi), zeros(std::vector<float>(total, 0.f));
  dvector<int> d_const(const_ids);
  VectorStateBatch<1> states(x.data(), total, d_const.data(), 1);
  states.SetNumActiveStates(total, 1);
  states.SetBounds(d_lo.data(), d_hi.data());
  ASSERT_TRUE(HasBoxBounds(&states));

  PriorVectorFactorBatch<1> prior(reinterpret_cast<const Vector<1> *>(d_t.data()), total);
  prior.SetNumActiveFactors(total);
  const int num_edges = num_chains * (n - 1);
  WeightedFactorBatch<VectorBetweenFactorBatch<1>> smooth(
      w, reinterpret_cast<const Vector<1> *>(zeros.data()), static_cast<size_t>(num_edges));
  smooth.SetNumActiveFactors(num_edges);

  Problem problem;
  problem.AddStateBatch(&states);
  std::vector<float *> prior_ptrs, edge_ptrs;
  for (int i = 0; i < total; ++i) prior_ptrs.push_back(states.StateDevicePtr(i));
  for (int c = 0; c < num_chains; ++c) {
    for (int k = 0; k + 1 < n; ++k) {
      edge_ptrs.push_back(states.StateDevicePtr(c * n + k));
      edge_ptrs.push_back(states.StateDevicePtr(c * n + k + 1));
    }
  }
  problem.AddFactorBatch(&prior, prior_ptrs);
  problem.AddFactorBatch(&smooth, edge_ptrs);
  std::vector<int> ids(total);
  for (int i = 0; i < total; ++i) ids[i] = i / n;
  dvector<int> d_ids(ids);
  if (num_chains > 1) problem.SetProblemPartition(num_chains, {d_ids.data()});

  // Bounds are enforced by the augmented Lagrangian (here without constraint
  // rows: every subproblem is feasible, one full inner solve with the default
  // line search).
  AugmentedLagrangianMinimizer(Options(jacobians)).Minimize(stream.GetStream(), problem);

  std::vector<float> result(total);
  x.CopyToHost(result.data(), total);
  const auto expected = Reference(num_chains, n, w, t, lo, hi, x0, constant);
  for (int i = 0; i < total; ++i) {
    EXPECT_GE(result[i], lo[i]) << "component " << i;
    if (!constant[i]) EXPECT_LE(result[i], hi[i]) << "component " << i;
    EXPECT_NEAR(result[i], expected[i], tolerance) << "component " << i;
  }
  EXPECT_EQ(result[0], 3.f);  // constant, outside the box, untouched
}

class ProjectedBoundsTest : public ::testing::TestWithParam<int> {};

TEST_P(ProjectedBoundsTest, CoupledChainsMatchReference) {
  // Levenberg-Marquardt accepts steps by comparing float32 costs; near the
  // minimum the relative decrease reaches the float32 round-off and it stops a
  // little before the exact minimum.
  SolveCoupledChains(GetParam(), JacobianMode::kAnalytic, 1e-3);
}

// Numeric (forward-difference) Jacobians of states at their bounds: the
// perturbations cross the bound (the bounded Plus would clamp them and give a
// zero derivative at an upper bound, wrongly holding the component there), so
// the solution is the same as with analytic Jacobians.
TEST_P(ProjectedBoundsTest, NumericJacobiansMatchReference) {
  SolveCoupledChains(GetParam(), JacobianMode::kNumeric, 2e-3);
}

INSTANTIATE_TEST_SUITE_P(Chains, ProjectedBoundsTest, ::testing::Values(1, 3));

// A problem that is already at zero cost exits before iterating; its states
// are still projected onto their bounds (single problem and partitioned).
// Gauss-Newton and Levenberg-Marquardt alone reject the bounded problem.
TEST(ProjectedBounds, EarlyExitWritesProjectedStates) {
  for (int chains : {1, 2}) {
    // Targets outside the box: the priors are at zero cost at the start.
    const std::vector<float> start = {3.f, -4.f, 0.5f, 2.f};
    dvector<float> x(start), t(start), lo(std::vector<float>(4, -1.f)),
        hi(std::vector<float>(4, 1.f));
    VectorStateBatch<1> states(x.data(), 4);
    states.SetNumActiveStates(4);
    states.SetBounds(lo.data(), hi.data());
    PriorVectorFactorBatch<1> prior(reinterpret_cast<const Vector<1> *>(t.data()), 4);
    prior.SetNumActiveFactors(4);
    Problem problem;
    problem.AddStateBatch(&states);
    std::vector<float *> ptrs;
    for (int i = 0; i < 4; ++i) ptrs.push_back(states.StateDevicePtr(i));
    problem.AddFactorBatch(&prior, ptrs);
    dvector<int> ids(std::vector<int>{0, 0, 1, 1});
    if (chains > 1) problem.SetProblemPartition(chains, {ids.data()});
    // Huge cost tolerance: every (sub)problem counts as converged at the start.
    MinimizerOptions options;
    options.cost_tolerance = 1e30f;
    options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
    GaussNewtonMinimizer gn(options);
    CudaStream stream;
    EXPECT_THROW(gn.Minimize(stream.GetStream(), problem), std::invalid_argument);
    LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    LevenbergMarquardtMinimizer lm(lm_options);
    EXPECT_THROW(lm.Minimize(stream.GetStream(), problem), std::invalid_argument);
    AugmentedLagrangianMinimizer(test_utils::WithInnerSettings(lm_options))
        .Minimize(stream.GetStream(), problem);
    std::vector<float> result(4);
    x.CopyToHost(result.data(), 4);
    EXPECT_EQ(result, (std::vector<float>{1.f, -1.f, 0.5f, 1.f})) << chains << " subproblems";
  }
}

// A real-time solve of a bounds-only problem (no constraint rows) reads nothing
// back: after a first call that sets the structure up, a call is captured in a
// CUDA graph with exactly the configured budget, and replaying the graph lands
// on the projection.
TEST(ProjectedBounds, RealTimeBoundsOnlySolveIsGraphCapturable) {
  const std::vector<float> start = {0.f, 0.f, 0.f, 0.f}, targets = {3.f, -4.f, 0.5f, 2.f};
  dvector<float> x(start), t(targets), lo(std::vector<float>(4, -1.f)),
      hi(std::vector<float>(4, 1.f));
  VectorStateBatch<1> states(x.data(), 4);
  states.SetNumActiveStates(4);
  states.SetBounds(lo.data(), hi.data());
  PriorVectorFactorBatch<1> prior(reinterpret_cast<const Vector<1> *>(t.data()), 4);
  prior.SetNumActiveFactors(4);
  Problem problem;
  problem.AddStateBatch(&states);
  std::vector<float *> ptrs;
  for (int i = 0; i < 4; ++i) ptrs.push_back(states.StateDevicePtr(i));
  problem.AddFactorBatch(&prior, ptrs);
  AugmentedLagrangianMinimizerOptions options;
  options.base_options.real_time = true;
  options.base_options.reuse_structure = true;
  options.base_options.max_num_iterations = 4;
  options.base_options.max_line_search_steps = 0;
  options.max_outer_iterations = 1;
  options.max_bound_refinements = 0;
  AugmentedLagrangianMinimizer solver(options);
  CudaStream stream;
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kNotDetermined);
  EXPECT_EQ(summary.outer_iterations, 1u);
  EXPECT_EQ(summary.inner_iterations, 4u);

  x.CopyFromHost(start.data(), start.size());
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(cudaStreamBeginCapture(stream.GetStream(), cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  solver.Minimize(stream.GetStream(), problem);
  ASSERT_EQ(cudaStreamEndCapture(stream.GetStream(), &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, stream.GetStream()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.GetStream()), cudaSuccess);
  std::vector<float> result(4);
  x.CopyToHost(result.data(), 4);
  const std::vector<float> expected = {1.f, -1.f, 0.5f, 1.f};
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(result[i], expected[i], 1e-3f) << i;
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
}

/** Four 1-vectors in [-1, 1] pulled by priors to 3, -4, 0.5 and 2: a bounds-only problem. */
struct BoundedPriors {
  const std::vector<float> start = {0.f, 0.f, 0.f, 0.f};
  const std::vector<float> projected = {1.f, -1.f, 0.5f, 1.f};  // the solution
  dvector<float> x{start}, t{std::vector<float>{3.f, -4.f, 0.5f, 2.f}};
  dvector<float> lo{std::vector<float>(4, -1.f)}, hi{std::vector<float>(4, 1.f)};
  VectorStateBatch<1> states{x.data(), 4};
  PriorVectorFactorBatch<1> prior{reinterpret_cast<const Vector<1> *>(t.data()), 4};
  Problem problem;

  BoundedPriors() {
    states.SetNumActiveStates(4);
    states.SetBounds(lo.data(), hi.data());
    prior.SetNumActiveFactors(4);
    problem.AddStateBatch(&states);
    std::vector<float *> ptrs;
    for (int i = 0; i < 4; ++i) ptrs.push_back(states.StateDevicePtr(i));
    problem.AddFactorBatch(&prior, ptrs);
  }

  std::vector<float> X() const {
    std::vector<float> out(4);
    x.CopyToHost(out.data(), 4);
    return out;
  }
};

AugmentedLagrangianMinimizerOptions RealTimeBoundsOptions(size_t refinements) {
  AugmentedLagrangianMinimizerOptions options;
  options.base_options.real_time = true;
  options.base_options.reuse_structure = true;
  options.base_options.max_num_iterations = 4;
  options.base_options.max_line_search_steps = 0;
  options.max_outer_iterations = 1;
  options.max_bound_refinements = refinements;
  return options;
}

// Raising max_bound_refinements after the first call (SetOptions) allocates
// nothing: the refinement buffers are sized with the bound mask, so the next
// call is still captured, and the replayed graph runs the refinement.
TEST(ProjectedBounds, MoreBoundRefinementsAfterTheFirstCallStayCapturable) {
  BoundedPriors p;
  AugmentedLagrangianMinimizer solver(RealTimeBoundsOptions(0));
  CudaStream stream;
  solver.Minimize(stream.GetStream(), p.problem);
  solver.SetOptions(RealTimeBoundsOptions(1));

  p.x.CopyFromHost(p.start.data(), p.start.size());
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(cudaStreamBeginCapture(stream.GetStream(), cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  solver.Minimize(stream.GetStream(), p.problem);
  ASSERT_EQ(cudaStreamEndCapture(stream.GetStream(), &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, stream.GetStream()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.GetStream()), cudaSuccess);
  const auto result = p.X();
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(result[i], p.projected[i], 1e-3f) << i;
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
}

// After the active counts change, reuse_structure sets the bounds up again
// (the per-state columns of the old count are not reused), and a captured call
// throws before it records anything, the projection into the box included.
TEST(ProjectedBounds, ChangedActiveCountsAreSetUpAgain) {
  BoundedPriors p;
  AugmentedLagrangianMinimizer solver(RealTimeBoundsOptions(0));
  CudaStream stream;
  solver.Minimize(stream.GetStream(), p.problem);
  p.states.SetNumActiveStates(2);
  p.prior.SetNumActiveFactors(2);

  p.x.CopyFromHost(p.start.data(), p.start.size());
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamBeginCapture(stream.GetStream(), cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  EXPECT_THROW(solver.Minimize(stream.GetStream(), p.problem), std::invalid_argument);
  ASSERT_EQ(cudaStreamEndCapture(stream.GetStream(), &graph), cudaSuccess);
  size_t num_nodes = 1;
  ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &num_nodes), cudaSuccess);
  EXPECT_EQ(num_nodes, 0u) << "work was recorded before the call threw";
  cudaGraphDestroy(graph);

  solver.Minimize(stream.GetStream(), p.problem);  // outside the capture: set up again
  ASSERT_EQ(cudaStreamSynchronize(stream.GetStream()), cudaSuccess);
  const auto result = p.X();
  EXPECT_NEAR(result[0], p.projected[0], 1e-3f);
  EXPECT_NEAR(result[1], p.projected[1], 1e-3f);
  EXPECT_EQ(result[2], 0.f);  // inactive: untouched
  EXPECT_EQ(result[3], 0.f);
}

TEST(ProjectedBounds, SetBoundsNeedsBothOrNeither) {
  dvector<float> x(std::vector<float>(2, 0.f)), lo(std::vector<float>(2, -1.f));
  VectorStateBatch<2> states(x.data(), 1);
  EXPECT_FALSE(HasBoxBounds(&states));
  EXPECT_THROW(states.SetBounds(lo.data(), nullptr), std::invalid_argument);
  states.SetBounds(lo.data(), lo.data());
  EXPECT_TRUE(HasBoxBounds(&states));
  states.SetBounds(nullptr, nullptr);
  EXPECT_FALSE(HasBoxBounds(&states));
}

}  // namespace
}  // namespace cunls
