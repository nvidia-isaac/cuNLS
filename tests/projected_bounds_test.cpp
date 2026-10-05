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
// AugmentedLagrangianMinimizer (projected Gauss-Newton in its inner solves,
// with a Gauss-Newton or Levenberg-Marquardt inner minimizer): coupled
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

namespace cunls {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

enum class Kind { kGaussNewton, kLevenbergMarquardt };

std::unique_ptr<Minimizer> MakeMinimizer(Kind kind,
                                         JacobianMode jacobians = JacobianMode::kAnalytic) {
  MinimizerOptions options;
  options.max_num_iterations = 50;
  options.jacobian_mode = jacobians;
  // One-sided differences: a perturbation clamped at an upper bound would give
  // a zero derivative there (see NumericJacobiansMatchReference).
  options.numeric_diff_options.method = NumericDiffOptions::Method::kForward;
  options.state_tolerance = 1e-12f;
  options.cost_tolerance = 1e-12f;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
  if (kind == Kind::kGaussNewton) return std::make_unique<GaussNewtonMinimizer>(options);
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = options;
  lm.relative_reduction_tolerance = 1e-12f;
  return std::make_unique<LevenbergMarquardtMinimizer>(lm);
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
void SolveCoupledChains(Kind kind, int num_chains, JacobianMode jacobians, double tolerance) {
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
  // rows: one projected inner solve, with its default line search).
  auto minimizer = MakeMinimizer(kind, jacobians);
  AugmentedLagrangianMinimizer(*minimizer).Minimize(stream.GetStream(), problem);

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

class ProjectedBoundsTest : public ::testing::TestWithParam<std::tuple<Kind, int>> {};

TEST_P(ProjectedBoundsTest, CoupledChainsMatchReference) {
  // Levenberg-Marquardt accepts steps by comparing float32 costs; near the
  // minimum the relative decrease reaches the float32 round-off and it stops a
  // little earlier than Gauss-Newton's exact Newton step.
  const Kind kind = std::get<0>(GetParam());
  SolveCoupledChains(kind, std::get<1>(GetParam()), JacobianMode::kAnalytic,
                     kind == Kind::kGaussNewton ? 2e-4 : 1e-3);
}

// Numeric (forward-difference) Jacobians of states at their bounds: the
// perturbations cross the bound (the bounded Plus would clamp them and give a
// zero derivative at an upper bound, wrongly holding the component there), so
// the solution is the same as with analytic Jacobians.
TEST_P(ProjectedBoundsTest, NumericJacobiansMatchReference) {
  SolveCoupledChains(std::get<0>(GetParam()), std::get<1>(GetParam()), JacobianMode::kNumeric,
                     2e-3);
}

INSTANTIATE_TEST_SUITE_P(Minimizers, ProjectedBoundsTest,
                         ::testing::Combine(::testing::Values(Kind::kGaussNewton,
                                                              Kind::kLevenbergMarquardt),
                                            ::testing::Values(1, 3)));

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
    GaussNewtonMinimizer minimizer(options);
    CudaStream stream;
    EXPECT_THROW(minimizer.Minimize(stream.GetStream(), problem), std::invalid_argument);
    LevenbergMarquardtMinimizer lm;
    EXPECT_THROW(lm.Minimize(stream.GetStream(), problem), std::invalid_argument);
    AugmentedLagrangianMinimizer(minimizer).Minimize(stream.GetStream(), problem);
    std::vector<float> result(4);
    x.CopyToHost(result.data(), 4);
    EXPECT_EQ(result, (std::vector<float>{1.f, -1.f, 0.5f, 1.f})) << chains << " subproblems";
  }
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
