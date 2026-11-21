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

// AugmentedLagrangianMinimizer (augmented Lagrangian) on problems with closed-form
// solutions: projections onto a box, a halfspace and an affine subspace
// (KKT), a batch of box projections with a different active set per
// subproblem, and an SE2 trajectory whose last pose is constrained to a goal.

#include "cunls/minimizer/augmented_lagrangian_minimizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/bound_factor_batch.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/halfspace_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

/** A Levenberg-Marquardt minimizer run to convergence (for plain solves). */
std::unique_ptr<LevenbergMarquardtMinimizer> MakeMinimizer(size_t max_num_iterations = 50) {
  MinimizerOptions options = test_utils::ConvergedSolveOptions();
  options.max_num_iterations = max_num_iterations;
  return test_utils::MakeLevenbergMarquardt(options, 1e-12f);
}

/** Augmented Lagrangian options whose inner solves run to convergence. */
AugmentedLagrangianMinimizerOptions Options() {
  AugmentedLagrangianMinimizerOptions options;
  const size_t line_search = options.base_options.max_line_search_steps;
  options.base_options = test_utils::ConvergedSolveOptions();
  options.base_options.max_line_search_steps = line_search;
  options.relative_reduction_tolerance = 1e-12f;
  return options;
}

/**
 * @brief min ½‖x - t‖² s.t. lower <= x <= upper over target.size() / Dim
 * states of dimension Dim, starting at x = 0; with `partitioned`, every state
 * is its own subproblem. The solution is Projection(i) = clamp(t, lower, upper).
 */
template <int Dim>
struct BoxProblem {
  BoxProblem(const std::vector<float> &target, const std::vector<float> &lower,
             const std::vector<float> &upper, bool partitioned = false)
      : target_host(target),
        lower_host(lower),
        upper_host(upper),
        n(target.size() / Dim),
        x(std::vector<float>(target.size(), 0.f)),
        t(target),
        lo(lower),
        hi(upper),
        states(x.data(), n),
        prior(reinterpret_cast<const Vector<Dim> *>(t.data()), n),
        bounds(lo.data(), hi.data(), n) {
    states.SetNumActiveStates(n);
    prior.SetNumActiveFactors(n);
    bounds.SetNumActiveFactors(n);
    std::vector<float *> ptrs;
    std::vector<int> ids_host;
    for (size_t s = 0; s < n; ++s) {
      ptrs.push_back(states.StateDevicePtr(s));
      ids_host.push_back(static_cast<int>(s));
    }
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&prior, ptrs);
    problem.AddFactorBatch(&bounds, ptrs);
    if (partitioned) {
      ids = dvector<int>(ids_host);
      problem.SetProblemPartition(n, {ids.data()});
    }
  }

  float Projection(size_t i) const {
    return std::min(std::max(target_host[i], lower_host[i]), upper_host[i]);
  }

  std::vector<float> target_host, lower_host, upper_host;
  size_t n;
  dvector<float> x, t, lo, hi;
  dvector<int> ids;
  VectorStateBatch<Dim> states;
  PriorVectorFactorBatch<Dim> prior;
  BoundFactorBatch<Dim> bounds;
  Problem problem;
};

/** Targets and bounds of the two-state box test (some sides unbounded). */
struct MixedBox {
  static constexpr float kInf = std::numeric_limits<float>::infinity();
  std::vector<float> target = {2.f, -3.f, 0.5f, 7.f, -4.f, 0.25f};
  std::vector<float> lower = {-1.f, -1.f, -1.f, -kInf, -2.f, 0.5f};
  std::vector<float> upper = {1.f, 1.f, 1.f, 3.f, kInf, kInf};
};

// min ½‖x - t‖² s.t. lo <= x <= hi: x = clamp(t, lo, hi). Some components are
// unbounded on one side (±inf), some targets are inside the box.
TEST(AugmentedLagrangianMinimizerTest, BoxProjection) {
  const MixedBox box;
  BoxProblem<3> b(box.target, box.lower, box.upper);
  CudaStream stream;
  AugmentedLagrangianMinimizer solver(Options());
  const auto summary = solver.Minimize(stream.GetStream(), b.problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LE(summary.max_violation, 1e-4f);
  const auto result = test_utils::ToHost(b.x);
  double expected_cost = 0.0;
  for (size_t i = 0; i < 6; ++i) {
    const float expected = b.Projection(i);
    EXPECT_NEAR(result[i], expected, 2e-4f) << "component " << i;
    expected_cost += 0.5 * (expected - box.target[i]) * (expected - box.target[i]);
  }
  EXPECT_NEAR(summary.final_cost, expected_cost, 1e-3 * expected_cost);
}

// min ½‖x - t‖² s.t. aᵀx <= b: x = t - max(0, aᵀt - b) a / ‖a‖².
TEST(AugmentedLagrangianMinimizerTest, HalfspaceProjection) {
  const std::array<float, 2> a = {1.f, 2.f};
  const float b = 1.f;
  for (const std::array<float, 2> target : {std::array<float, 2>{3.f, 4.f}, {-1.f, 0.5f}}) {
    CudaStream stream;
    dvector<float> x(std::vector<float>(2, 0.f));
    dvector<float> t(std::vector<float>(target.begin(), target.end()));
    dvector<float> normal(std::vector<float>(a.begin(), a.end())), offset(std::vector<float>{b});
    VectorStateBatch<2> states(x.data(), 1);
    states.SetNumActiveStates(1);
    PriorVectorFactorBatch<2> prior(reinterpret_cast<const Vector<2> *>(t.data()), 1);
    prior.SetNumActiveFactors(1);
    HalfspaceFactorBatch<2> halfspace(normal.data(), offset.data(), 1);
    halfspace.SetNumActiveFactors(1);
    ConstraintFactorBatch constraint(&halfspace, ConstraintKind::kInequality);
    Problem problem;
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
    problem.AddFactorBatch(&constraint, {states.StateDevicePtr(0)});

    AugmentedLagrangianMinimizer solver(Options());
    const auto summary = solver.Minimize(stream.GetStream(), problem);
    EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
    const float excess = std::max(0.f, a[0] * target[0] + a[1] * target[1] - b) / 5.f;
    const auto result = test_utils::ToHost(x);
    EXPECT_NEAR(result[0], target[0] - excess * a[0], 2e-4f);
    EXPECT_NEAR(result[1], target[1] - excess * a[1], 2e-4f);
  }
}

// The runtime-sized vector classes at a size with no fixed-size class: state 0 is projected on
// a box, state 1 on the halfspace sum(x) <= -3.
TEST(AugmentedLagrangianMinimizerTest, RuntimeSizedVectorProjections) {
  constexpr int kDim = 13;
  constexpr float kOffset = -3.f;
  std::vector<float> target(2 * kDim);
  for (int i = 0; i < 2 * kDim; ++i) target[i] = 0.5f * static_cast<float>(i % 7) - 1.5f;
  dvector<float> x(std::vector<float>(2 * kDim, 0.f)), t(target);
  dvector<float> lo(std::vector<float>(kDim, -1.f)), hi(std::vector<float>(kDim, 1.f));
  dvector<float> normal(std::vector<float>(kDim, 1.f)), offset(std::vector<float>{kOffset});
  DynamicVectorStateBatch states(kDim, x.data(), 2);
  states.SetNumActiveStates(2);
  DynamicPriorVectorFactorBatch prior(kDim, t.data(), 2);
  prior.SetNumActiveFactors(2);
  DynamicBoundFactorBatch bound(kDim, lo.data(), hi.data(), 1);
  bound.SetNumActiveFactors(1);
  DynamicHalfspaceFactorBatch halfspace(kDim, normal.data(), offset.data(), 1);
  halfspace.SetNumActiveFactors(1);
  ConstraintFactorBatch constraint(&halfspace, ConstraintKind::kInequality);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0), states.StateDevicePtr(1)});
  problem.AddFactorBatch(&bound, {states.StateDevicePtr(0)});
  problem.AddFactorBatch(&constraint, {states.StateDevicePtr(1)});

  CudaStream stream;
  AugmentedLagrangianMinimizer solver(Options());
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  float sum = 0.f;
  for (int i = 0; i < kDim; ++i) sum += target[kDim + i];
  const float excess = std::max(0.f, sum - kOffset) / kDim;
  ASSERT_GT(excess, 0.1f);  // the halfspace is active
  const auto result = test_utils::ToHost(x);
  for (int i = 0; i < kDim; ++i) {
    EXPECT_NEAR(result[i], std::clamp(target[i], -1.f, 1.f), 2e-4f) << "box " << i;
    EXPECT_NEAR(result[kDim + i], target[kDim + i] - excess, 2e-4f) << "halfspace " << i;
  }
}

// min ½‖x - t‖² s.t. A x = c (2 rows, x in R^4): the KKT solution
// x = t - Aᵀ (A Aᵀ)⁻¹ (A t - c).
TEST(AugmentedLagrangianMinimizerTest, AffineEqualityMatchesKkt) {
  const std::array<double, 8> A = {1, 2, 0, -1, 0, 1, 3, 1};
  const std::array<double, 2> c = {1, -2};
  const std::array<double, 4> target = {0.3, -1.2, 2.0, 0.7};
  // Reference in double.
  std::array<double, 2> residual{};
  std::array<double, 4> aat{};
  for (int i = 0; i < 2; ++i) {
    residual[i] = -c[i];
    for (int k = 0; k < 4; ++k) residual[i] += A[i * 4 + k] * target[k];
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 4; ++k) aat[i * 2 + j] += A[i * 4 + k] * A[j * 4 + k];
  }
  const double det = aat[0] * aat[3] - aat[1] * aat[2];
  const std::array<double, 2> y = {(aat[3] * residual[0] - aat[1] * residual[1]) / det,
                                   (-aat[2] * residual[0] + aat[0] * residual[1]) / det};
  std::array<double, 4> expected{};
  for (int k = 0; k < 4; ++k) expected[k] = target[k] - A[k] * y[0] - A[4 + k] * y[1];

  CudaStream stream;
  dvector<float> x(std::vector<float>(4, 0.f));
  dvector<float> t(std::vector<float>(target.begin(), target.end()));
  dvector<float> normals(std::vector<float>(A.begin(), A.end()));
  dvector<float> offsets(std::vector<float>(c.begin(), c.end()));
  VectorStateBatch<4> states(x.data(), 1);
  states.SetNumActiveStates(1);
  PriorVectorFactorBatch<4> prior(reinterpret_cast<const Vector<4> *>(t.data()), 1);
  prior.SetNumActiveFactors(1);
  HalfspaceFactorBatch<4> rows(normals.data(), offsets.data(), 2);
  rows.SetNumActiveFactors(2);
  ConstraintFactorBatch constraint(&rows, ConstraintKind::kEquality);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
  problem.AddFactorBatch(&constraint, {states.StateDevicePtr(0), states.StateDevicePtr(0)});

  AugmentedLagrangianMinimizer solver(Options());
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LE(summary.max_violation, 1e-4f);
  const auto result = test_utils::ToHost(x);
  for (int k = 0; k < 4; ++k) EXPECT_NEAR(result[k], expected[k], 2e-4) << "component " << k;
  // The multipliers converge to the KKT multipliers y (x = t - Aᵀλ).
  std::vector<float> lambda(2);
  THROW_ON_CUDA_ERROR(cudaMemcpy(lambda.data(), constraint.Multipliers(), 2 * sizeof(float),
                                 cudaMemcpyDeviceToHost));
  EXPECT_NEAR(lambda[0], y[0], 1e-2 * std::max(1.0, std::fabs(y[0])));
  EXPECT_NEAR(lambda[1], y[1], 1e-2 * std::max(1.0, std::fabs(y[1])));
}

// B box projections in one partitioned problem; each subproblem has its own
// target and so its own active set, and stops on its own.
TEST(AugmentedLagrangianMinimizerTest, BatchedBoxProjections) {
  constexpr int kBatch = 64;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uniform(-3.f, 3.f);
  std::vector<float> target(kBatch * 3), lower(kBatch * 3), upper(kBatch * 3);
  for (int i = 0; i < kBatch * 3; ++i) {
    target[i] = uniform(rng);
    // Width varies with the subproblem: from all components active to none.
    const float half = 0.1f + 3.f * static_cast<float>(i / 3) / kBatch;
    lower[i] = -half;
    upper[i] = half;
  }
  BoxProblem<3> b(target, lower, upper, /*partitioned=*/true);
  CudaStream stream;

  AugmentedLagrangianMinimizer solver(Options());
  const auto summary = solver.Minimize(stream.GetStream(), b.problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_EQ(summary.num_problems, static_cast<size_t>(kBatch));
  EXPECT_EQ(summary.num_converged, static_cast<size_t>(kBatch));
  // Finished subproblems keep their multipliers, so they stay feasible while
  // the others iterate.
  EXPECT_LE(summary.max_violation, 1e-4f);
  const auto result = test_utils::ToHost(b.x);
  for (int i = 0; i < kBatch * 3; ++i) {
    EXPECT_NEAR(result[i], b.Projection(i), 2e-4f) << "subproblem " << i / 3;
  }
}

// A batch where one subproblem is infeasible (lower bound above the upper
// one): it stops at max_penalty, and every other subproblem still finishes
// feasible and stationary at its projection. Each subproblem gets its own full
// inner solve once it is feasible, independently of the infeasible one.
TEST(AugmentedLagrangianMinimizerTest, BatchFinishesBesideAnInfeasibleSubproblem) {
  constexpr int kBatch = 25, kInfeasible = 12;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> uniform(-3.f, 3.f);
  std::vector<float> target(kBatch), lower(kBatch), upper(kBatch);
  for (int i = 0; i < kBatch; ++i) {
    target[i] = uniform(rng);
    lower[i] = -1.f + 0.02f * i;
    upper[i] = 1.f + 0.02f * i;
  }
  lower[kInfeasible] = 1.f;
  upper[kInfeasible] = -1.f;
  BoxProblem<1> b(target, lower, upper, /*partitioned=*/true);
  CudaStream stream;

  AugmentedLagrangianMinimizerOptions options = Options();
  options.max_penalty = 1e4f;
  AugmentedLagrangianMinimizer solver(options);
  const auto summary = solver.Minimize(stream.GetStream(), b.problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kMaxPenalty);
  EXPECT_EQ(summary.num_converged, static_cast<size_t>(kBatch - 1));
  EXPECT_EQ(summary.num_max_penalty, 1u);
  const auto result = test_utils::ToHost(b.x);
  for (int i = 0; i < kBatch; ++i) {
    if (i == kInfeasible) continue;
    EXPECT_NEAR(result[i], b.Projection(i), 2e-4f) << "subproblem " << i;
  }
}

// An SE2 chain (prior on the first pose, odometry between poses) whose last
// pose must equal a goal that the odometry does not reach: the constraint
// holds to tolerance, and the solution is the limit of ever stiffer soft
// goal priors.
TEST(AugmentedLagrangianMinimizerTest, Se2GoalEquality) {
  constexpr int kPoses = 6;
  auto pose = [](float x, float y, float th) {
    const float c = std::cos(th), s = std::sin(th);
    return std::vector<float>{c, -s, x, s, c, y, 0, 0, 1};
  };
  auto solve = [&](bool hard, float soft_weight, std::vector<float> *last_pose) {
    CudaStream stream;
    std::vector<float> init, odometry;
    for (int i = 0; i < kPoses; ++i) {
      const auto p = pose(static_cast<float>(i), 0.f, 0.f);
      init.insert(init.end(), p.begin(), p.end());
    }
    for (int i = 0; i + 1 < kPoses; ++i) {
      // The between factor's delta for a unit step forward with a slight turn.
      const auto d = pose(-1.f, 0.f, -0.1f);
      odometry.insert(odometry.end(), d.begin(), d.end());
    }
    dvector<float> poses(init), deltas(odometry), first(pose(0.f, 0.f, 0.f)),
        goal(pose(4.f, 2.f, 0.8f));
    SE2StateBatch states(poses.data(), kPoses);
    states.SetNumActiveStates(kPoses);
    SE2PriorFactorBatch prior(reinterpret_cast<const SE2Transform *>(first.data()), 1);
    prior.SetNumActiveFactors(1);
    SE2BetweenFactorBatch between(reinterpret_cast<const SE2Transform *>(deltas.data()),
                                  kPoses - 1);
    between.SetNumActiveFactors(kPoses - 1);
    SE2PriorFactorBatch goal_prior(reinterpret_cast<const SE2Transform *>(goal.data()), 1);
    goal_prior.SetNumActiveFactors(1);
    ConstraintFactorBatch goal_constraint(&goal_prior, ConstraintKind::kEquality);
    WeightedFactorBatch<SE2PriorFactorBatch> soft_goal(
        soft_weight, reinterpret_cast<const SE2Transform *>(goal.data()), 1);
    soft_goal.SetNumActiveFactors(1);
    std::vector<float *> between_ptrs;
    for (int i = 0; i + 1 < kPoses; ++i) {
      between_ptrs.push_back(states.StateDevicePtr(i));
      between_ptrs.push_back(states.StateDevicePtr(i + 1));
    }
    Problem problem;
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
    problem.AddFactorBatch(&between, between_ptrs);
    if (hard) {
      problem.AddFactorBatch(&goal_constraint, {states.StateDevicePtr(kPoses - 1)});
    } else {
      problem.AddFactorBatch(&soft_goal, {states.StateDevicePtr(kPoses - 1)});
    }
    AugmentedLagrangianMinimizer solver(Options());
    const auto summary = solver.Minimize(stream.GetStream(), problem);
    const auto all = test_utils::ToHost(poses);
    last_pose->assign(all.end() - 9, all.end());
    return summary;
  };

  std::vector<float> hard_pose, soft_pose;
  const auto summary = solve(true, 0.f, &hard_pose);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LE(summary.max_violation, 1e-4f);
  EXPECT_NEAR(hard_pose[2], 4.f, 1e-3f);
  EXPECT_NEAR(hard_pose[5], 2.f, 1e-3f);
  EXPECT_NEAR(std::atan2(hard_pose[3], hard_pose[0]), 0.8f, 1e-3f);
  EXPECT_GT(summary.final_cost, 0.f);
  // A stiff soft goal (weight 300: penalty 9e4) lands within ~1e-4 of it.
  solve(false, 300.f, &soft_pose);
  for (int k : {0, 1, 2, 3, 4, 5}) EXPECT_NEAR(hard_pose[k], soft_pose[k], 2e-3f) << k;
}

// Without constraint batches every subproblem is feasible from the start: its
// first inner solve is its full solve, one outer iteration.
TEST(AugmentedLagrangianMinimizerTest, NoConstraintsIsOneFullSolve) {
  CudaStream stream;
  dvector<float> x(std::vector<float>(3, 0.f)), t(std::vector<float>{1.f, 2.f, 3.f});
  VectorStateBatch<3> states(x.data(), 1);
  states.SetNumActiveStates(1);
  PriorVectorFactorBatch<3> prior(reinterpret_cast<const Vector<3> *>(t.data()), 1);
  prior.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
  AugmentedLagrangianMinimizer solver(Options());
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_EQ(summary.outer_iterations, 1u);
  const auto result = test_utils::ToHost(x);
  EXPECT_NEAR(result[0], 1.f, 1e-5f);
  EXPECT_NEAR(result[2], 3.f, 1e-5f);
}

// A capped solve names the cap that stopped it: max_num_iterations for a plain
// Minimize(), the AugmentedLagrangianMinimizer's budget (base_options.max_num_iterations)
// when it runs the solve, where raising max_num_iterations would not help.
TEST(AugmentedLagrangianMinimizerTest, IterationCapMessageNamesTheCapThatApplied) {
  CudaStream stream;
  const float c = std::cos(2.5f), s = std::sin(2.5f);
  dvector<float> x(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1}),
      t(std::vector<float>{c, -s, 3.f, s, c, 2.f, 0, 0, 1});
  SE2StateBatch states(x.data(), 1);
  states.SetNumActiveStates(1);
  SE2PriorFactorBatch prior(reinterpret_cast<const SE2Transform *>(t.data()), 1);
  prior.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
  const MinimizerSummary plain = MakeMinimizer(1)->Minimize(stream.GetStream(), problem);
  ASSERT_EQ(plain.termination_reason, TerminationReason::MaxIterations) << plain.message;
  EXPECT_NE(plain.message.find("reached max_num_iterations (1)"), std::string::npos)
      << plain.message;

  x.CopyFromHost(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1}.data(), 9);
  AugmentedLagrangianMinimizerOptions al = Options();
  al.max_outer_iterations = 1;
  al.base_options.max_num_iterations = 1;
  AugmentedLagrangianMinimizer solver(al);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kMaxOuterIterations);
  EXPECT_EQ(summary.inner_iterations, 1u);
  EXPECT_NE(summary.message.find("base_options.max_num_iterations"), std::string::npos)
      << summary.message;
  EXPECT_EQ(summary.message.find("raise max_num_iterations"), std::string::npos) << summary.message;
}

// An inner solve that fails (here: a NaN objective, NonFiniteCost) while the
// constraints happen to hold is not convergence: the subproblem is reported as
// failed, with the inner reason, instead of kConverged.
TEST(AugmentedLagrangianMinimizerTest, FailedInnerSolveIsNotConvergence) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  CudaStream stream;
  dvector<float> x(std::vector<float>(2, 0.f)), t(std::vector<float>{nan, 1.f});
  dvector<float> normal(std::vector<float>{1.f, 2.f}), offset(std::vector<float>{1.f});
  VectorStateBatch<2> states(x.data(), 1);
  states.SetNumActiveStates(1);
  PriorVectorFactorBatch<2> prior(reinterpret_cast<const Vector<2> *>(t.data()), 1);
  prior.SetNumActiveFactors(1);
  HalfspaceFactorBatch<2> halfspace(normal.data(), offset.data(), 1);  // x = 0 satisfies it
  halfspace.SetNumActiveFactors(1);
  ConstraintFactorBatch constraint(&halfspace, ConstraintKind::kInequality);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
  problem.AddFactorBatch(&constraint, {states.StateDevicePtr(0)});
  AugmentedLagrangianMinimizer solver(Options());
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kInnerSolveFailed)
      << summary.message;
  EXPECT_EQ(summary.num_converged, 0u);
  EXPECT_EQ(summary.num_inner_failed, 1u);
  EXPECT_NE(summary.message.find("non-finite"), std::string::npos) << summary.message;
}

// Contradicting bounds (lower > upper): no feasible point. The penalty
// reaches its cap and the solve ends with kMaxPenalty, not a loop.
TEST(AugmentedLagrangianMinimizerTest, InfeasibleStopsAtMaxPenalty) {
  BoxProblem<1> b({0.f}, {1.f}, {-1.f});
  CudaStream stream;
  AugmentedLagrangianMinimizerOptions options = Options();
  options.max_penalty = 1e4f;
  AugmentedLagrangianMinimizer solver(options);
  const auto summary = solver.Minimize(stream.GetStream(), b.problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kMaxPenalty);
  EXPECT_EQ(summary.num_max_penalty, 1u);
  EXPECT_LT(summary.outer_iterations, options.max_outer_iterations);
  EXPECT_NEAR(summary.max_violation, 1.f, 1e-2f);  // x = 0 is the least-violating point
}

// Re-solving with warm_start keeps the multipliers and penalties: the second
// solve of the same problem from a perturbed start needs fewer outer
// iterations and reaches the same solution.
TEST(AugmentedLagrangianMinimizerTest, WarmStartReusesMultipliers) {
  BoxProblem<3> b({2.f, -3.f, 0.5f}, std::vector<float>(3, -1.f), std::vector<float>(3, 1.f));
  CudaStream stream;
  AugmentedLagrangianMinimizerOptions options = Options();
  options.warm_start = true;
  AugmentedLagrangianMinimizer solver(options);
  const auto cold = solver.Minimize(stream.GetStream(), b.problem);
  ASSERT_EQ(cold.status, AugmentedLagrangianMinimizerStatus::kConverged);
  const std::vector<float> perturbed = {0.9f, -0.8f, 0.3f};
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(b.x.data(), perturbed.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
  const auto warm = solver.Minimize(stream.GetStream(), b.problem);
  EXPECT_EQ(warm.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LT(warm.outer_iterations, cold.outer_iterations);
  const auto result = test_utils::ToHost(b.x);
  EXPECT_NEAR(result[0], 1.f, 2e-4f);
  EXPECT_NEAR(result[1], -1.f, 2e-4f);
  EXPECT_NEAR(result[2], 0.5f, 2e-4f);
}

// Real time: a fixed budget without per-iteration read-backs reaches the box
// projection; a warm follow-up call keeps it.
TEST(AugmentedLagrangianMinimizerTest, RealTimeBudget) {
  const MixedBox box;
  BoxProblem<3> b(box.target, box.lower, box.upper);
  CudaStream stream;
  AugmentedLagrangianMinimizerOptions options = Options();
  options.base_options.real_time = true;
  options.max_outer_iterations = 12;
  options.inner_iterations = 3;
  options.base_options.max_num_iterations = 3;
  options.warm_start = true;
  options.base_options.reuse_structure = true;
  AugmentedLagrangianMinimizer solver(options);
  for (int call = 0; call < 2; ++call) {
    const auto summary = solver.Minimize(stream.GetStream(), b.problem);
    EXPECT_EQ(summary.outer_iterations, 12u);
    EXPECT_TRUE(std::isnan(summary.final_cost));
    EXPECT_LE(summary.max_violation, 1e-3f) << "call " << call;
    const auto result = test_utils::ToHost(b.x);
    for (size_t i = 0; i < 6; ++i) {
      EXPECT_NEAR(result[i], b.Projection(i), 1e-3f) << "call " << call << " component " << i;
    }
  }
}

// Robust objective under constraints: min Σ ρ((x - t_i)²) s.t. x >= 0.5, with
// t = {0, 0.1, -0.1, 0.05, 10} (one outlier). Without a loss the mean 2.01
// satisfies the bound (inactive); with a Huber loss the outlier is
// down-weighted, the robust estimate lies below 0.5, and the bound is active.
TEST(AugmentedLagrangianMinimizerTest, RobustObjective) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::vector<float> targets = {0.f, 0.1f, -0.1f, 0.05f, 10.f};
  for (const bool robust : {false, true}) {
    CudaStream stream;
    dvector<float> x(std::vector<float>{3.f}), t(targets);
    dvector<float> lo(std::vector<float>{0.5f}), hi(std::vector<float>{kInf});
    VectorStateBatch<1> states(x.data(), 1);
    states.SetNumActiveStates(1);
    PriorVectorFactorBatch<1> prior(reinterpret_cast<const Vector<1> *>(t.data()), 5);
    prior.SetNumActiveFactors(5);
    HuberLossFunctionBatch huber(0.5f);
    BoundFactorBatch<1> bound(lo.data(), hi.data(), 1);
    bound.SetNumActiveFactors(1);
    Problem problem;
    problem.AddStateBatch(&states);
    const std::vector<float *> ptrs(5, states.StateDevicePtr(0));
    if (robust) {
      problem.AddFactorBatch(&prior, &huber, ptrs);
    } else {
      problem.AddFactorBatch(&prior, ptrs);
    }
    problem.AddFactorBatch(&bound, {states.StateDevicePtr(0)});
    AugmentedLagrangianMinimizer solver(Options());
    const auto summary = solver.Minimize(stream.GetStream(), problem);
    EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged) << robust;
    EXPECT_NEAR(test_utils::ToHost(x)[0], robust ? 0.5f : 2.01f, 2e-3f) << "robust " << robust;
  }
}

// Constraints are hard: a robust loss on a constraint batch is rejected, and
// the plain minimizers reject constraint batches (only the augmented
// Lagrangian manages their multipliers and penalties).
TEST(AugmentedLagrangianMinimizerTest, RejectsMisusedConstraints) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  CudaStream stream;
  dvector<float> x(std::vector<float>{3.f}), t(std::vector<float>{0.f});
  dvector<float> lo(std::vector<float>{0.5f}), hi(std::vector<float>{kInf});
  VectorStateBatch<1> states(x.data(), 1);
  states.SetNumActiveStates(1);
  PriorVectorFactorBatch<1> prior(reinterpret_cast<const Vector<1> *>(t.data()), 1);
  prior.SetNumActiveFactors(1);
  BoundFactorBatch<1> bound(lo.data(), hi.data(), 1);
  bound.SetNumActiveFactors(1);
  HuberLossFunctionBatch huber(0.5f);
  const std::vector<float *> ptr = {states.StateDevicePtr(0)};
  {
    Problem problem;
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&prior, ptr);
    problem.AddFactorBatch(&bound, &huber, ptr);
    AugmentedLagrangianMinimizer solver(Options());
    EXPECT_THROW(solver.Minimize(stream.GetStream(), problem), std::invalid_argument);
  }
  {
    Problem problem;
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&prior, ptr);
    problem.AddFactorBatch(&bound, ptr);
    auto plain = MakeMinimizer();
    EXPECT_THROW(plain->Minimize(stream.GetStream(), problem), std::invalid_argument);
  }
}

TEST(AugmentedLagrangianMinimizerOptions, RejectsInvalidOptions) {
  AugmentedLagrangianMinimizerOptions options;
  options.max_outer_iterations = 0;
  EXPECT_THROW(AugmentedLagrangianMinimizer{options}, std::invalid_argument);
  options = AugmentedLagrangianMinimizerOptions();
  options.inner_iterations = 0;
  EXPECT_THROW(AugmentedLagrangianMinimizer{options}, std::invalid_argument);
  options = AugmentedLagrangianMinimizerOptions();
  options.base_options.max_num_iterations = 0;
  EXPECT_THROW(AugmentedLagrangianMinimizer{options}, std::invalid_argument);
  EXPECT_EQ(AugmentedLagrangianMinimizerOptions().base_options.max_line_search_steps, 10u);
  AugmentedLagrangianMinimizer solver;
  options = AugmentedLagrangianMinimizerOptions();
  options.max_outer_iterations = 0;
  EXPECT_THROW(solver.SetOptions(options), std::invalid_argument);
}

// The AL transform: equality r = √ρ (s c + λ/ρ); an inactive inequality row
// (s c + μ/ρ <= 0) has zero residual and zero Jacobian.
TEST(ConstraintFactorBatch, AugmentedLagrangianRows) {
  CudaStream stream;
  dvector<float> x(std::vector<float>{0.5f, 3.f});
  dvector<float> t(std::vector<float>{1.f, 1.f});
  VectorStateBatch<2> states(x.data(), 1);
  states.SetNumActiveStates(1);
  PriorVectorFactorBatch<2> prior(reinterpret_cast<const Vector<2> *>(t.data()), 1);
  prior.SetNumActiveFactors(1);  // c = x - t = (-0.5, 2)
  dvector<float *> ptrs(std::vector<float *>{states.StateDevicePtr(0)});
  dvector<float> res(2), jac(4);
  for (const auto kind : {ConstraintKind::kEquality, ConstraintKind::kInequality}) {
    ConstraintFactorBatch constraint(&prior, kind, /*scale=*/2.f);
    constraint.SetPenalty(4.f, stream.GetStream());
    const std::vector<float> lambda = {0.4f, -1.f};
    THROW_ON_CUDA_ERROR(cudaMemcpy(constraint.Multipliers(), lambda.data(), 2 * sizeof(float),
                                   cudaMemcpyHostToDevice));
    ASSERT_TRUE(constraint.Evaluate(res.data(), jac.data(), ptrs.data(), stream.GetStream()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    const auto r = test_utils::ToHost(res), j = test_utils::ToHost(jac);
    // z = s c + λ/ρ = (-1 + 0.1, 4 - 0.25) = (-0.9, 3.75); √ρ = 2.
    if (kind == ConstraintKind::kEquality) {
      EXPECT_NEAR(r[0], -1.8f, 1e-6f);
      EXPECT_NEAR(j[0], 4.f, 1e-6f);  // s √ρ
    } else {
      EXPECT_EQ(r[0], 0.f);
      EXPECT_EQ(j[0], 0.f);
    }
    EXPECT_NEAR(r[1], 7.5f, 1e-5f);
    EXPECT_EQ(j[1], 0.f);
    EXPECT_NEAR(j[3], 4.f, 1e-6f);
  }
}

}  // namespace
}  // namespace cunls
