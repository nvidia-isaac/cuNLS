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
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

enum class Kind { kGaussNewton, kLevenbergMarquardt };

std::unique_ptr<GaussNewtonMinimizer> MakeMinimizer(Kind kind) {
  MinimizerOptions options;
  options.max_num_iterations = 50;
  options.state_tolerance = 1e-12f;
  options.cost_tolerance = 1e-12f;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
  if (kind == Kind::kGaussNewton) return std::make_unique<GaussNewtonMinimizer>(options);
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = options;
  lm.relative_reduction_tolerance = 1e-12f;
  return std::make_unique<LevenbergMarquardtMinimizer>(lm);
}

std::vector<float> Download(const dvector<float> &v) {
  std::vector<float> out(v.size());
  v.CopyToHost(out.data(), out.size());
  return out;
}

class AugmentedLagrangianMinimizerTest : public ::testing::TestWithParam<Kind> {};

// min ½‖x - t‖² s.t. lo <= x <= hi: x = clamp(t, lo, hi). Some components are
// unbounded on one side (±inf), some targets are inside the box.
TEST_P(AugmentedLagrangianMinimizerTest, BoxProjection) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::vector<float> target = {2.f, -3.f, 0.5f, 7.f, -4.f, 0.25f};
  const std::vector<float> lower = {-1.f, -1.f, -1.f, -kInf, -2.f, 0.5f};
  const std::vector<float> upper = {1.f, 1.f, 1.f, 3.f, kInf, kInf};
  CudaStream stream;
  dvector<float> x(std::vector<float>(6, 0.f)), t(target), lo(lower), hi(upper);
  VectorStateBatch<3> states(x.data(), 2);
  states.SetNumActiveStates(2);
  PriorVectorFactorBatch<3> prior(reinterpret_cast<const Vector<3> *>(t.data()), 2);
  prior.SetNumActiveFactors(2);
  BoundFactorBatch<3> bounds(lo.data(), hi.data(), 2);
  bounds.SetNumActiveFactors(2);
  Problem problem;
  problem.AddStateBatch(&states);
  const std::vector<float *> ptrs = {states.StateDevicePtr(0), states.StateDevicePtr(1)};
  problem.AddFactorBatch(&prior, ptrs);
  problem.AddFactorBatch(&bounds, ptrs);

  auto inner = MakeMinimizer(GetParam());
  AugmentedLagrangianMinimizer solver(*inner);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LE(summary.max_violation, 1e-4f);
  const auto result = Download(x);
  double expected_cost = 0.0;
  for (size_t i = 0; i < 6; ++i) {
    const float expected = std::min(std::max(target[i], lower[i]), upper[i]);
    EXPECT_NEAR(result[i], expected, 2e-4f) << "component " << i;
    expected_cost += 0.5 * (expected - target[i]) * (expected - target[i]);
  }
  EXPECT_NEAR(summary.final_cost, expected_cost, 1e-3 * expected_cost);
}

// min ½‖x - t‖² s.t. aᵀx <= b: x = t - max(0, aᵀt - b) a / ‖a‖².
TEST_P(AugmentedLagrangianMinimizerTest, HalfspaceProjection) {
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

    auto inner = MakeMinimizer(GetParam());
    AugmentedLagrangianMinimizer solver(*inner);
    const auto summary = solver.Minimize(stream.GetStream(), problem);
    EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
    const float excess = std::max(0.f, a[0] * target[0] + a[1] * target[1] - b) / 5.f;
    const auto result = Download(x);
    EXPECT_NEAR(result[0], target[0] - excess * a[0], 2e-4f);
    EXPECT_NEAR(result[1], target[1] - excess * a[1], 2e-4f);
  }
}

// min ½‖x - t‖² s.t. A x = c (2 rows, x in R^4): the KKT solution
// x = t - Aᵀ (A Aᵀ)⁻¹ (A t - c).
TEST_P(AugmentedLagrangianMinimizerTest, AffineEqualityMatchesKkt) {
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

  auto inner = MakeMinimizer(GetParam());
  AugmentedLagrangianMinimizer solver(*inner);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LE(summary.max_violation, 1e-4f);
  const auto result = Download(x);
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
TEST_P(AugmentedLagrangianMinimizerTest, BatchedBoxProjections) {
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
  CudaStream stream;
  dvector<float> x(std::vector<float>(kBatch * 3, 0.f)), t(target), lo(lower), hi(upper);
  VectorStateBatch<3> states(x.data(), kBatch);
  states.SetNumActiveStates(kBatch);
  PriorVectorFactorBatch<3> prior(reinterpret_cast<const Vector<3> *>(t.data()), kBatch);
  prior.SetNumActiveFactors(kBatch);
  BoundFactorBatch<3> bounds(lo.data(), hi.data(), kBatch);
  bounds.SetNumActiveFactors(kBatch);
  std::vector<float *> ptrs;
  std::vector<int> ids_host;
  for (int s = 0; s < kBatch; ++s) {
    ptrs.push_back(states.StateDevicePtr(s));
    ids_host.push_back(s);
  }
  dvector<int> ids(ids_host);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, ptrs);
  problem.AddFactorBatch(&bounds, ptrs);
  problem.SetProblemPartition(kBatch, {ids.data()});

  auto inner = MakeMinimizer(GetParam());
  AugmentedLagrangianMinimizer solver(*inner);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_EQ(summary.num_problems, static_cast<size_t>(kBatch));
  EXPECT_EQ(summary.num_converged, static_cast<size_t>(kBatch));
  // Finished subproblems keep their multipliers, so they stay feasible while
  // the others iterate.
  EXPECT_LE(summary.max_violation, 1e-4f);
  const auto result = Download(x);
  for (int i = 0; i < kBatch * 3; ++i) {
    EXPECT_NEAR(result[i], std::min(std::max(target[i], lower[i]), upper[i]), 2e-4f)
        << "subproblem " << i / 3;
  }
}

// An SE2 chain (prior on the first pose, odometry between poses) whose last
// pose must equal a goal that the odometry does not reach: the constraint
// holds to tolerance, and the solution is the limit of ever stiffer soft
// goal priors.
TEST_P(AugmentedLagrangianMinimizerTest, Se2GoalEquality) {
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
    auto inner = MakeMinimizer(GetParam());
    AugmentedLagrangianMinimizer solver(*inner);
    const auto summary = solver.Minimize(stream.GetStream(), problem);
    const auto all = Download(poses);
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

// Without constraint batches the solve is the wrapped minimizer's.
TEST_P(AugmentedLagrangianMinimizerTest, NoConstraintsIsPlainSolve) {
  CudaStream stream;
  dvector<float> x(std::vector<float>(3, 0.f)), t(std::vector<float>{1.f, 2.f, 3.f});
  VectorStateBatch<3> states(x.data(), 1);
  states.SetNumActiveStates(1);
  PriorVectorFactorBatch<3> prior(reinterpret_cast<const Vector<3> *>(t.data()), 1);
  prior.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
  auto inner = MakeMinimizer(GetParam());
  AugmentedLagrangianMinimizer solver(*inner);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_EQ(summary.outer_iterations, 1u);
  const auto result = Download(x);
  EXPECT_NEAR(result[0], 1.f, 1e-5f);
  EXPECT_NEAR(result[2], 3.f, 1e-5f);
}

// Contradicting bounds (lower > upper): no feasible point. The penalty
// reaches its cap and the solve ends with kMaxPenalty, not a loop.
TEST_P(AugmentedLagrangianMinimizerTest, InfeasibleStopsAtMaxPenalty) {
  CudaStream stream;
  dvector<float> x(std::vector<float>{0.f}), t(std::vector<float>{0.f});
  dvector<float> lo(std::vector<float>{1.f}), hi(std::vector<float>{-1.f});
  VectorStateBatch<1> states(x.data(), 1);
  states.SetNumActiveStates(1);
  PriorVectorFactorBatch<1> prior(reinterpret_cast<const Vector<1> *>(t.data()), 1);
  prior.SetNumActiveFactors(1);
  BoundFactorBatch<1> bounds(lo.data(), hi.data(), 1);
  bounds.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
  problem.AddFactorBatch(&bounds, {states.StateDevicePtr(0)});
  auto inner = MakeMinimizer(GetParam());
  AugmentedLagrangianMinimizerOptions options;
  options.max_penalty = 1e4f;
  AugmentedLagrangianMinimizer solver(*inner, options);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kMaxPenalty);
  EXPECT_EQ(summary.num_max_penalty, 1u);
  EXPECT_LT(summary.outer_iterations, options.max_outer_iterations);
  EXPECT_NEAR(summary.max_violation, 1.f, 1e-2f);  // x = 0 is the least-violating point
}

// Re-solving with warm_start keeps the multipliers and penalties: the second
// solve of the same problem from a perturbed start needs fewer outer
// iterations and reaches the same solution.
TEST_P(AugmentedLagrangianMinimizerTest, WarmStartReusesMultipliers) {
  const std::vector<float> target = {2.f, -3.f, 0.5f};
  CudaStream stream;
  dvector<float> x(std::vector<float>(3, 0.f)), t(target);
  dvector<float> lo(std::vector<float>(3, -1.f)), hi(std::vector<float>(3, 1.f));
  VectorStateBatch<3> states(x.data(), 1);
  states.SetNumActiveStates(1);
  PriorVectorFactorBatch<3> prior(reinterpret_cast<const Vector<3> *>(t.data()), 1);
  prior.SetNumActiveFactors(1);
  BoundFactorBatch<3> bounds(lo.data(), hi.data(), 1);
  bounds.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0)});
  problem.AddFactorBatch(&bounds, {states.StateDevicePtr(0)});
  auto inner = MakeMinimizer(GetParam());
  AugmentedLagrangianMinimizerOptions options;
  options.warm_start = true;
  AugmentedLagrangianMinimizer solver(*inner, options);
  const auto cold = solver.Minimize(stream.GetStream(), problem);
  ASSERT_EQ(cold.status, AugmentedLagrangianMinimizerStatus::kConverged);
  const std::vector<float> perturbed = {0.9f, -0.8f, 0.3f};
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(x.data(), perturbed.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
  const auto warm = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(warm.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LT(warm.outer_iterations, cold.outer_iterations);
  const auto result = Download(x);
  EXPECT_NEAR(result[0], 1.f, 2e-4f);
  EXPECT_NEAR(result[1], -1.f, 2e-4f);
  EXPECT_NEAR(result[2], 0.5f, 2e-4f);
}

// Real time: a fixed budget without per-iteration read-backs reaches the box
// projection; a warm follow-up call keeps it.
TEST_P(AugmentedLagrangianMinimizerTest, RealTimeBudget) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::vector<float> target = {2.f, -3.f, 0.5f, 7.f, -4.f, 0.25f};
  const std::vector<float> lower = {-1.f, -1.f, -1.f, -kInf, -2.f, 0.5f};
  const std::vector<float> upper = {1.f, 1.f, 1.f, 3.f, kInf, kInf};
  CudaStream stream;
  dvector<float> x(std::vector<float>(6, 0.f)), t(target), lo(lower), hi(upper);
  VectorStateBatch<3> states(x.data(), 2);
  states.SetNumActiveStates(2);
  PriorVectorFactorBatch<3> prior(reinterpret_cast<const Vector<3> *>(t.data()), 2);
  prior.SetNumActiveFactors(2);
  BoundFactorBatch<3> bounds(lo.data(), hi.data(), 2);
  bounds.SetNumActiveFactors(2);
  Problem problem;
  problem.AddStateBatch(&states);
  const std::vector<float *> ptrs = {states.StateDevicePtr(0), states.StateDevicePtr(1)};
  problem.AddFactorBatch(&prior, ptrs);
  problem.AddFactorBatch(&bounds, ptrs);
  auto inner = MakeMinimizer(GetParam());
  AugmentedLagrangianMinimizerOptions options;
  options.real_time = true;
  options.max_outer_iterations = 12;
  options.inner_iterations = 3;
  options.warm_start = true;
  options.reuse_structure = true;
  AugmentedLagrangianMinimizer solver(*inner, options);
  for (int call = 0; call < 2; ++call) {
    const auto summary = solver.Minimize(stream.GetStream(), problem);
    EXPECT_EQ(summary.outer_iterations, 12u);
    EXPECT_TRUE(std::isnan(summary.final_cost));
    EXPECT_LE(summary.max_violation, 1e-3f) << "call " << call;
    const auto result = Download(x);
    for (size_t i = 0; i < 6; ++i) {
      const float expected = std::min(std::max(target[i], lower[i]), upper[i]);
      EXPECT_NEAR(result[i], expected, 1e-3f) << "call " << call << " component " << i;
    }
  }
}

TEST(AugmentedLagrangianMinimizerOptions, RejectsInvalidOptions) {
  auto inner = MakeMinimizer(Kind::kGaussNewton);
  AugmentedLagrangianMinimizerOptions options;
  options.max_outer_iterations = 0;
  EXPECT_THROW(AugmentedLagrangianMinimizer(*inner, options), std::invalid_argument);
  options = AugmentedLagrangianMinimizerOptions();
  options.inner_iterations = 0;
  EXPECT_THROW(AugmentedLagrangianMinimizer(*inner, options), std::invalid_argument);
  AugmentedLagrangianMinimizer solver(*inner);
  options = AugmentedLagrangianMinimizerOptions();
  options.max_outer_iterations = 0;
  EXPECT_THROW(solver.SetOptions(options), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(Minimizers, AugmentedLagrangianMinimizerTest,
                         ::testing::Values(Kind::kGaussNewton, Kind::kLevenbergMarquardt));

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
    const auto r = Download(res), j = Download(jac);
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
