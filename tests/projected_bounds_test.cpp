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
// projection in the Gauss-Newton and Levenberg-Marquardt minimizers: coupled
// box-constrained least squares against a float64 reference (single problem
// and a batch of subproblems), a constant state outside its bounds stays as
// it is, and an infeasible initial guess is projected.

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
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

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

class ProjectedBoundsTest : public ::testing::TestWithParam<std::tuple<Kind, int>> {};

TEST_P(ProjectedBoundsTest, CoupledChainsMatchReference) {
  const Kind kind = std::get<0>(GetParam());
  const int num_chains = std::get<1>(GetParam());
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
  ASSERT_TRUE(states.HasBounds());

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

  auto minimizer = MakeMinimizer(kind);
  MinimizeCallOptions call;
  call.max_num_iterations = 50;
  call.max_line_search_steps = 10;
  minimizer->Minimize(stream.GetStream(), problem, call);

  std::vector<float> result(total);
  x.CopyToHost(result.data(), total);
  const auto expected = Reference(num_chains, n, w, t, lo, hi, x0, constant);
  // Levenberg-Marquardt accepts steps by comparing float32 costs; near the
  // minimum the relative decrease reaches the float32 round-off and it stops a
  // little earlier than Gauss-Newton's exact Newton step.
  const double tolerance = kind == Kind::kGaussNewton ? 2e-4 : 1e-3;
  for (int i = 0; i < total; ++i) {
    EXPECT_GE(result[i], lo[i]) << "component " << i;
    if (!constant[i]) EXPECT_LE(result[i], hi[i]) << "component " << i;
    EXPECT_NEAR(result[i], expected[i], tolerance) << "component " << i;
  }
  EXPECT_EQ(result[0], 3.f);  // constant, outside the box, untouched
}

INSTANTIATE_TEST_SUITE_P(Minimizers, ProjectedBoundsTest,
                         ::testing::Combine(::testing::Values(Kind::kGaussNewton,
                                                              Kind::kLevenbergMarquardt),
                                            ::testing::Values(1, 3)));

TEST(ProjectedBounds, SetBoundsNeedsBothOrNeither) {
  dvector<float> x(std::vector<float>(2, 0.f)), lo(std::vector<float>(2, -1.f));
  VectorStateBatch<2> states(x.data(), 1);
  EXPECT_FALSE(states.HasBounds());
  EXPECT_THROW(states.SetBounds(lo.data(), nullptr), std::invalid_argument);
  states.SetBounds(lo.data(), lo.data());
  EXPECT_TRUE(states.HasBounds());
  states.SetBounds(nullptr, nullptr);
  EXPECT_FALSE(states.HasBounds());
}

}  // namespace
}  // namespace cunls
