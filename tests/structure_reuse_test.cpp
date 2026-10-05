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

// Structure reuse (AugmentedLagrangianMinimizerOptions::reuse_structure, which
// the bounded problem needs; it reaches the inner minimizer through
// internal::InnerSolve): a solve that reuses the structure of the previous
// call gives the same result as a fresh minimizer, for a single problem and a
// partitioned one, with Gauss-Newton and Levenberg-Marquardt; a size change
// falls back to the full setup. The real-time mode (fixed iterations, device
// step control) reaches the same solution.

#include <gtest/gtest.h>

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
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

enum class Kind { kGaussNewton, kLevenbergMarquardt };

std::unique_ptr<Minimizer> MakeMinimizer(Kind kind) {
  MinimizerOptions options;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
  if (kind == Kind::kGaussNewton) return std::make_unique<GaussNewtonMinimizer>(options);
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = options;
  return std::make_unique<LevenbergMarquardtMinimizer>(lm);
}

/** Chains of n 2-D points pulled to targets and smoothed, with bounds. */
struct Chains {
  static constexpr int kN = 6;
  int num_chains;
  int total;
  dvector<float> x, targets, lower, upper, zeros;
  dvector<int> ids;
  std::unique_ptr<VectorStateBatch<2>> states;
  std::unique_ptr<PriorVectorFactorBatch<2>> prior;
  std::unique_ptr<WeightedFactorBatch<VectorBetweenFactorBatch<2>>> smooth;
  Problem problem;

  explicit Chains(int chains) : num_chains(chains), total(chains * kN) {
    x = dvector<float>(std::vector<float>(2 * total, 0.f));
    targets = dvector<float>(std::vector<float>(2 * total, 0.f));
    lower = dvector<float>(std::vector<float>(2 * total, -1.f));
    upper = dvector<float>(std::vector<float>(2 * total, 1.f));
    zeros = dvector<float>(std::vector<float>(2 * total, 0.f));
    states = std::make_unique<VectorStateBatch<2>>(x.data(), total);
    states->SetNumActiveStates(total);
    states->SetBounds(lower.data(), upper.data());
    prior = std::make_unique<PriorVectorFactorBatch<2>>(
        reinterpret_cast<const Vector<2> *>(targets.data()), total);
    prior->SetNumActiveFactors(total);
    const int edges = chains * (kN - 1);
    smooth = std::make_unique<WeightedFactorBatch<VectorBetweenFactorBatch<2>>>(
        1.5f, reinterpret_cast<const Vector<2> *>(zeros.data()), static_cast<size_t>(edges));
    smooth->SetNumActiveFactors(edges);
    problem.AddStateBatch(states.get());
    std::vector<float *> prior_ptrs, edge_ptrs;
    for (int i = 0; i < total; ++i) prior_ptrs.push_back(states->StateDevicePtr(i));
    for (int c = 0; c < chains; ++c) {
      for (int k = 0; k + 1 < kN; ++k) {
        edge_ptrs.push_back(states->StateDevicePtr(c * kN + k));
        edge_ptrs.push_back(states->StateDevicePtr(c * kN + k + 1));
      }
    }
    problem.AddFactorBatch(prior.get(), prior_ptrs);
    problem.AddFactorBatch(smooth.get(), edge_ptrs);
    std::vector<int> h_ids(total);
    for (int i = 0; i < total; ++i) h_ids[i] = i / kN;
    ids = dvector<int>(h_ids);
    if (chains > 1) problem.SetProblemPartition(chains, {ids.data()});
  }

  void Reset(unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.f, 2.f);
    std::vector<float> t(2 * total);
    for (float &v : t) v = normal(rng);
    targets.CopyFromHost(t.data(), t.size());
    std::vector<float> zero(2 * total, 0.f);
    x.CopyFromHost(zero.data(), zero.size());
  }

  /** One solve through the augmented Lagrangian (the states are bounded; no constraint rows). */
  std::vector<float> Solve(Minimizer &minimizer, bool reuse, bool fixed = false,
                           size_t iterations = 50) {
    CudaStream stream;
    AugmentedLagrangianMinimizerOptions options;
    options.inner_line_search_steps = fixed ? 3 : 10;
    options.reuse_structure = reuse;
    options.real_time = fixed;
    options.inner_iterations = iterations;
    options.final_inner_iterations = iterations;
    AugmentedLagrangianMinimizer(minimizer, options).Minimize(stream.GetStream(), problem);
    std::vector<float> out(2 * total);
    x.CopyToHost(out.data(), out.size());
    return out;
  }
};

class StructureReuseTest : public ::testing::TestWithParam<std::tuple<Kind, int>> {};

TEST_P(StructureReuseTest, MatchesFreshSolves) {
  const Kind kind = std::get<0>(GetParam());
  Chains chains(std::get<1>(GetParam()));
  // Levenberg-Marquardt's accept/reject decisions compare float32 costs summed
  // with atomics; near the minimum they can go either way, so two identical
  // solves agree only to about 1e-3.
  const float tolerance = kind == Kind::kGaussNewton ? 1e-5f : 1e-3f;
  auto reused = MakeMinimizer(kind);
  for (unsigned seed = 1; seed <= 3; ++seed) {
    chains.Reset(seed);
    const auto with_reuse = chains.Solve(*reused, /*reuse=*/seed > 1);
    chains.Reset(seed);
    auto fresh = MakeMinimizer(kind);
    const auto expected = chains.Solve(*fresh, /*reuse=*/false);
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(with_reuse[i], expected[i], tolerance) << "seed " << seed << " component " << i;
    }
  }
  // Fewer active priors at the same capacities: the size check rejects the
  // reuse and the call sets the structure up again.
  chains.prior->SetNumActiveFactors(chains.total - 2);
  chains.Reset(9);
  const auto after_resize = chains.Solve(*reused, /*reuse=*/true);
  chains.Reset(9);
  auto fresh = MakeMinimizer(kind);
  const auto expected = chains.Solve(*fresh, /*reuse=*/false);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(after_resize[i], expected[i], tolerance) << "component " << i;
  }
}

TEST_P(StructureReuseTest, FixedIterationsReachTheSameSolution) {
  const Kind kind = std::get<0>(GetParam());
  Chains chains(std::get<1>(GetParam()));
  const float tolerance = kind == Kind::kGaussNewton ? 1e-4f : 1e-3f;
  chains.Reset(4);
  auto converged = MakeMinimizer(kind);
  const auto expected = chains.Solve(*converged, /*reuse=*/false);
  // As a receding horizon uses it: a converged call, then fixed-iteration
  // calls reusing its structure (the partition is built on the first of them).
  auto real_time = MakeMinimizer(kind);
  chains.Reset(4);
  chains.Solve(*real_time, /*reuse=*/false);
  for (int call = 1; call <= 2; ++call) {
    chains.Reset(4);
    const auto fixed = chains.Solve(*real_time, /*reuse=*/true, /*fixed=*/true, 30);
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(fixed[i], expected[i], tolerance) << "call " << call << " component " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Minimizers, StructureReuseTest,
                         ::testing::Combine(::testing::Values(Kind::kGaussNewton,
                                                              Kind::kLevenbergMarquardt),
                                            ::testing::Values(1, 3)));

}  // namespace
}  // namespace cunls
