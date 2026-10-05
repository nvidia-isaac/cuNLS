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

// Problem::SetProblemPartition: a batch of independent SE2 pose graphs solved
// as one problem with per-subproblem step control gives every graph the
// solution it gets when solved alone (Gauss-Newton and Levenberg-Marquardt);
// invalid partitions are rejected.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

using Mat3 = std::array<double, 9>;

Mat3 Pose(double x, double y, double theta) {
  const double c = std::cos(theta), s = std::sin(theta);
  return {c, -s, x, s, c, y, 0, 0, 1};
}

Mat3 Mul(const Mat3 &a, const Mat3 &b) {
  Mat3 m{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) m[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return m;
}

Mat3 Inv(const Mat3 &t) {
  // [R p; 0 1]^-1 = [R^T, -R^T p; 0 1]
  return {t[0], t[3], -(t[0] * t[2] + t[3] * t[5]), t[1], t[4], -(t[1] * t[2] + t[4] * t[5]), 0,
          0,    1};
}

/** B independent pose graphs (chain plus two loop closures), n poses each. */
struct Graphs {
  int batch = 0, n = 0;
  std::vector<std::pair<int, int>> edges;  // within one graph
  std::vector<Mat3> init, deltas, priors;  // [batch * n], [batch * edges], [batch]

  Graphs(int b, int num_poses, uint32_t seed) : batch(b), n(num_poses) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i + 1 < n; ++i) edges.push_back({i, i + 1});
    edges.push_back({0, n / 2});
    edges.push_back({1, n - 1});
    for (int s = 0; s < batch; ++s) {
      std::vector<Mat3> gt;
      Mat3 t = Pose(normal(rng), normal(rng), normal(rng));
      for (int i = 0; i < n; ++i) {
        gt.push_back(t);
        t = Mul(t, Pose(1.0 + 0.3 * normal(rng), 0.3 * normal(rng), 0.4 * normal(rng)));
      }
      // Subproblem s starts further off the larger s is: different paths.
      const double off = 0.05 + 0.1 * s;
      for (int i = 0; i < n; ++i) {
        init.push_back(Mul(gt[i], Pose(off * normal(rng), off * normal(rng), off * normal(rng))));
      }
      for (auto [a, b2] : edges) {
        // SE2BetweenFactorBatch: r = Log(Delta * L^-1 * R) = 0 for Delta = R^-1 * L.
        const Mat3 noise = Pose(0.02 * normal(rng), 0.02 * normal(rng), 0.02 * normal(rng));
        deltas.push_back(Mul(Mul(Inv(gt[b2]), gt[a]), noise));
      }
      priors.push_back(gt[0]);
    }
  }
};

std::vector<float> ToFloat(const std::vector<Mat3> &m, size_t first, size_t count) {
  std::vector<float> out;
  for (size_t i = first; i < first + count; ++i) out.insert(out.end(), m[i].begin(), m[i].end());
  return out;
}

enum class Kind { kGaussNewton, kLevenbergMarquardt };

std::unique_ptr<Minimizer> MakeMinimizer(Kind kind) {
  MinimizerOptions options;
  options.max_num_iterations = 50;
  options.state_tolerance = 1e-12f;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
  if (kind == Kind::kGaussNewton) return std::make_unique<GaussNewtonMinimizer>(options);
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = options;
  lm.relative_reduction_tolerance = 1e-12f;  // run to convergence: compare solutions
  return std::make_unique<LevenbergMarquardtMinimizer>(lm);
}

/**
 * Solves graphs [first, first + count) as one problem (with a partition when
 * `partition`); returns the optimized poses, 9 floats each.
 */
std::vector<float> Solve(const Graphs &g, int first, int count, bool partition, Kind kind,
                         const std::vector<int> *ids_override = nullptr) {
  const int n = g.n, m = static_cast<int>(g.edges.size());
  CudaStream stream;
  dvector<float> poses(ToFloat(g.init, size_t(first) * n, size_t(count) * n));
  dvector<float> deltas(ToFloat(g.deltas, size_t(first) * m, size_t(count) * m));
  dvector<float> priors(ToFloat(g.priors, first, count));
  SE2StateBatch states(poses.data(), count * n);
  states.SetNumActiveStates(count * n);
  SE2PriorFactorBatch prior(reinterpret_cast<const SE2Transform *>(priors.data()), count);
  prior.SetNumActiveFactors(count);
  SE2BetweenFactorBatch between(reinterpret_cast<const SE2Transform *>(deltas.data()), count * m);
  between.SetNumActiveFactors(count * m);
  std::vector<float *> prior_ptrs, between_ptrs;
  for (int s = 0; s < count; ++s) {
    prior_ptrs.push_back(states.StateDevicePtr(s * n));
    for (auto [a, b] : g.edges) {
      between_ptrs.push_back(states.StateDevicePtr(s * n + a));
      between_ptrs.push_back(states.StateDevicePtr(s * n + b));
    }
  }
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, prior_ptrs);
  problem.AddFactorBatch(&between, between_ptrs);
  std::vector<int> ids_host;
  for (int s = 0; s < count; ++s) ids_host.insert(ids_host.end(), n, s);
  if (ids_override != nullptr) ids_host = *ids_override;
  dvector<int> ids(ids_host);
  if (partition) problem.SetProblemPartition(count, {ids.data()});
  auto minimizer = MakeMinimizer(kind);
  minimizer->Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> out(poses.size());
  poses.CopyToHost(out.data(), out.size());
  return out;
}

class ProblemPartitionTest : public ::testing::TestWithParam<Kind> {};

TEST_P(ProblemPartitionTest, BatchMatchesIndividualSolves) {
  const Graphs g(8, 12, 3);
  const auto batched = Solve(g, 0, g.batch, /*partition=*/true, GetParam());
  for (int s = 0; s < g.batch; ++s) {
    const auto alone = Solve(g, s, 1, /*partition=*/false, GetParam());
    float worst = 0.f, size = 0.f;
    for (size_t k = 0; k < alone.size(); ++k) {
      worst = std::max(worst, std::fabs(batched[s * g.n * 9 + k] - alone[k]));
      size = std::max(size, std::fabs(alone[k]));
    }
    // float32: near the minimum, poses ~1e-4 (relative) apart have equal cost,
    // and two standalone solves of the same graph already differ by ~1e-5.
    EXPECT_LT(worst, 5e-4f * std::max(1.f, size)) << "subproblem " << s;
  }
}

INSTANTIATE_TEST_SUITE_P(Minimizers, ProblemPartitionTest,
                         ::testing::Values(Kind::kGaussNewton, Kind::kLevenbergMarquardt));

TEST(ProblemPartition, RejectsFactorsAcrossSubproblems) {
  const Graphs g(2, 6, 4);
  // Poses of graph 1 labelled as subproblem 0 except one: a between factor of
  // graph 1 then connects two subproblems.
  std::vector<int> ids(2 * g.n, 0);
  for (int i = g.n; i < 2 * g.n; ++i) ids[i] = 1;
  ids[g.n + 2] = 0;
  EXPECT_THROW(Solve(g, 0, 2, true, Kind::kGaussNewton, &ids), std::invalid_argument);
  std::vector<int> out_of_range(2 * g.n, 0);
  out_of_range[0] = 7;
  EXPECT_THROW(Solve(g, 0, 2, true, Kind::kGaussNewton, &out_of_range), std::invalid_argument);
}

TEST(ProblemPartition, SetterValidatesArguments) {
  Problem problem;
  dvector<float> poses(9);
  SE2StateBatch states(poses.data(), 1);
  problem.AddStateBatch(&states);
  dvector<int> ids(std::vector<int>{0});
  EXPECT_THROW(problem.SetProblemPartition(2, {}), std::invalid_argument);
  EXPECT_THROW(problem.SetProblemPartition(2, {nullptr}), std::invalid_argument);
  problem.SetProblemPartition(2, {ids.data()});
  EXPECT_EQ(problem.NumProblems(), 2u);
  problem.SetProblemPartition(0, {});
  EXPECT_EQ(problem.NumProblems(), 1u);
}

// A state batch added after SetProblemPartition leaves one batch without an
// id array: the solve rejects the partition (it would otherwise run as one
// subproblem).
TEST(ProblemPartition, RejectsMissingIdArrays) {
  const Graphs g(2, 4, 5);
  dvector<float> poses(ToFloat(g.init, 0, 2 * g.n)), extra(std::vector<float>(1, 0.f));
  SE2StateBatch states(poses.data(), 2 * g.n);
  states.SetNumActiveStates(2 * g.n);
  dvector<float> priors(ToFloat(g.priors, 0, 2));
  SE2PriorFactorBatch prior(reinterpret_cast<const SE2Transform *>(priors.data()), 2);
  prior.SetNumActiveFactors(2);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, {states.StateDevicePtr(0), states.StateDevicePtr(g.n)});
  std::vector<int> ids_host(2 * g.n);
  for (int i = 0; i < 2 * g.n; ++i) ids_host[i] = i / g.n;
  dvector<int> ids(ids_host);
  problem.SetProblemPartition(2, {ids.data()});
  VectorStateBatch<1> late(extra.data(), 1);
  late.SetNumActiveStates(1, 0);
  problem.AddStateBatch(&late);
  CudaStream stream;
  GaussNewtonMinimizer minimizer;
  try {
    minimizer.Minimize(stream.GetStream(), problem);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("expected 2, got 1"), std::string::npos) << e.what();
  }
}

}  // namespace

// InnerSolve::problem_frozen (AugmentedLagrangianMinimizer's finished
// subproblems): a frozen subproblem keeps its states exactly, the others are
// solved (Gauss-Newton and Levenberg-Marquardt).
TEST(ProblemPartition, FrozenSubproblemsKeepTheirStates) {
  for (bool lm : {false, true}) {
    CudaStream stream;
    // Two subproblems of two states each, priors pulling every state to 1.
    const std::vector<float> start = {0.f, 0.5f, -2.f, 3.f};
    dvector<float> x(start), targets(std::vector<float>(4, 1.f));
    dvector<int> ids(std::vector<int>{0, 0, 1, 1}), frozen(std::vector<int>{1, 0});
    VectorStateBatch<1> states(x.data(), 4);
    states.SetNumActiveStates(4);
    PriorVectorFactorBatch<1> priors(reinterpret_cast<const Vector<1> *>(targets.data()), 4);
    priors.SetNumActiveFactors(4);
    Problem problem;
    problem.AddStateBatch(&states);
    std::vector<float *> ptrs;
    for (int i = 0; i < 4; ++i) ptrs.push_back(states.StateDevicePtr(i));
    problem.AddFactorBatch(&priors, ptrs);
    problem.SetProblemPartition(2, {ids.data()});
    MinimizerOptions options;
    options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
    internal::InnerSolve settings;
    settings.max_num_iterations = options.max_num_iterations;
    settings.problem_frozen = frozen.data();
    if (lm) {
      LevenbergMarquardtMinimizerOptions lm_options;
      lm_options.base_options = options;
      LevenbergMarquardtMinimizer minimizer(lm_options);
      internal::InnerMinimize(minimizer, stream.GetStream(), problem, settings);
    } else {
      GaussNewtonMinimizer minimizer(options);
      internal::InnerMinimize(minimizer, stream.GetStream(), problem, settings);
    }
    std::vector<float> out(4);
    x.CopyToHost(out.data(), 4);
    EXPECT_EQ(out[0], start[0]) << "lm " << lm;  // frozen: bit-identical
    EXPECT_EQ(out[1], start[1]) << "lm " << lm;
    EXPECT_NEAR(out[2], 1.f, 1e-3f) << "lm " << lm;
    EXPECT_NEAR(out[3], 1.f, 1e-3f) << "lm " << lm;
  }
}

}  // namespace cunls
