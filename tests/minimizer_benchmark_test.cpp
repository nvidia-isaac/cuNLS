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

// Minimize() wall time on representative problems, Gauss-Newton and
// Levenberg-Marquardt: a single small pose (PnP), a single small and a large
// pose graph, a batch of small pose graphs (one subproblem each) and
// real-time augmented Lagrangian MPC calls. Disabled by default; run with
//   nls_tests --gtest_also_run_disabled_tests --gtest_filter='*MinimizerBenchmark*'
// Every case prints one line:
//   BENCH <case> <minimizer> median_ms=... min_ms=... iterations=... final_cost=...

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/dynamics/se2_differential_drive_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

using Mat3 = std::array<double, 9>;
using Mat4 = std::array<double, 16>;

Mat3 Pose2(double x, double y, double theta) {
  const double c = std::cos(theta), s = std::sin(theta);
  return {c, -s, x, s, c, y, 0, 0, 1};
}

template <size_t N, size_t D>
std::array<double, N> Mul(const std::array<double, N> &a, const std::array<double, N> &b) {
  std::array<double, N> m{};
  for (size_t i = 0; i < D; ++i)
    for (size_t j = 0; j < D; ++j)
      for (size_t k = 0; k < D; ++k) m[i * D + j] += a[i * D + k] * b[k * D + j];
  return m;
}

/** Rigid inverse of a homogeneous D x D transform. */
template <size_t N, size_t D>
std::array<double, N> Inv(const std::array<double, N> &t) {
  std::array<double, N> m{};
  for (size_t i = 0; i + 1 < D; ++i) {
    for (size_t j = 0; j + 1 < D; ++j) m[i * D + j] = t[j * D + i];
    for (size_t k = 0; k + 1 < D; ++k) m[i * D + D - 1] -= t[k * D + i] * t[k * D + D - 1];
  }
  m[N - 1] = 1;
  return m;
}

/** SE(3) from a rotation vector (Rodrigues) and a translation. */
Mat4 Pose3(const std::array<double, 3> &w, const std::array<double, 3> &t) {
  const double th = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  const double a = th < 1e-12 ? 1.0 : std::sin(th) / th;
  const double b = th < 1e-12 ? 0.5 : (1 - std::cos(th)) / (th * th);
  const double K[9] = {0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0};
  double K2[9] = {};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) K2[i * 3 + j] += K[i * 3 + k] * K[k * 3 + j];
  Mat4 m{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) m[i * 4 + j] = (i == j) + a * K[i * 3 + j] + b * K2[i * 3 + j];
    m[i * 4 + 3] = t[i];
  }
  m[15] = 1;
  return m;
}

template <class T>
std::vector<float> ToFloat(const std::vector<T> &m) {
  std::vector<float> out;
  for (const auto &x : m) out.insert(out.end(), x.begin(), x.end());
  return out;
}

struct Timing {
  double median_ms = 0, min_ms = 0;
  size_t iterations = 0;
  float final_cost = 0;
};

/** Times `solve` (which returns the summary) `reps` times after `reset`, one warm-up. */
Timing Time(int reps, const std::function<void()> &reset,
            const std::function<MinimizerSummary()> &solve) {
  std::vector<double> ms;
  MinimizerSummary summary;
  for (int r = 0; r <= reps; ++r) {
    reset();
    THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    summary = solve();
    THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    if (r > 0) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  if (std::getenv("BENCH_VERBOSE") != nullptr) {
    std::printf("  iteration_costs:");
    for (float c : summary.iteration_costs) std::printf(" %.6g", c);
    std::printf(" | initial %.6g final %.6g\n", summary.initial_cost, summary.final_cost);
  }
  std::sort(ms.begin(), ms.end());
  return {ms[ms.size() / 2], ms.front(), summary.num_iterations, summary.final_cost};
}

void Print(const std::string &name, const char *minimizer, const Timing &t) {
  std::printf("BENCH %-28s %-3s median_ms=%9.3f min_ms=%9.3f iterations=%3zu final_cost=%.6g\n",
              name.c_str(), minimizer, t.median_ms, t.min_ms, t.iterations, t.final_cost);
  std::fflush(stdout);
}

MinimizerOptions Options(SparseLinearSolverType solver, size_t iterations) {
  MinimizerOptions o;
  o.max_num_iterations = iterations;
  o.state_tolerance = 1e-10f;
  o.cost_tolerance = 1e-10f;
  o.sparse_linear_solver_type = solver;
  return o;
}

/** Runs the problem with GN and with LM (fresh minimizers), resetting the states each time. */
void RunBoth(const std::string &name, Problem &problem, SparseLinearSolverType solver,
             size_t iterations, int reps, const std::function<void()> &reset) {
  CudaStream stream;
  {
    GaussNewtonMinimizer gn(Options(solver, iterations));
    Print(name, "GN", Time(reps, reset, [&] { return gn.Minimize(stream.GetStream(), problem); }));
  }
  {
    LevenbergMarquardtMinimizerOptions lm;
    lm.base_options = Options(solver, iterations);
    lm.relative_reduction_tolerance = 1e-10f;
    LevenbergMarquardtMinimizer minimizer(lm);
    Print(name, "LM",
          Time(reps, reset, [&] { return minimizer.Minimize(stream.GetStream(), problem); }));
  }
}

/** B independent SE(2) pose graphs: a chain of n poses plus `closures` loop closures each. */
void SE2Graphs(const std::string &name, int batch, int n, int closures,
               SparseLinearSolverType solver, int reps) {
  std::mt19937 rng(3);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_int_distribution<int> pick(0, n - 1);
  std::vector<std::pair<int, int>> edges;
  for (int i = 0; i + 1 < n; ++i) edges.push_back({i, i + 1});
  while (static_cast<int>(edges.size()) < n - 1 + closures) {
    const int a = pick(rng), b = pick(rng);
    if (std::abs(a - b) > 1) edges.push_back({a, b});
  }
  std::vector<Mat3> init, deltas, priors;
  for (int s = 0; s < batch; ++s) {
    std::vector<Mat3> gt;
    Mat3 t = Pose2(normal(rng), normal(rng), normal(rng));
    for (int i = 0; i < n; ++i) {
      gt.push_back(t);
      t = Mul<9, 3>(t, Pose2(1.0 + 0.3 * normal(rng), 0.3 * normal(rng), 0.2 * normal(rng)));
    }
    for (int i = 0; i < n; ++i) {
      init.push_back(
          Mul<9, 3>(gt[i], Pose2(0.1 * normal(rng), 0.1 * normal(rng), 0.1 * normal(rng))));
    }
    for (auto [a, b] : edges) {
      const Mat3 noise = Pose2(0.02 * normal(rng), 0.02 * normal(rng), 0.02 * normal(rng));
      deltas.push_back(Mul<9, 3>(Mul<9, 3>(Inv<9, 3>(gt[b]), gt[a]), noise));
    }
    priors.push_back(gt[0]);
  }
  const int m = static_cast<int>(edges.size());
  const std::vector<float> init_f = ToFloat(init);
  dvector<float> init_d(init_f), poses(init_f);
  dvector<float> deltas_d(ToFloat(deltas)), priors_d(ToFloat(priors));
  SE2StateBatch states(poses.data(), batch * n);
  states.SetNumActiveStates(batch * n);
  SE2PriorFactorBatch prior(reinterpret_cast<const SE2Transform *>(priors_d.data()), batch);
  prior.SetNumActiveFactors(batch);
  SE2BetweenFactorBatch between(reinterpret_cast<const SE2Transform *>(deltas_d.data()), batch * m);
  between.SetNumActiveFactors(batch * m);
  std::vector<float *> prior_ptrs, between_ptrs;
  for (int s = 0; s < batch; ++s) {
    prior_ptrs.push_back(states.StateDevicePtr(s * n));
    for (auto [a, b] : edges) {
      between_ptrs.push_back(states.StateDevicePtr(s * n + a));
      between_ptrs.push_back(states.StateDevicePtr(s * n + b));
    }
  }
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&prior, prior_ptrs);
  problem.AddFactorBatch(&between, between_ptrs);
  std::vector<int> ids_host;
  for (int s = 0; s < batch; ++s) ids_host.insert(ids_host.end(), n, s);
  dvector<int> ids(ids_host);
  if (batch > 1) problem.SetProblemPartition(batch, {ids.data()});
  RunBoth(name, problem, solver, 30, reps, [&] {
    THROW_ON_CUDA_ERROR(cudaMemcpy(poses.data(), init_d.data(), init_f.size() * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
  });
}

TEST(MinimizerBenchmark, DISABLED_PnPSinglePose) {
  constexpr int kPoints = 200;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> xy(-2.f, 2.f), z(4.f, 8.f);
  std::normal_distribution<float> noise(0.f, 1e-3f);
  std::vector<Vector<3>> points(kPoints);
  std::vector<Vector<2>> obs(kPoints);
  for (int i = 0; i < kPoints; ++i) {
    points[i] = {xy(rng), xy(rng), z(rng)};
    obs[i] = {points[i][0] / points[i][2] + noise(rng), points[i][1] / points[i][2] + noise(rng)};
  }
  const Mat4 start = Pose3({0.05, -0.04, 0.08}, {0.2, -0.1, 0.3});
  std::vector<float> init_f(start.begin(), start.end());
  dvector<float> init_d(init_f), pose(init_f);
  dvector<Vector<3>> points_d(points);
  dvector<Vector<2>> obs_d(obs);
  PnPFactorBatch pnp(obs_d.data(), points_d.data(), kPoints, 1e-3f);
  pnp.SetNumActiveFactors(kPoints);
  SE3StateBatch state(pose.data(), 1);
  state.SetNumActiveStates(1);
  Problem problem;
  problem.AddStateBatch(&state);
  problem.AddFactorBatch(&pnp, std::vector<float *>(kPoints, state.StateDevicePtr(0)));
  RunBoth("pnp_single_200", problem, SparseLinearSolverType::DenseLDLT, 30, 200, [&] {
    THROW_ON_CUDA_ERROR(
        cudaMemcpy(pose.data(), init_d.data(), 16 * sizeof(float), cudaMemcpyDeviceToDevice));
  });
}

TEST(MinimizerBenchmark, DISABLED_SE2GraphSingleSmall) {
  SE2Graphs("se2_graph_1x12", 1, 12, 2, SparseLinearSolverType::DenseLDLT, 200);
}

TEST(MinimizerBenchmark, DISABLED_SE2GraphSingleLarge) {
  SE2Graphs("se2_graph_1x5000", 1, 5000, 1000, SparseLinearSolverType::cuDSS, 20);
}

TEST(MinimizerBenchmark, DISABLED_SE2GraphsBatched) {
  SE2Graphs("se2_graph_256x12", 256, 12, 2, SparseLinearSolverType::cuDSS, 50);
}

TEST(MinimizerBenchmark, DISABLED_SE3LoopClosurePGO) {
  constexpr int kPoses = 5000, kClosures = 2000;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> r(-0.1, 0.1), t(-0.4, 0.4);
  std::normal_distribution<double> jitter(0.0, 0.02);
  std::vector<Mat4> gt(kPoses), init(kPoses);
  gt[0] = Pose3({0, 0, 0}, {0, 0, 0});
  for (int i = 1; i < kPoses; ++i) {
    gt[i] = Mul<16, 4>(gt[i - 1], Pose3({r(rng), r(rng), r(rng)}, {t(rng), t(rng), t(rng)}));
  }
  init[0] = gt[0];
  for (int i = 1; i < kPoses; ++i) {
    init[i] = Mul<16, 4>(gt[i], Pose3({jitter(rng), jitter(rng), jitter(rng)},
                                      {jitter(rng), jitter(rng), jitter(rng)}));
  }
  std::vector<std::pair<int, int>> edges;
  for (int i = 0; i + 1 < kPoses; ++i) edges.push_back({i, i + 1});
  std::uniform_int_distribution<int> pick(0, kPoses - 1);
  while (static_cast<int>(edges.size()) < kPoses - 1 + kClosures) {
    const int a = pick(rng), b = pick(rng);
    if (std::abs(a - b) > 1) edges.push_back({a, b});
  }
  std::vector<Mat4> deltas;
  for (auto [a, b] : edges) deltas.push_back(Mul<16, 4>(gt[a], Inv<16, 4>(gt[b])));
  const std::vector<float> init_f = ToFloat(init);
  dvector<float> init_d(init_f), poses(init_f), deltas_d(ToFloat(deltas));
  dvector<int> const_ids(std::vector<int>{0});
  SE3StateBatch states(poses.data(), kPoses, const_ids.data(), 1);
  states.SetNumActiveStates(kPoses, 1);
  SE3BetweenFactorBatch between(reinterpret_cast<const SE3Transform *>(deltas_d.data()),
                                edges.size());
  between.SetNumActiveFactors(edges.size());
  std::vector<float *> ptrs;
  for (auto [a, b] : edges) {
    ptrs.push_back(states.StateDevicePtr(a));
    ptrs.push_back(states.StateDevicePtr(b));
  }
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&between, ptrs);
  RunBoth("se3_pgo_5000_lc2000", problem, SparseLinearSolverType::cuDSS, 15, 10, [&] {
    THROW_ON_CUDA_ERROR(cudaMemcpy(poses.data(), init_d.data(), init_f.size() * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
  });
}

/** Real-time AL calls on B SE(2) differential-drive trajectories of N stages (as real_time_test).
 */
TEST(MinimizerBenchmark, DISABLED_RealTimeMpc) {
  constexpr int B = 16, N = 20;
  std::vector<float> p, t;
  std::vector<int> cst, pid, cid, pst, cstg;
  for (int i = 0; i < B; ++i) {
    for (int k = 0; k <= N; ++k) {
      const Mat3 x = Pose2(0, 0, 0);
      p.insert(p.end(), x.begin(), x.end());
      pid.push_back(i);
      pst.push_back(k);
    }
    cst.push_back(i * (N + 1));
    for (int k = 0; k < N; ++k) {
      const Mat3 x = Pose2(0.1 * (k + 1), 0.25 * i, 0);
      t.insert(t.end(), x.begin(), x.end());
      cid.push_back(i);
      cstg.push_back(k);
    }
  }
  dvector<float> poses(p), init_poses(p), targets(t);
  const std::vector<float> zeros(2 * B * N, 0.f);
  dvector<float> controls(zeros), init_controls(zeros), nominal(zeros);
  dvector<float> dts(std::vector<float>(B * N, 0.1f));
  dvector<float> lower(std::vector<float>(2 * B * N, -12.f));
  dvector<float> upper(std::vector<float>(2 * B * N, 12.f));
  dvector<int> pose_const(cst), pose_ids(pid), control_ids(cid), pose_stages(pst),
      control_stages(cstg);
  SE2StateBatch pose_states(poses.data(), B * (N + 1), pose_const.data(), B);
  pose_states.SetNumActiveStates(B * (N + 1), B);
  VectorStateBatch<2> control_states(controls.data(), B * N);
  control_states.SetNumActiveStates(B * N);
  control_states.SetBounds(lower.data(), upper.data());
  SE2DifferentialDriveFactorBatch dynamics(dts.data(), 0.125f, 0.5f, B * N);
  dynamics.SetNumActiveFactors(B * N);
  ConstraintFactorBatch hard_dynamics(&dynamics, ConstraintKind::kEquality);
  SE2PriorFactorBatch track(reinterpret_cast<const SE2Transform *>(targets.data()), B * N);
  track.SetNumActiveFactors(B * N);
  WeightedFactorBatch<PriorVectorFactorBatch<2>> effort(
      0.05f, reinterpret_cast<const Vector<2> *>(nominal.data()), static_cast<size_t>(B * N));
  effort.SetNumActiveFactors(B * N);
  Problem problem;
  problem.AddStateBatch(&pose_states);
  problem.AddStateBatch(&control_states);
  std::vector<float *> dyn, trk, eff;
  for (int i = 0; i < B; ++i) {
    for (int k = 0; k < N; ++k) {
      dyn.push_back(pose_states.StateDevicePtr(i * (N + 1) + k));
      dyn.push_back(control_states.StateDevicePtr(i * N + k));
      dyn.push_back(pose_states.StateDevicePtr(i * (N + 1) + k + 1));
      trk.push_back(pose_states.StateDevicePtr(i * (N + 1) + k + 1));
      eff.push_back(control_states.StateDevicePtr(i * N + k));
    }
  }
  problem.AddFactorBatch(&hard_dynamics, dyn);
  problem.AddFactorBatch(&track, trk);
  problem.AddFactorBatch(&effort, eff);
  problem.SetProblemPartition(B, {pose_ids.data(), control_ids.data()});
  problem.SetStateStages({pose_stages.data(), control_stages.data()});

  MinimizerOptions mo;
  mo.sparse_linear_solver_type = SparseLinearSolverType::BlockTridiagonal;
  mo.state_tolerance = 1e-5f;
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = mo;
  LevenbergMarquardtMinimizer inner(lm);
  CudaStream stream;
  auto reset = [&] {
    THROW_ON_CUDA_ERROR(cudaMemcpy(poses.data(), init_poses.data(), p.size() * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
    THROW_ON_CUDA_ERROR(cudaMemcpy(controls.data(), init_controls.data(),
                                   zeros.size() * sizeof(float), cudaMemcpyDeviceToDevice));
  };
  auto summary = [](const AugmentedLagrangianMinimizerSummary &s) {
    MinimizerSummary out;
    out.num_iterations = s.inner_iterations;
    out.final_cost = s.final_cost;
    return out;
  };
  {
    AugmentedLagrangianMinimizerOptions options;
    options.max_penalty = 1e4f;
    AugmentedLagrangianMinimizer solver(inner, options);
    Print("al_mpc_16x20_converged", "LM",
          Time(20, reset, [&] { return summary(solver.Minimize(stream.GetStream(), problem)); }));
  }
  {
    AugmentedLagrangianMinimizerOptions options;
    options.warm_start = true;
    options.reuse_structure = true;
    options.real_time = true;
    options.max_outer_iterations = 2;
    options.inner_iterations = 2;
    options.inner_line_search_steps = 2;
    options.max_penalty = 1e3f;
    AugmentedLagrangianMinimizer solver(inner, options);
    Print("al_mpc_16x20_real_time", "LM",
          Time(
              200, [] {}, [&] { return summary(solver.Minimize(stream.GetStream(), problem)); }));
  }
}

}  // namespace
}  // namespace cunls
