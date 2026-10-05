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

// SparseLinearSolverType::BlockTridiagonal on MPC-shaped problems: SE(2)
// trajectories with a constant first pose, wheel-speed controls, soft
// differential-drive dynamics, pose and control priors, one subproblem per
// trajectory. Gauss-Newton with it matches Gauss-Newton with the dense
// Cholesky solver; a factor that skips a stage is rejected.

#include "cunls/linear_solver/block_tridiagonal_solver.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/dynamics/se2_differential_drive_factor_batch.h"
#include "cunls/factor/information/information_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

std::vector<float> Se2(float x, float y, float th) {
  const float c = std::cos(th), s = std::sin(th);
  return {c, -s, x, s, c, y, 0.f, 0.f, 1.f};
}

/** B trajectories of N steps; stage k holds pose k and control k. */
struct Trajectories {
  int B, N;
  dvector<float> poses, controls, targets, nominal, dts;
  dvector<int> pose_const, pose_ids, control_ids, pose_stages, control_stages;
  std::unique_ptr<SE2StateBatch> pose_states;
  std::unique_ptr<VectorStateBatch<2>> control_states;
  std::unique_ptr<WeightedFactorBatch<SE2DifferentialDriveFactorBatch>> dynamics;
  std::unique_ptr<SE2PriorFactorBatch> track;
  std::unique_ptr<WeightedFactorBatch<PriorVectorFactorBatch<2>>> effort;
  Problem problem;

  Trajectories(int b, int n) : B(b), N(n) {
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> p, t, ids_p, unused;
    std::vector<int> cst, pid, cid, pst, cstg;
    for (int i = 0; i < B; ++i) {
      for (int k = 0; k <= N; ++k) {
        const auto x = Se2(0.1f * normal(rng), 0.1f * normal(rng), 0.1f * normal(rng));
        p.insert(p.end(), x.begin(), x.end());
        pid.push_back(i);
        pst.push_back(k);
      }
      cst.push_back(i * (N + 1));
      for (int k = 0; k < N; ++k) {
        const auto x = Se2(0.3f * (k + 1), 0.2f * i + 0.1f * normal(rng), 0.05f * normal(rng));
        t.insert(t.end(), x.begin(), x.end());
        cid.push_back(i);
        cstg.push_back(k);
      }
    }
    poses = dvector<float>(p);
    targets = dvector<float>(t);
    controls = dvector<float>(std::vector<float>(2 * B * N, 0.f));
    nominal = dvector<float>(std::vector<float>(2 * B * N, 1.f));
    dts = dvector<float>(std::vector<float>(B * N, 0.1f));
    pose_const = dvector<int>(cst);
    pose_ids = dvector<int>(pid);
    control_ids = dvector<int>(cid);
    pose_stages = dvector<int>(pst);
    control_stages = dvector<int>(cstg);

    pose_states = std::make_unique<SE2StateBatch>(poses.data(), B * (N + 1), pose_const.data(), B);
    pose_states->SetNumActiveStates(B * (N + 1), B);
    control_states = std::make_unique<VectorStateBatch<2>>(controls.data(), B * N);
    control_states->SetNumActiveStates(B * N);
    dynamics = std::make_unique<WeightedFactorBatch<SE2DifferentialDriveFactorBatch>>(
        10.f, dts.data(), 0.125f, 0.5f, static_cast<size_t>(B * N));
    dynamics->SetNumActiveFactors(B * N);
    track = std::make_unique<SE2PriorFactorBatch>(
        reinterpret_cast<const SE2Transform *>(targets.data()), B * N);
    track->SetNumActiveFactors(B * N);
    effort = std::make_unique<WeightedFactorBatch<PriorVectorFactorBatch<2>>>(
        0.05f, reinterpret_cast<const Vector<2> *>(nominal.data()), static_cast<size_t>(B * N));
    effort->SetNumActiveFactors(B * N);

    problem.AddStateBatch(pose_states.get());
    problem.AddStateBatch(control_states.get());
    std::vector<float *> dyn, trk, eff;
    for (int i = 0; i < B; ++i) {
      for (int k = 0; k < N; ++k) {
        dyn.push_back(pose_states->StateDevicePtr(i * (N + 1) + k));
        dyn.push_back(control_states->StateDevicePtr(i * N + k));
        dyn.push_back(pose_states->StateDevicePtr(i * (N + 1) + k + 1));
        trk.push_back(pose_states->StateDevicePtr(i * (N + 1) + k + 1));
        eff.push_back(control_states->StateDevicePtr(i * N + k));
      }
    }
    problem.AddFactorBatch(dynamics.get(), dyn);
    problem.AddFactorBatch(track.get(), trk);
    problem.AddFactorBatch(effort.get(), eff);
    if (B > 1) problem.SetProblemPartition(B, {pose_ids.data(), control_ids.data()});
    problem.SetStateStages({pose_stages.data(), control_stages.data()});
  }

  std::vector<float> Solve(SparseLinearSolverType type) {
    MinimizerOptions options;
    options.sparse_linear_solver_type = type;
    // A few iterations: the steps of both solvers agree to float round-off;
    // further on, the batched iteration (no line search) amplifies round-off.
    options.max_num_iterations = 4;
    GaussNewtonMinimizer minimizer(options);
    CudaStream stream;
    minimizer.Minimize(stream.GetStream(), problem);
    std::vector<float> out(poses.size() + controls.size());
    poses.CopyToHost(out.data(), poses.size());
    controls.CopyToHost(out.data() + poses.size(), controls.size());
    return out;
  }
};

class BlockTridiagonalSolverTest : public ::testing::TestWithParam<int> {};

TEST_P(BlockTridiagonalSolverTest, MatchesDenseCholesky) {
  const int B = GetParam();
  Trajectories a(B, 12), b(B, 12);
  const auto tridiagonal = a.Solve(SparseLinearSolverType::BlockTridiagonal);
  const auto dense = b.Solve(SparseLinearSolverType::DenseCholesky);
  ASSERT_EQ(tridiagonal.size(), dense.size());
  for (size_t i = 0; i < dense.size(); ++i) {
    EXPECT_NEAR(tridiagonal[i], dense[i], 2e-4f) << "entry " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Subproblems, BlockTridiagonalSolverTest, ::testing::Values(1, 5));

TEST(BlockTridiagonalSolver, RejectsNonTridiagonalCoupling) {
  Trajectories t(1, 6);
  // A between factor from stage 1 to stage 3 breaks the block-tridiagonal pattern.
  dvector<float> delta(Se2(0.f, 0.f, 0.f));
  SE2BetweenFactorBatch skip(reinterpret_cast<const SE2Transform *>(delta.data()), 1);
  skip.SetNumActiveFactors(1);
  t.problem.AddFactorBatch(&skip,
                           {t.pose_states->StateDevicePtr(1), t.pose_states->StateDevicePtr(3)});
  t.problem.SetStateStages({t.pose_stages.data(), t.control_stages.data()});
  EXPECT_THROW(t.Solve(SparseLinearSolverType::BlockTridiagonal), std::runtime_error);
}

TEST(BlockTridiagonalSolver, NeedsStages) {
  Trajectories t(1, 4);
  t.problem.SetStateStages({});
  EXPECT_THROW(t.Solve(SparseLinearSolverType::BlockTridiagonal), std::runtime_error);
}

// A singular system (the second component of every state is unconstrained):
// with safety checks the solve fails and the minimizer throws; without them
// (the default, CUDA-graph capturable) it runs through on the floored pivots.
TEST(BlockTridiagonalSolver, ReportsSingularPivotsWithSafetyChecks) {
  for (bool safety_checks : {true, false}) {
    constexpr int kStates = 4;
    dvector<float> x(std::vector<float>(2 * kStates, 0.f)),
        targets(std::vector<float>(2 * kStates, 1.f));
    dvector<Matrix<2>> sqrt_info(std::vector<Matrix<2>>(kStates, Matrix<2>{1.f, 0.f, 0.f, 0.f}));
    dvector<int> stages(std::vector<int>{0, 1, 2, 3});
    VectorStateBatch<2> states(x.data(), kStates);
    states.SetNumActiveStates(kStates);
    InformationFactorBatch<PriorVectorFactorBatch<2>> priors(
        sqrt_info.data(), kStates, reinterpret_cast<const Vector<2> *>(targets.data()), kStates);
    priors.SetNumActiveFactors(kStates);
    std::vector<float *> ptrs;
    for (int i = 0; i < kStates; ++i) ptrs.push_back(states.StateDevicePtr(i));
    Problem problem;
    problem.AddStateBatch(&states);
    problem.AddFactorBatch(&priors, ptrs);
    problem.SetStateStages({stages.data()});
    MinimizerOptions options;
    options.sparse_linear_solver_type = SparseLinearSolverType::BlockTridiagonal;
    options.disable_safety_checks = !safety_checks;
    GaussNewtonMinimizer gn(options);
    CudaStream stream;
    if (safety_checks) {
      EXPECT_THROW(gn.Minimize(stream.GetStream(), problem), std::runtime_error);
    } else {
      EXPECT_NO_THROW(gn.Minimize(stream.GetStream(), problem));
    }
  }
}

/**
 * A random SPD system with P subproblems of K stages of M unknowns, block
 * tridiagonal in stage order, as the solver sees it: CSR in the order of one
 * VectorStateBatch<M> whose state p*K + k is stage k of subproblem p.
 */
template <int M>
struct RandomSystem {
  int P, K;
  size_t n;
  std::vector<double> dense_blocks;  // per (p, k): D (M*M), C (M*M), host copy for the reference
  std::vector<float> rhs;
  CSRSparseMatrix csr;
  dvector<float> states;
  dvector<int> ids, stages;
  std::unique_ptr<VectorStateBatch<M>> batch;
  Problem problem;

  RandomSystem(int p, int k, unsigned seed) : P(p), K(k), n(static_cast<size_t>(p) * k * M) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    // H = Jᵀ J + I with J block bidiagonal: SPD and block tridiagonal.
    std::vector<double> H(static_cast<size_t>(P) * K * 2 * M * M, 0.0);
    auto D = [&](int pp, int kk) { return &H[(static_cast<size_t>(pp) * K + kk) * 2 * M * M]; };
    auto C = [&](int pp, int kk) { return D(pp, kk) + M * M; };
    for (int pp = 0; pp < P; ++pp) {
      for (int kk = 0; kk < K; ++kk) {
        std::vector<double> A(M * M), B(M * M);  // J rows of stage kk: A x_kk + B x_kk+1
        for (double &v : A) v = normal(rng) / std::sqrt(static_cast<double>(M));
        for (double &v : B) v = normal(rng) / std::sqrt(static_cast<double>(M));
        if (kk + 1 == K) std::fill(B.begin(), B.end(), 0.0);
        for (int i = 0; i < M; ++i) {
          for (int j = 0; j < M; ++j) {
            double aa = 0, ab = 0, bb = 0;
            for (int r = 0; r < M; ++r) {
              aa += A[r * M + i] * A[r * M + j];
              ab += A[r * M + i] * B[r * M + j];
              bb += B[r * M + i] * B[r * M + j];
            }
            D(pp, kk)[i * M + j] += aa + (i == j ? 1.0 : 0.0);
            C(pp, kk)[i * M + j] += ab;
            if (kk + 1 < K) D(pp, kk + 1)[i * M + j] += bb;
          }
        }
      }
    }
    dense_blocks = H;
    std::vector<int> offsets = {0}, cols;
    std::vector<float> values;
    for (int pp = 0; pp < P; ++pp) {
      for (int kk = 0; kk < K; ++kk) {
        for (int i = 0; i < M; ++i) {
          if (kk > 0) {  // H_{k,k-1} = C_{k-1}ᵀ
            for (int j = 0; j < M; ++j) {
              cols.push_back((pp * K + kk - 1) * M + j);
              values.push_back(static_cast<float>(C(pp, kk - 1)[j * M + i]));
            }
          }
          for (int j = 0; j < M; ++j) {
            cols.push_back((pp * K + kk) * M + j);
            values.push_back(static_cast<float>(D(pp, kk)[i * M + j]));
          }
          if (kk + 1 < K) {
            for (int j = 0; j < M; ++j) {
              cols.push_back((pp * K + kk + 1) * M + j);
              values.push_back(static_cast<float>(C(pp, kk)[i * M + j]));
            }
          }
          offsets.push_back(static_cast<int>(cols.size()));
        }
      }
    }
    csr.row_offsets = dvector<int>(offsets);
    csr.col_ids = dvector<int>(cols);
    csr.values = dvector<float>(values);
    rhs.resize(n);
    for (float &v : rhs) v = static_cast<float>(normal(rng));
    std::vector<int> h_ids(static_cast<size_t>(P) * K), h_stages(static_cast<size_t>(P) * K);
    for (int s = 0; s < P * K; ++s) {
      h_ids[s] = s / K;
      h_stages[s] = s % K;
    }
    states = dvector<float>(std::vector<float>(n, 0.f));
    ids = dvector<int>(h_ids);
    stages = dvector<int>(h_stages);
    batch = std::make_unique<VectorStateBatch<M>>(states.data(), static_cast<size_t>(P) * K);
    batch->SetNumActiveStates(static_cast<size_t>(P) * K);
    problem.AddStateBatch(batch.get());
    if (P > 1) problem.SetProblemPartition(P, {ids.data()});
    problem.SetStateStages({stages.data()});
  }

  /** Float64 block Cholesky reference for subproblem pp. */
  std::vector<double> Reference(int pp) const {
    std::vector<double> x(static_cast<size_t>(K) * M);
    // Dense solve of the subproblem (small K * M in the tests).
    const int N = K * M;
    std::vector<double> A(static_cast<size_t>(N) * N, 0.0), b(N);
    for (int kk = 0; kk < K; ++kk) {
      const double *Dk = &dense_blocks[(static_cast<size_t>(pp) * K + kk) * 2 * M * M];
      const double *Ck = Dk + M * M;
      for (int i = 0; i < M; ++i) {
        b[kk * M + i] = rhs[(static_cast<size_t>(pp) * K + kk) * M + i];
        for (int j = 0; j < M; ++j) {
          A[static_cast<size_t>(kk * M + i) * N + kk * M + j] = Dk[i * M + j];
          if (kk + 1 < K) {
            A[static_cast<size_t>(kk * M + i) * N + (kk + 1) * M + j] = Ck[i * M + j];
            A[static_cast<size_t>((kk + 1) * M + j) * N + kk * M + i] = Ck[i * M + j];
          }
        }
      }
    }
    // Gaussian elimination (SPD, no pivoting needed).
    for (int c = 0; c < N; ++c) {
      for (int r = c + 1; r < N; ++r) {
        const double f = A[static_cast<size_t>(r) * N + c] / A[static_cast<size_t>(c) * N + c];
        if (f == 0.0) continue;
        for (int j = c; j < N; ++j)
          A[static_cast<size_t>(r) * N + j] -= f * A[static_cast<size_t>(c) * N + j];
        b[r] -= f * b[c];
      }
    }
    for (int r = N - 1; r >= 0; --r) {
      double v = b[r];
      for (int j = r + 1; j < N; ++j) v -= A[static_cast<size_t>(r) * N + j] * x[j];
      x[r] = v / A[static_cast<size_t>(r) * N + r];
    }
    return x;
  }
};

template <int M>
void CheckRandomSystem(int P, int K) {
  RandomSystem<M> system(P, K, 11 + M);
  dvector<float> rhs(system.rhs), result(std::vector<float>(system.n, 0.f));
  CudaStream stream;
  BlockTridiagonalSolver solver;
  ASSERT_TRUE(solver.Initialize(stream.GetStream(), system.problem, system.csr, rhs, result));
  ASSERT_TRUE(solver.Solve(stream.GetStream(), system.csr, rhs, result));
  std::vector<float> x(system.n);
  result.CopyToHost(x.data(), x.size());
  for (int pp = 0; pp < P; ++pp) {
    const auto expected = system.Reference(pp);
    for (size_t i = 0; i < expected.size(); ++i) {
      const float got = x[static_cast<size_t>(pp) * K * M + i];
      EXPECT_NEAR(got, expected[i], 1e-3 * (1.0 + std::abs(expected[i])))
          << "M " << M << " subproblem " << pp << " entry " << i;
    }
  }
}

TEST(BlockTridiagonalSolver, RandomSystemsMatchFloat64) {
  CheckRandomSystem<1>(3, 7);
  CheckRandomSystem<5>(2, 9);
  CheckRandomSystem<16>(3, 6);
  CheckRandomSystem<24>(2, 5);
  CheckRandomSystem<32>(1, 4);
}

template <int M>
void TimeSolver(int P, int K) {
  RandomSystem<M> system(P, K, 3);
  dvector<float> rhs(system.rhs), result(std::vector<float>(system.n, 0.f));
  CudaStream stream;
  BlockTridiagonalSolver solver;
  ASSERT_TRUE(solver.Initialize(stream.GetStream(), system.problem, system.csr, rhs, result));
  for (int i = 0; i < 3; ++i) solver.Solve(stream.GetStream(), system.csr, rhs, result);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  constexpr int kRuns = 20;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kRuns; ++i) solver.Solve(stream.GetStream(), system.csr, rhs, result);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const double us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() /
      kRuns;
  std::printf("BlockTridiagonal P=%5d K=%3d M=%2d: %9.1f us per solve\n", P, K, M, us);
}

// Throughput of the solve (scatter, kernel, gather) on the MPC sizes; run with
// --gtest_also_run_disabled_tests.
TEST(BlockTridiagonalSolver, DISABLED_Throughput) {
  TimeSolver<5>(1, 51);
  TimeSolver<7>(1, 51);
  TimeSolver<16>(1, 41);
  TimeSolver<24>(1, 21);
  TimeSolver<16>(4096, 41);
  TimeSolver<24>(1024, 21);
}

}  // namespace
}  // namespace cunls
