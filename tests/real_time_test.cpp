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

// Real-time augmented Lagrangian on an MPC-shaped problem (SE(2) trajectories
// with hard differential-drive dynamics, wheel-speed bounds by projection,
// pose tracking, the block-tridiagonal solver, one subproblem per
// trajectory): warm-started real-time calls with a reused structure keep the
// fixed budget and stay feasible while the measured first pose changes every
// call, and bound buffers rebound between calls take effect.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/dynamics/se2_differential_drive_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

std::vector<float> Se2(float x, float y, float th) {
  const float c = std::cos(th), s = std::sin(th);
  return {c, -s, x, s, c, y, 0.f, 0.f, 1.f};
}

struct Mpc {
  static constexpr int B = 4, N = 20;
  dvector<float> poses, controls, targets, nominal, dts, lower, upper;
  dvector<int> pose_const, pose_ids, control_ids, pose_stages, control_stages;
  std::unique_ptr<SE2StateBatch> pose_states;
  std::unique_ptr<VectorStateBatch<2>> control_states;
  std::unique_ptr<SE2DifferentialDriveFactorBatch> dynamics;
  std::unique_ptr<ConstraintFactorBatch> hard_dynamics;
  std::unique_ptr<SE2PriorFactorBatch> track;
  std::unique_ptr<WeightedFactorBatch<PriorVectorFactorBatch<2>>> effort;
  Problem problem;
  std::unique_ptr<LevenbergMarquardtMinimizer> inner;
  std::unique_ptr<AugmentedLagrangianMinimizer> solver;

  Mpc() {
    std::vector<float> p, t;
    std::vector<int> cst, pid, cid, pst, cstg;
    for (int i = 0; i < B; ++i) {
      for (int k = 0; k <= N; ++k) {
        const auto x = Se2(0.f, 0.f, 0.f);
        p.insert(p.end(), x.begin(), x.end());
        pid.push_back(i);
        pst.push_back(k);
      }
      cst.push_back(i * (N + 1));
      for (int k = 0; k < N; ++k) {
        const auto x = Se2(0.1f * (k + 1), 0.25f * i, 0.f);
        t.insert(t.end(), x.begin(), x.end());
        cid.push_back(i);
        cstg.push_back(k);
      }
    }
    poses = dvector<float>(p);
    targets = dvector<float>(t);
    controls = dvector<float>(std::vector<float>(2 * B * N, 0.f));
    nominal = dvector<float>(std::vector<float>(2 * B * N, 0.f));
    dts = dvector<float>(std::vector<float>(B * N, 0.1f));
    lower = dvector<float>(std::vector<float>(2 * B * N, -12.f));
    upper = dvector<float>(std::vector<float>(2 * B * N, 12.f));
    pose_const = dvector<int>(cst);
    pose_ids = dvector<int>(pid);
    control_ids = dvector<int>(cid);
    pose_stages = dvector<int>(pst);
    control_stages = dvector<int>(cstg);

    pose_states = std::make_unique<SE2StateBatch>(poses.data(), B * (N + 1), pose_const.data(), B);
    pose_states->SetNumActiveStates(B * (N + 1), B);
    control_states = std::make_unique<VectorStateBatch<2>>(controls.data(), B * N);
    control_states->SetNumActiveStates(B * N);
    control_states->SetBounds(lower.data(), upper.data());
    dynamics = std::make_unique<SE2DifferentialDriveFactorBatch>(dts.data(), 0.125f, 0.5f, B * N);
    dynamics->SetNumActiveFactors(B * N);
    hard_dynamics =
        std::make_unique<ConstraintFactorBatch>(dynamics.get(), ConstraintKind::kEquality);
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
    problem.AddFactorBatch(hard_dynamics.get(), dyn);
    problem.AddFactorBatch(track.get(), trk);
    problem.AddFactorBatch(effort.get(), eff);
    problem.SetProblemPartition(B, {pose_ids.data(), control_ids.data()});
    problem.SetStateStages({pose_stages.data(), control_stages.data()});

    MinimizerOptions mo;
    mo.sparse_linear_solver_type = SparseLinearSolverType::BlockTridiagonal;
    mo.state_tolerance = 1e-5f;
    LevenbergMarquardtMinimizerOptions lm;
    lm.base_options = mo;
    inner = std::make_unique<LevenbergMarquardtMinimizer>(lm);
    AugmentedLagrangianMinimizerOptions options;
    options.warm_start = true;
    options.reuse_structure = true;
    options.max_penalty = 1e4f;
    solver = std::make_unique<AugmentedLagrangianMinimizer>(*inner, options);
    // After the first (converged) call: the real-time budget.
    options.real_time = true;
    options.max_outer_iterations = 2;
    options.inner_iterations = 2;
    options.inner_line_search_steps = 2;
    options.max_penalty = 1e3f;
    real_time_options = options;
  }

  AugmentedLagrangianMinimizerOptions real_time_options;

  /** Writes the measured first pose of every trajectory. */
  void Measure(int call) {
    for (int i = 0; i < B; ++i) {
      const auto x = Se2(0.02f * call, 0.25f * i + 0.1f * std::sin(0.7f * call), 0.05f * call);
      THROW_ON_CUDA_ERROR(cudaMemcpy(pose_states->StateDevicePtr(i * (N + 1)), x.data(),
                                     9 * sizeof(float), cudaMemcpyHostToDevice));
    }
  }

  std::vector<float> States() const {
    std::vector<float> out(poses.size() + controls.size());
    poses.CopyToHost(out.data(), poses.size());
    controls.CopyToHost(out.data() + poses.size(), controls.size());
    return out;
  }
};

TEST(RealTime, WarmStartedCallsStayFeasibleAndFollowNewBounds) {
  Mpc m;
  CudaStream stream;
  for (int call = 0; call < 8; ++call) {
    m.Measure(call);
    const auto summary = m.solver->Minimize(stream.GetStream(), m.problem);
    if (call == 0) {
      EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
      m.solver->SetOptions(m.real_time_options);
    } else {
      EXPECT_EQ(summary.outer_iterations, 2u) << "call " << call;
      EXPECT_LE(summary.max_violation, 5e-2f) << "call " << call;
    }
  }
  // New bound buffers (other addresses, tighter values) between real-time
  // calls with a reused structure: the new bounds hold.
  dvector<float> lower2(std::vector<float>(2 * Mpc::B * Mpc::N, -6.f));
  dvector<float> upper2(std::vector<float>(2 * Mpc::B * Mpc::N, 6.f));
  m.control_states->SetBounds(lower2.data(), upper2.data());
  const size_t first_control = Mpc::B * (Mpc::N + 1) * 9;
  for (int call = 8; call < 11; ++call) {
    m.Measure(call);
    const auto summary = m.solver->Minimize(stream.GetStream(), m.problem);
    EXPECT_LE(summary.max_violation, 5e-2f) << "call " << call;
    const auto s = m.States();
    for (size_t i = first_control; i < s.size(); ++i) {
      EXPECT_LE(std::abs(s[i]), 6.f + 1e-5f) << "call " << call << " entry " << i;
    }
  }
}

}  // namespace
}  // namespace cunls
