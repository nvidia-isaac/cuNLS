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

/**
 * @file motion_factor_batch_test.cpp
 * @brief The constant-velocity and constant-acceleration motion priors on
 * SO(2), SO(3), SE(2) and SE(3) against a float64 reference
 * (tests/dynamics_test_support.h). With twist := Log(pose_k^{-1} pose_{k+1})
 * and Jl_inv := J_l^{-1}(twist):
 *
 *     r_pose  = twist - dt vel_k [- 0.5 dt^2 accel_k]
 *     r_vel   = Jl_inv vel_{k+1} - vel_k [- dt accel_k]
 *     r_accel = Jl_inv accel_{k+1} - accel_k            (acceleration only)
 *
 * The factors treat the pose-block Jacobians of r_vel and r_accel as zero (a
 * documented simplification), so the reference holds Jl_inv at the unperturbed
 * poses of each item: its central differences then check every other block.
 * Half of the items follow the motion model exactly and must have zero
 * residuals.
 */

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cunls/factor/motion/constant_acceleration_se2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_se3_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so3_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "dynamics_test_support.h"
#include "tests/utils.h"

namespace cunls {
namespace {

using namespace dynamics_test;

constexpr int kItems = 300;  // several thread blocks and a partial one

/** Group operations of the float64 reference. */
struct Group {
  SlotKind kind;
  int n;
  std::function<Vec(const double *)> exp;
  std::function<Vec(const Vec &)> log;
  std::function<Vec(const Vec &, const Vec &)> mul;
  std::function<Vec(const Vec &)> inv;
};

Group SO2() { return {SlotKind::kSO2, 1, ExpSO2, LogSO2, MulSO2, TransposeSO2}; }
Group SO3() { return {SlotKind::kSO3, 3, ExpSO3, LogSO3, Mul3, TransposeSO3}; }
Group SE2() { return {SlotKind::kSE2, 3, Exp3, Log3, Mul3, Inv3}; }
Group SE3() { return {SlotKind::kSE3, 6, Exp6, Log6, Mul4, Inv4}; }

/**
 * J_l(xi) u (or J_l^{-1}(xi) u when `inverse`) by central differences of
 * Exp(xi + d) = Exp(J_l(xi) d) Exp(xi), i.e. Log(Exp(u) Exp(xi)) = xi + J_l^{-1}(xi) u.
 */
Vec LeftJacobianTimes(const Group &g, const Vec &xi, const Vec &u, bool inverse) {
  const double h = 1e-5;
  auto at = [&](double s) {
    Vec d(g.n);
    if (inverse) {
      for (int i = 0; i < g.n; ++i) d[i] = s * u[i];
      return g.log(g.mul(g.exp(d.data()), g.exp(xi.data())));
    }
    for (int i = 0; i < g.n; ++i) d[i] = xi[i] + s * u[i];
    return g.log(g.mul(g.exp(d.data()), g.inv(g.exp(xi.data()))));
  };
  const Vec p = at(h), m = at(-h);
  Vec out(g.n);
  for (int i = 0; i < g.n; ++i) out[i] = (p[i] - m[i]) / (2 * h);
  return out;
}

using FactorFactory = std::function<std::unique_ptr<FactorBatch>(const float *dt, size_t n)>;

struct MotionCase {
  std::string name;
  Group (*group)();
  bool acceleration;  ///< constant acceleration (6 states) or constant velocity (4)
  FactorFactory make;
};

template <typename Factor>
FactorFactory Make() {
  return [](const float *dt, size_t n) { return std::make_unique<Factor>(dt, n); };
}

class MotionFactorTest : public ::testing::TestWithParam<MotionCase> {};

TEST_P(MotionFactorTest, MatchesReference) {
  const MotionCase &c = GetParam();
  const Group g = c.group();
  const int n = g.n;

  std::vector<float> dts(kItems);
  for (int t = 0; t < kItems; ++t) dts[t] = 0.1f + 0.1f * static_cast<float>(t % 4);
  dvector<float> d_dts(dts);
  auto factor = c.make(d_dts.data(), kItems);
  EXPECT_EQ(factor->StateSizes(), std::vector<size_t>(c.acceleration ? 6 : 4, n));

  std::mt19937 rng(17 + n + (c.acceleration ? 100 : 0));
  std::normal_distribution<double> normal(0.0, 1.0);
  auto random = [&](double scale) {
    Vec v(n);
    for (double &x : v) x = scale * normal(rng);
    return v;
  };

  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    const double dt = dts[t];
    const bool exact = t % 2 == 0;
    // Velocities from tiny (series branches) to large steps.
    const double scale = t % 3 == 0 ? 1e-4 : t % 3 == 1 ? 0.5 : 1.5;
    const Vec x = random(0.7), vel = random(scale), accel = random(c.acceleration ? scale : 0.0);
    Vec step(n), vel_plus_dt_accel(n);
    for (int i = 0; i < n; ++i) {
      step[i] = dt * vel[i] + 0.5 * dt * dt * accel[i];
      vel_plus_dt_accel[i] = vel[i] + dt * accel[i];
    }
    const Vec pose_k = g.exp(x.data());
    Vec pose_k1 = g.mul(pose_k, g.exp(step.data()));
    if (!exact) pose_k1 = g.mul(pose_k1, g.exp(random(0.2).data()));
    // Exact motion: the next velocity (and acceleration) is the transported one.
    const Vec vel_k1 = exact ? LeftJacobianTimes(g, step, vel_plus_dt_accel, false) : random(scale);
    std::vector<Slot> item = {{g.kind, ToFloat(pose_k)},
                              {g.kind, ToFloat(pose_k1)},
                              {SlotKind::kVector, ToFloat(vel)},
                              {SlotKind::kVector, ToFloat(vel_k1)}};
    if (c.acceleration) {
      const Vec accel_k1 = exact ? LeftJacobianTimes(g, step, accel, false) : random(scale);
      item.push_back({SlotKind::kVector, ToFloat(accel)});
      item.push_back({SlotKind::kVector, ToFloat(accel_k1)});
    }
    items.push_back(item);
  }

  // Jl_inv of each item at its unperturbed poses (column by column).
  std::vector<std::vector<Vec>> jl_inv(kItems);
  for (int t = 0; t < kItems; ++t) {
    const auto s = Perturbed(items[t], Vec(n * (c.acceleration ? 6 : 4), 0.0));
    const Vec twist = g.log(g.mul(g.inv(s[0]), s[1]));
    for (int j = 0; j < n; ++j) {
      Vec e(n, 0.0);
      e[j] = 1.0;
      jl_inv[t].push_back(LeftJacobianTimes(g, twist, e, true));
    }
  }

  const ReferenceResidual reference = [&](int t, const std::vector<Vec> &s) {
    const double dt = dts[t];
    const Vec twist = g.log(g.mul(g.inv(s[0]), s[1]));
    auto transported = [&](const Vec &v) {
      Vec out(n, 0.0);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) out[i] += jl_inv[t][j][i] * v[j];
      return out;
    };
    const Vec vel_k1 = transported(s[3]);
    Vec r(n * (c.acceleration ? 3 : 2));
    for (int i = 0; i < n; ++i) {
      const double accel = c.acceleration ? s[4][i] : 0.0;
      r[i] = twist[i] - dt * s[2][i] - 0.5 * dt * dt * accel;
      r[n + i] = vel_k1[i] - s[2][i] - dt * accel;
    }
    if (c.acceleration) {
      const Vec accel_k1 = transported(s[5]);
      for (int i = 0; i < n; ++i) r[2 * n + i] = accel_k1[i] - s[4][i];
    }
    return r;
  };

  for (int t = 0; t < kItems; t += 2) {
    const Vec r = reference(t, Perturbed(items[t], Vec(n * (c.acceleration ? 6 : 4), 0.0)));
    for (size_t i = 0; i < r.size(); ++i) {
      ASSERT_NEAR(r[i], 0.0, 1e-5) << "exact-motion item " << t << " row " << i;
    }
  }
  CheckFactor(*factor, items, reference, c.name);
}

INSTANTIATE_TEST_SUITE_P(
    AllGroups, MotionFactorTest,
    ::testing::Values(
        MotionCase{"ConstantVelocitySO2", SO2, false, Make<ConstantVelocitySO2FactorBatch>()},
        MotionCase{"ConstantVelocitySO3", SO3, false, Make<ConstantVelocitySO3FactorBatch>()},
        MotionCase{"ConstantVelocitySE2", SE2, false, Make<ConstantVelocitySE2FactorBatch>()},
        MotionCase{"ConstantVelocitySE3", SE3, false, Make<ConstantVelocitySE3FactorBatch>()},
        MotionCase{"ConstantAccelerationSO2", SO2, true,
                   Make<ConstantAccelerationSO2FactorBatch>()},
        MotionCase{"ConstantAccelerationSO3", SO3, true,
                   Make<ConstantAccelerationSO3FactorBatch>()},
        MotionCase{"ConstantAccelerationSE2", SE2, true,
                   Make<ConstantAccelerationSE2FactorBatch>()},
        MotionCase{"ConstantAccelerationSE3", SE3, true,
                   Make<ConstantAccelerationSE3FactorBatch>()}),
    [](const ::testing::TestParamInfo<MotionCase> &info) { return info.param.name; });

// Fixes pose_k, pose_k+1 as constant and solves for vel_k, vel_k+1 only. With
// poses fixed, the residual is exactly linear in the velocities, so the
// minimizer must recover the unique zero-residual solution.
TEST(ConstantVelocitySE3FactorBatchTest, RecoversVelocityWithPosesFixed) {
  const float dt = 0.4f;
  const Group g = SE3();
  const Vec x = {0.1, -0.15, 0.05, 0.4, 0.2, -0.3};
  const Vec step = {0.2, -0.1, 0.05, 0.6, -0.4, 0.1};
  const Vec pose_k = g.exp(x.data());
  const Vec pose_k1 = g.mul(pose_k, g.exp(step.data()));
  Vec expected_vel_k(6);
  for (int i = 0; i < 6; i++) expected_vel_k[i] = step[i] / dt;
  const Vec expected_vel_k1 = LeftJacobianTimes(g, step, expected_vel_k, false);

  std::vector<float> poses = ToFloat(pose_k);
  const auto pose_k1_f = ToFloat(pose_k1);
  poses.insert(poses.end(), pose_k1_f.begin(), pose_k1_f.end());
  dvector<float> poses_dev(poses);
  dvector<int> const_ids_dev(std::vector<int>{0, 1});
  SE3StateBatch pose_batch(poses_dev.data(), 2, const_ids_dev.data(), 2);
  pose_batch.SetNumActiveStates(pose_batch.Capacity(), pose_batch.ConstCapacity());

  // Initialize velocities away from the expected solution.
  dvector<Vector<6>> vel_dev(std::vector<Vector<6>>(2, Vector<6>{}));
  VectorStateBatch<6> vel_batch(reinterpret_cast<const float *>(vel_dev.data()), 2);
  vel_batch.SetNumActiveStates(vel_batch.Capacity(), vel_batch.ConstCapacity());

  dvector<float> dt_dev(std::vector<float>{dt});
  ConstantVelocitySE3FactorBatch factor(dt_dev.data(), 1);
  factor.SetNumActiveFactors(factor.Capacity());

  Problem problem;
  problem.AddStateBatch(&pose_batch);
  problem.AddStateBatch(&vel_batch);
  problem.AddFactorBatch(&factor, {pose_batch.StateDevicePtr(0), pose_batch.StateDevicePtr(1),
                                   vel_batch.StateDevicePtr(0), vel_batch.StateDevicePtr(1)});
  ASSERT_TRUE(problem.CheckConsistency());

  CudaStream stream;
  MinimizerOptions options;
  options.max_num_iterations = 5;
  options.state_tolerance = 1e-8f;
  options.cost_tolerance = 1e-8f;
  options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
  options.sparse_linear_solver_config.block_sparse_pcg_options.block_size =
      test_utils::PCGBlockSizeFromEnv(6);
  options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations =
      test_utils::PCGMaxIterFromEnv(400);
  options.sparse_linear_solver_config.block_sparse_pcg_options.relative_tolerance =
      test_utils::PCGTolFromEnv(1e-6f);

  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-4f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  ASSERT_LT(summary.final_cost, 1e-4f);

  std::vector<Vector<6>> recovered(2);
  vel_dev.CopyToHost(recovered.data(), 2);
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(recovered[0][i], expected_vel_k[i], 1e-2f) << "vel_k[" << i << "]";
    EXPECT_NEAR(recovered[1][i], expected_vel_k1[i], 1e-2f) << "vel_k+1[" << i << "]";
  }
}

}  // namespace
}  // namespace cunls
