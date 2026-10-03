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

// SE2DifferentialDriveFactorBatch and SE2KinematicBicycleFactorBatch against a
// float64 reference written here (SE(2) exponential and logarithm, the Euler
// step): residuals, analytic Jacobians against central differences, the
// residual-only path, and a Carter driving to a goal under
// AugmentedLagrangianMinimizer.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/bound_factor_batch.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/dynamics/se2_differential_drive_factor_batch.h"
#include "cunls/factor/dynamics/se2_kinematic_bicycle_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "evaluate_items_check.h"

namespace cunls {
namespace {

// --- float64 SE(2) reference: 3x3 row-major matrices, tangent [v_x, v_y, θ] ---

using Mat3 = std::array<double, 9>;

Mat3 Mul(const Mat3 &a, const Mat3 &b) {
  Mat3 c{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return c;
}

Mat3 Inv(const Mat3 &t) {
  return {t[0], t[3], -(t[0] * t[2] + t[3] * t[5]), t[1], t[4], -(t[1] * t[2] + t[4] * t[5]), 0,
          0,    1};
}

// sin θ / θ and (1 - cos θ) / θ = 2 sin²(θ/2) / θ (no cancellation: central
// differences evaluate these at θ ~ 1e-8).
void Coefficients(double th, double *a, double *b) {
  if (th == 0.0) {
    *a = 1;
    *b = 0;
  } else {
    const double h = std::sin(0.5 * th);
    *a = std::sin(th) / th;
    *b = 2 * h * h / th;
  }
}

Mat3 Exp(double vx, double vy, double th) {
  double a, b;
  Coefficients(th, &a, &b);
  return {std::cos(th),
          -std::sin(th),
          a * vx - b * vy,
          std::sin(th),
          std::cos(th),
          b * vx + a * vy,
          0,
          0,
          1};
}

std::array<double, 3> Log(const Mat3 &t) {
  const double th = std::atan2(t[3], t[0]);
  double a, b;
  Coefficients(th, &a, &b);
  const double d = a * a + b * b;
  return {(a * t[2] + b * t[5]) / d, (-b * t[2] + a * t[5]) / d, th};
}

Mat3 ToMat(const float *m) {
  Mat3 r;
  for (int i = 0; i < 9; ++i) r[i] = m[i];
  return r;
}

std::vector<float> ToFloat(const Mat3 &m) { return std::vector<float>(m.begin(), m.end()); }

/** Body twist of each model at (state z, control u). */
struct Model {
  int vector_size;  // 0 (Carter) or 2 (car: z = (v, δ))
  std::function<std::array<double, 3>(const double *z, const double *u)> twist;
};

constexpr double kR = 0.125, kB = 0.5, kL = 2.75;  // exact in float

Model Carter() {
  return {0, [](const double *, const double *u) {
            return std::array<double, 3>{kR * (u[1] + u[0]) / 2, 0, kR * (u[1] - u[0]) / kB};
          }};
}

Model Car() {
  return {2, [](const double *z, const double *) {
            return std::array<double, 3>{z[0], 0, z[0] * std::tan(z[1]) / kL};
          }};
}

/**
 * Reference residual with every input perturbed by `d` (column order of the
 * factor: T_k, [z_k], u_k, T_{k+1}, [z_{k+1}]); poses perturbed on the right.
 */
std::vector<double> Residual(const Model &m, const Mat3 &X, const std::vector<double> &z,
                             const std::vector<double> &u, const Mat3 &Y,
                             const std::vector<double> &zn, double dt, const double *d) {
  const int nz = m.vector_size;
  int c = 0;
  const Mat3 Xp = Mul(X, Exp(d[c], d[c + 1], d[c + 2]));
  c += 3;
  std::vector<double> zp(z), up(u), znp(zn);
  for (int i = 0; i < nz; ++i) zp[i] += d[c++];
  for (int i = 0; i < 2; ++i) up[i] += d[c++];
  const Mat3 Yp = Mul(Y, Exp(d[c], d[c + 1], d[c + 2]));
  c += 3;
  for (int i = 0; i < nz; ++i) znp[i] += d[c++];
  const auto xi = m.twist(zp.data(), up.data());
  const Mat3 Phi = Mul(Xp, Exp(dt * xi[0], dt * xi[1], dt * xi[2]));
  const auto r = Log(Mul(Inv(Phi), Yp));
  std::vector<double> out(r.begin(), r.end());
  for (int i = 0; i < nz; ++i) out.push_back(znp[i] - zp[i] - dt * up[i]);
  return out;
}

/** Random steps: next states consistent with the Euler step (even items) or perturbed (odd). */
void CheckAgainstReference(const Model &m, FactorBatch &factor, const std::string &label) {
  const int nz = m.vector_size, n = 3 + nz, cols = 2 * n + 2;
  const int items = static_cast<int>(factor.Capacity());
  std::mt19937 rng(3);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<Mat3> X(items), Y(items);
  std::vector<std::vector<double>> z(items), u(items), zn(items);
  std::vector<float> dts(items), storage;
  std::vector<size_t> offsets;
  auto push = [&](const std::vector<float> &v) {
    offsets.push_back(storage.size());
    storage.insert(storage.end(), v.begin(), v.end());
  };
  for (int t = 0; t < items; ++t) {
    dts[t] = 0.05f + 0.05f * static_cast<float>(t % 4);
    X[t] = ToMat(ToFloat(Exp(normal(rng), normal(rng), normal(rng))).data());
    z[t] = {std::round(3.0 * normal(rng) * 64) / 64, std::round(0.4 * normal(rng) * 64) / 64};
    z[t].resize(nz);
    u[t] = {std::round(4.0 * normal(rng) * 64) / 64, std::round(4.0 * normal(rng) * 64) / 64};
    // Y: the Euler step itself, perturbed for odd items.
    const auto xi = m.twist(z[t].data(), u[t].data());
    Mat3 Phi = Mul(X[t], Exp(dts[t] * xi[0], dts[t] * xi[1], dts[t] * xi[2]));
    const double s = t % 2 == 0 ? 0.0 : 0.2;
    Y[t] = ToMat(ToFloat(Mul(Phi, Exp(s * normal(rng), s * normal(rng), s * normal(rng)))).data());
    zn[t] = z[t];
    for (int i = 0; i < nz; ++i) {
      zn[t][i] += dts[t] * u[t][i] + s * normal(rng);
      zn[t][i] = static_cast<float>(zn[t][i]);
    }
    push(ToFloat(X[t]));
    if (nz > 0) push(std::vector<float>(z[t].begin(), z[t].end()));
    push(std::vector<float>(u[t].begin(), u[t].end()));
    push(ToFloat(Y[t]));
    if (nz > 0) push(std::vector<float>(zn[t].begin(), zn[t].end()));
  }
  CudaStream stream;
  dvector<float> d_storage(storage);
  std::vector<float *> ptrs;
  for (size_t o : offsets) ptrs.push_back(d_storage.data() + o);
  dvector<float *> d_ptrs(ptrs);
  dvector<float> d_res(items * n), d_jac(items * n * cols), d_res_only(items * n);
  factor.SetNumActiveFactors(items);
  ASSERT_TRUE(factor.Evaluate(d_res.data(), d_jac.data(), d_ptrs.data(), stream.GetStream()));
  ASSERT_TRUE(factor.Evaluate(d_res_only.data(), nullptr, d_ptrs.data(), stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> res(d_res.size()), jac(d_jac.size()), res_only(d_res_only.size());
  d_res.CopyToHost(res.data(), res.size());
  d_jac.CopyToHost(jac.data(), jac.size());
  d_res_only.CopyToHost(res_only.data(), res_only.size());

  // The item contract (factor_ids / repeated copies) matches plain evaluation.
  evaluate_items_test::CheckEvaluateItems(factor, 3, [&](int) { return ptrs; });

  for (int t = 0; t < items; ++t) {
    std::vector<double> d(cols, 0.0);
    const auto r0 = Residual(m, X[t], z[t], u[t], Y[t], zn[t], dts[t], d.data());
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(res[t * n + i], r0[i], 2e-5 * (1 + std::fabs(r0[i])))
          << label << " item " << t << " row " << i;
      EXPECT_EQ(res_only[t * n + i], res[t * n + i]) << label << " residual-only path";
    }
    // Central differences of every column.
    const double h = 1e-6;
    std::vector<double> fd(n * cols);
    for (int c = 0; c < cols; ++c) {
      d.assign(cols, 0.0);
      d[c] = h;
      const auto rp = Residual(m, X[t], z[t], u[t], Y[t], zn[t], dts[t], d.data());
      d[c] = -h;
      const auto rm = Residual(m, X[t], z[t], u[t], Y[t], zn[t], dts[t], d.data());
      for (int i = 0; i < n; ++i) fd[i * cols + c] = (rp[i] - rm[i]) / (2 * h);
    }
    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < cols; ++c) {
        const double expected = fd[i * cols + c];
        EXPECT_NEAR(jac[(t * n + i) * cols + c], expected, 1e-4 * (1 + std::fabs(expected)))
            << label << " item " << t << " row " << i << " col " << c;
      }
    }
  }
}

TEST(SE2DifferentialDriveFactorBatch, MatchesReference) {
  constexpr int kItems = 300;  // several thread blocks and a partial one
  std::vector<float> dts(kItems);
  for (int t = 0; t < kItems; ++t) dts[t] = 0.05f + 0.05f * static_cast<float>(t % 4);
  dvector<float> d_dts(dts);
  SE2DifferentialDriveFactorBatch factor(d_dts.data(), float(kR), float(kB), kItems);
  EXPECT_EQ(factor.ResidualsSize(), 3u);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{3, 2, 3}));
  CheckAgainstReference(Carter(), factor, "carter");
}

TEST(SE2KinematicBicycleFactorBatch, MatchesReference) {
  constexpr int kItems = 300;  // several thread blocks and a partial one
  std::vector<float> dts(kItems);
  for (int t = 0; t < kItems; ++t) dts[t] = 0.05f + 0.05f * static_cast<float>(t % 4);
  dvector<float> d_dts(dts);
  SE2KinematicBicycleFactorBatch factor(d_dts.data(), float(kL), kItems);
  EXPECT_EQ(factor.ResidualsSize(), 5u);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{3, 2, 2, 3, 2}));
  CheckAgainstReference(Car(), factor, "car");
}

TEST(SE2DifferentialDriveFactorBatch, RejectsInvalidArguments) {
  dvector<float> dts(std::vector<float>{0.1f});
  EXPECT_THROW(SE2DifferentialDriveFactorBatch(nullptr, 0.1f, 0.5f, 1), std::invalid_argument);
  EXPECT_THROW(SE2DifferentialDriveFactorBatch(dts.data(), 0.f, 0.5f, 1), std::invalid_argument);
  EXPECT_THROW(SE2KinematicBicycleFactorBatch(dts.data(), -1.f, 1), std::invalid_argument);
}

// A Carter drives from the origin to a goal pose: dynamics and goal as
// equality constraints, wheel-speed limits, minimal control effort.
TEST(SE2DifferentialDriveFactorBatch, ReachesGoalWithAugmentedLagrangianMinimizer) {
  constexpr int kSteps = 30;
  constexpr float kDt = 0.1f, kMaxWheelSpeed = 25.f;
  CudaStream stream;
  std::vector<float> init;
  for (int k = 0; k <= kSteps; ++k) {
    const auto p = ToFloat(Exp(0, 0, 0));
    init.insert(init.end(), p.begin(), p.end());
  }
  const Mat3 goal_pose = {
      std::cos(1.2), -std::sin(1.2), 1.5, std::sin(1.2), std::cos(1.2), 0.8, 0, 0, 1};
  dvector<float> poses(init), controls(std::vector<float>(kSteps * 2, 0.f));
  dvector<float> dts(std::vector<float>(kSteps, kDt)), goal(ToFloat(goal_pose));
  dvector<float> zero(std::vector<float>(kSteps * 2, 0.f));
  dvector<float> lo(std::vector<float>(kSteps * 2, -kMaxWheelSpeed)),
      hi(std::vector<float>(kSteps * 2, kMaxWheelSpeed));
  dvector<int> const_ids(std::vector<int>{0});

  SE2StateBatch pose_states(poses.data(), kSteps + 1, const_ids.data(), 1);  // x_0 fixed
  pose_states.SetNumActiveStates(kSteps + 1, 1);
  VectorStateBatch<2> control_states(controls.data(), kSteps);
  control_states.SetNumActiveStates(kSteps);

  SE2DifferentialDriveFactorBatch dynamics(dts.data(), float(kR), float(kB), kSteps);
  dynamics.SetNumActiveFactors(kSteps);
  ConstraintFactorBatch dynamics_constraint(&dynamics, ConstraintKind::kEquality);
  SE2PriorFactorBatch goal_prior(reinterpret_cast<const SE2Transform *>(goal.data()), 1);
  goal_prior.SetNumActiveFactors(1);
  ConstraintFactorBatch goal_constraint(&goal_prior, ConstraintKind::kEquality);
  WeightedFactorBatch<PriorVectorFactorBatch<2>> effort(
      0.1f, reinterpret_cast<const Vector<2> *>(zero.data()), kSteps);
  effort.SetNumActiveFactors(kSteps);
  BoundFactorBatch<2> limits(lo.data(), hi.data(), kSteps);
  limits.SetNumActiveFactors(kSteps);

  std::vector<float *> dyn_ptrs, control_ptrs;
  for (int k = 0; k < kSteps; ++k) {
    dyn_ptrs.push_back(pose_states.StateDevicePtr(k));
    dyn_ptrs.push_back(control_states.StateDevicePtr(k));
    dyn_ptrs.push_back(pose_states.StateDevicePtr(k + 1));
    control_ptrs.push_back(control_states.StateDevicePtr(k));
  }
  Problem problem;
  problem.AddStateBatch(&pose_states);
  problem.AddStateBatch(&control_states);
  problem.AddFactorBatch(&dynamics_constraint, dyn_ptrs);
  problem.AddFactorBatch(&goal_constraint, {pose_states.StateDevicePtr(kSteps)});
  problem.AddFactorBatch(&effort, control_ptrs);
  problem.AddFactorBatch(&limits, control_ptrs);

  MinimizerOptions options;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
  options.max_num_iterations = 100;
  GaussNewtonMinimizer inner(options);
  const auto summary = AugmentedLagrangianMinimizer(inner).Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LE(summary.max_violation, 1e-4f);

  // Rolling the solved wheel speeds out through the reference model reaches the goal.
  std::vector<float> u(kSteps * 2);
  controls.CopyToHost(u.data(), u.size());
  Mat3 X = Exp(0, 0, 0);
  const Model carter = Carter();
  for (int k = 0; k < kSteps; ++k) {
    const double uk[2] = {u[2 * k], u[2 * k + 1]};
    const auto xi = carter.twist(nullptr, uk);
    X = Mul(X, Exp(kDt * xi[0], kDt * xi[1], kDt * xi[2]));
    EXPECT_LE(std::fabs(u[2 * k]), kMaxWheelSpeed + 1e-3f);
    EXPECT_LE(std::fabs(u[2 * k + 1]), kMaxWheelSpeed + 1e-3f);
  }
  EXPECT_NEAR(X[2], 1.5, 2e-3);
  EXPECT_NEAR(X[5], 0.8, 2e-3);
  EXPECT_NEAR(std::atan2(X[3], X[0]), 1.2, 2e-3);
}

}  // namespace
}  // namespace cunls
