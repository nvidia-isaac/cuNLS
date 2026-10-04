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

// SE3DifferentialDriveFactorBatch and SE3KinematicBicycleFactorBatch against a
// float64 reference (tests/dynamics_test_support.h): residuals and Jacobians
// over several thread blocks, and the terrain semantics (roll and pitch
// changes are free).

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/factor/dynamics/se3_differential_drive_factor_batch.h"
#include "cunls/factor/dynamics/se3_kinematic_bicycle_factor_batch.h"
#include "dynamics_test_support.h"

namespace cunls {
namespace {

using namespace dynamics_test;

constexpr double kR = 0.125, kB = 0.5, kL = 2.75;  // exact in float
constexpr int kItems = 300;                        // several thread blocks and a partial one

float StepDuration(int t) { return 0.05f + 0.05f * static_cast<float>(t % 4); }

/** Rows 2..5 of Log((X Exp(dt ξ))⁻¹ Y) for ξ = [0, 0, ω, v, 0, 0]. */
Vec PoseRows(const Vec &X, const Vec &Y, double dt, double omega, double v) {
  const double step[6] = {0, 0, dt * omega, dt * v, 0, 0};
  const Vec e = Log6(Mul4(Inv4(Mul4(X, Exp6(step))), Y));
  return {e[2], e[3], e[4], e[5]};
}

Vec RandomPose(std::mt19937 &rng, double scale) {
  std::normal_distribution<double> normal(0.0, 1.0);
  double x[6];
  for (double &v : x) v = scale * normal(rng);
  return Exp6(x);
}

/** Next pose: the step itself (even items) or perturbed in all six directions (odd). */
Vec NextPose(const Vec &X, double dt, double omega, double v, int t, std::mt19937 &rng) {
  const double step[6] = {0, 0, dt * omega, dt * v, 0, 0};
  return Mul4(Mul4(X, Exp6(step)), RandomPose(rng, t % 2 == 0 ? 0.0 : 0.2));
}

TEST(SE3DifferentialDriveFactorBatch, MatchesReference) {
  std::mt19937 rng(9);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<float> dts(kItems);
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    dts[t] = StepDuration(t);
    const Vec X = RandomPose(rng, 1.0);
    const float wl = static_cast<float>(10 * normal(rng)),
                wr = static_cast<float>(10 * normal(rng));
    const double omega = kR / kB * (wr - wl), v = 0.5 * kR * (wr + wl);
    items.push_back({{SlotKind::kSE3, ToFloat(X)},
                     {SlotKind::kVector, {wl, wr}},
                     {SlotKind::kSE3, ToFloat(NextPose(X, dts[t], omega, v, t, rng))}});
  }
  dvector<float> d_dts(dts);
  SE3DifferentialDriveFactorBatch factor(d_dts.data(), float(kR), float(kB), kItems);
  EXPECT_EQ(factor.ResidualsSize(), 4u);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{6, 2, 6}));
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        const double omega = kR / kB * (s[1][1] - s[1][0]);
        const double v = 0.5 * kR * (s[1][1] + s[1][0]);
        return PoseRows(s[0], s[2], dts[t], omega, v);
      },
      "se3 carter");
}

TEST(SE3KinematicBicycleFactorBatch, MatchesReference) {
  std::mt19937 rng(10);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<float> dts(kItems);
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    dts[t] = StepDuration(t);
    const Vec X = RandomPose(rng, 1.0);
    const float v = static_cast<float>(3 * normal(rng)),
                delta = static_cast<float>(0.4 * normal(rng));
    const float a = static_cast<float>(normal(rng)), rate = static_cast<float>(normal(rng));
    const double omega = v * std::tan(delta) / kL;
    const float s = t % 2 == 0 ? 0.f : 0.2f;
    items.push_back({{SlotKind::kSE3, ToFloat(X)},
                     {SlotKind::kVector, {v, delta}},
                     {SlotKind::kVector, {a, rate}},
                     {SlotKind::kSE3, ToFloat(NextPose(X, dts[t], omega, v, t, rng))},
                     {SlotKind::kVector,
                      {v + dts[t] * a + s * static_cast<float>(normal(rng)),
                       delta + dts[t] * rate + s * static_cast<float>(normal(rng))}}});
  }
  dvector<float> d_dts(dts);
  SE3KinematicBicycleFactorBatch factor(d_dts.data(), float(kL), kItems);
  EXPECT_EQ(factor.ResidualsSize(), 6u);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{6, 2, 2, 6, 2}));
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        const double v = s[1][0], delta = s[1][1];
        Vec r = PoseRows(s[0], s[3], dts[t], v * std::tan(delta) / kL, v);
        r.push_back(s[4][0] - v - dts[t] * s[2][0]);
        r.push_back(s[4][1] - delta - dts[t] * s[2][1]);
        return r;
      },
      "se3 car");
}

// Driving over a crest: the next pose pitches (and rolls) about the vehicle
// origin relative to the step; the dynamics residual stays zero.
TEST(SE3DifferentialDriveFactorBatch, RollAndPitchAreFree) {
  const double dt = 0.1, wl = 6.0, wr = 9.0;
  const double omega = kR / kB * (wr - wl), v = 0.5 * kR * (wr + wl);
  const double tilt[6] = {0.05, -0.12, 0, 0, 0, 0};  // roll, pitch
  const Vec X = Exp6(std::vector<double>{0.1, 0.2, 0.3, 1.0, 2.0, 0.5}.data());
  const double step[6] = {0, 0, dt * omega, dt * v, 0, 0};
  const Vec Y = Mul4(Mul4(X, Exp6(step)), Exp6(tilt));
  std::vector<float> storage = ToFloat(X);
  const std::vector<float> u = {float(wl), float(wr)}, y = ToFloat(Y);
  storage.insert(storage.end(), u.begin(), u.end());
  storage.insert(storage.end(), y.begin(), y.end());
  CudaStream stream;
  dvector<float> d_storage(storage), d_dts(std::vector<float>{float(dt)}), d_res(4);
  dvector<float *> d_ptrs(
      std::vector<float *>{d_storage.data(), d_storage.data() + 16, d_storage.data() + 18});
  SE3DifferentialDriveFactorBatch factor(d_dts.data(), float(kR), float(kB), 1);
  factor.SetNumActiveFactors(1);
  ASSERT_TRUE(factor.Evaluate(d_res.data(), nullptr, d_ptrs.data(), stream.GetStream()));
  std::vector<float> res(4);
  d_res.CopyToHost(res.data(), 4);
  for (float r : res) EXPECT_NEAR(r, 0.f, 2e-6f);
}

TEST(SE3DifferentialDriveFactorBatch, RejectsInvalidArguments) {
  dvector<float> dts(std::vector<float>{0.1f});
  EXPECT_THROW(SE3DifferentialDriveFactorBatch(nullptr, 0.1f, 0.5f, 1), std::invalid_argument);
  EXPECT_THROW(SE3DifferentialDriveFactorBatch(dts.data(), 0.1f, -0.5f, 1), std::invalid_argument);
  EXPECT_THROW(SE3KinematicBicycleFactorBatch(dts.data(), 0.f, 1), std::invalid_argument);
}

}  // namespace
}  // namespace cunls
