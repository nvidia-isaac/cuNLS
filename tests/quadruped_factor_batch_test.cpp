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

// QuadrupedFactorBatch against a float64 reference (residuals, Jacobians with
// random stance/swing patterns, several thread blocks with a partial warp), and
// standing still as an equilibrium.

#include "cunls/factor/dynamics/quadruped_factor_batch.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "dynamics_test_support.h"

namespace cunls {
namespace {

using namespace dynamics_test;

QuadrupedParameters Params() {
  QuadrupedParameters p;
  p.mass = 12.5f;
  p.inertia[0] = 0.0625f;
  p.inertia[1] = 0.25f;
  p.inertia[2] = 0.25f;
  p.gravity = 9.8125f;
  return p;
}

/** Float64 reference of the residual (the factor's documentation). */
Vec Reference(const QuadrupedParameters &p, double dt, const float *contact, const float *feet,
              const std::vector<Vec> &s) {
  const Vec &X = s[0], &v = s[1], &w = s[2], &F = s[3], &Y = s[4], &vn = s[5], &wn = s[6];
  double rho[3];
  for (int i = 0; i < 3; ++i) rho[i] = X[i] * v[0] + X[4 + i] * v[1] + X[8 + i] * v[2];
  const double step[6] = {dt * w[0], dt * w[1], dt * w[2], dt * rho[0], dt * rho[1], dt * rho[2]};
  Vec r = Log6(Mul4(Inv4(Mul4(X, Exp6(step))), Y));
  double force[3] = {0, 0, 0}, torque_w[3] = {0, 0, 0};
  for (int k = 0; k < 4; ++k) {
    const double a[3] = {feet[3 * k] - X[3], feet[3 * k + 1] - X[7], feet[3 * k + 2] - X[11]};
    const double *f = &F[3 * k];
    const double c[3] = {a[1] * f[2] - a[2] * f[1], a[2] * f[0] - a[0] * f[2],
                         a[0] * f[1] - a[1] * f[0]};
    for (int i = 0; i < 3; ++i) {
      force[i] += contact[k] * f[i];
      torque_w[i] += contact[k] * c[i];
    }
  }
  const double J[3] = {p.inertia[0], p.inertia[1], p.inertia[2]};
  const double Jw[3] = {J[0] * w[0], J[1] * w[1], J[2] * w[2]};
  const double gyro[3] = {w[1] * Jw[2] - w[2] * Jw[1], w[2] * Jw[0] - w[0] * Jw[2],
                          w[0] * Jw[1] - w[1] * Jw[0]};
  for (int i = 0; i < 3; ++i)
    r.push_back(vn[i] - v[i] - dt * (force[i] / p.mass - (i == 2 ? p.gravity : 0.0)));
  for (int i = 0; i < 3; ++i) {
    const double tb = X[i] * torque_w[0] + X[4 + i] * torque_w[1] + X[8 + i] * torque_w[2];
    r.push_back(wn[i] - w[i] - dt * (tb - gyro[i]) / J[i]);
  }
  return r;
}

TEST(QuadrupedFactorBatch, MatchesReference) {
  constexpr int kItems = 300;
  const QuadrupedParameters p = Params();
  std::mt19937 rng(41);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_int_distribution<int> coin(0, 1);
  std::vector<float> dts(kItems), contacts(4 * kItems), feet(12 * kItems);
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    dts[t] = 0.02f + 0.01f * static_cast<float>(t % 3);
    for (int k = 0; k < 4; ++k) contacts[4 * t + k] = static_cast<float>(coin(rng));
    double x[6];
    for (double &c : x) c = 0.5 * normal(rng);
    const Vec X = Exp6(x);
    for (int k = 0; k < 4; ++k) {  // feet around the base, on the ground below it
      feet[12 * t + 3 * k] = static_cast<float>(X[3] + (k < 2 ? 0.2 : -0.2) + 0.05 * normal(rng));
      feet[12 * t + 3 * k + 1] =
          static_cast<float>(X[7] + (k % 2 ? 0.15 : -0.15) + 0.05 * normal(rng));
      feet[12 * t + 3 * k + 2] = static_cast<float>(X[11] - 0.3 + 0.05 * normal(rng));
    }
    std::vector<float> v(3), w(3), F(12), vn(3), wn(3);
    for (float &c : v) c = static_cast<float>(normal(rng));
    for (float &c : w) c = static_cast<float>(2 * normal(rng));
    for (int k = 0; k < 4; ++k) {
      F[3 * k] = static_cast<float>(10 * normal(rng));
      F[3 * k + 1] = static_cast<float>(10 * normal(rng));
      F[3 * k + 2] = static_cast<float>(30 + 10 * normal(rng));
    }
    for (float &c : vn) c = static_cast<float>(normal(rng));
    for (float &c : wn) c = static_cast<float>(normal(rng));
    double noise[6];
    for (double &c : noise) c = 0.1 * normal(rng);
    items.push_back({{SlotKind::kSE3, ToFloat(X)},
                     {SlotKind::kVector, v},
                     {SlotKind::kVector, w},
                     {SlotKind::kVector, F},
                     {SlotKind::kSE3, ToFloat(Mul4(X, Exp6(noise)))},
                     {SlotKind::kVector, vn},
                     {SlotKind::kVector, wn}});
  }
  dvector<float> d_dts(dts), d_contacts(contacts), d_feet(feet);
  QuadrupedFactorBatch factor(d_dts.data(), d_contacts.data(), d_feet.data(), p, kItems);
  EXPECT_EQ(factor.ResidualsSize(), 12u);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{6, 3, 3, 12, 6, 3, 3}));
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        return Reference(p, dts[t], contacts.data() + 4 * t, feet.data() + 12 * t, s);
      },
      "quadruped", 5e-5, 1e-4);
}

TEST(QuadrupedFactorBatch, StandingIsAnEquilibrium) {
  // Base at (1, 2, 0.4), feet symmetric below it, each leg carries m g / 4.
  const QuadrupedParameters p = Params();
  const std::vector<float> T = {1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0.4f, 0, 0, 0, 1};
  const float load = p.mass * p.gravity / 4.f;
  const std::vector<float> F = {0, 0, load, 0, 0, load, 0, 0, load, 0, 0, load};
  const std::vector<float> zero = {0, 0, 0};
  std::vector<float> storage;
  std::vector<size_t> offsets;
  for (const auto *s : {&T, &zero, &zero, &F, &T, &zero, &zero}) {
    offsets.push_back(storage.size());
    storage.insert(storage.end(), s->begin(), s->end());
  }
  // Exact in float, so the lever arms cancel exactly.
  const std::vector<float> feet = {1.25f, 1.875f, 0, 1.25f, 2.125f, 0,
                                   0.75f, 1.875f, 0, 0.75f, 2.125f, 0};
  CudaStream stream;
  dvector<float> d_storage(storage), d_dts(std::vector<float>{0.02f}),
      d_contacts(std::vector<float>{1, 1, 1, 1}), d_feet(feet), d_res(12);
  std::vector<float *> ptrs;
  for (size_t o : offsets) ptrs.push_back(d_storage.data() + o);
  dvector<float *> d_ptrs(ptrs);
  QuadrupedFactorBatch factor(d_dts.data(), d_contacts.data(), d_feet.data(), p, 1);
  factor.SetNumActiveFactors(1);
  ASSERT_TRUE(factor.Evaluate(d_res.data(), nullptr, d_ptrs.data(), stream.GetStream()));
  std::vector<float> res(12);
  d_res.CopyToHost(res.data(), 12);
  for (float r : res) EXPECT_NEAR(r, 0.f, 2e-6f);
}

TEST(QuadrupedFactorBatch, RejectsInvalidArguments) {
  dvector<float> b(std::vector<float>(12, 0.f));
  EXPECT_THROW(QuadrupedFactorBatch(b.data(), nullptr, b.data(), Params(), 1),
               std::invalid_argument);
  QuadrupedParameters p = Params();
  p.inertia[1] = 0.f;
  EXPECT_THROW(QuadrupedFactorBatch(b.data(), b.data(), b.data(), p, 1), std::invalid_argument);
}

}  // namespace
}  // namespace cunls
