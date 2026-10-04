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

// QuadrotorFactorBatch against a float64 reference (residuals, Jacobians,
// several thread blocks with a partial warp), hover equilibrium, and the
// rotor-layout signs.

#include "cunls/factor/dynamics/quadrotor_factor_batch.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "dynamics_test_support.h"

namespace cunls {
namespace {

using namespace dynamics_test;

QuadrotorParameters Params() {
  QuadrotorParameters p;
  p.mass = 1.25f;
  p.inertia[0] = 0.0125f;
  p.inertia[1] = 0.015625f;
  p.inertia[2] = 0.025f;
  p.arm_length = 0.25f;
  p.torque_coefficient = 0.015625f;
  p.linear_drag = 0.125f;
  p.gravity = 9.8125f;
  return p;
}

/** Float64 reference of the residual (same conventions as the factor's documentation). */
Vec Reference(const QuadrotorParameters &p, double dt, const std::vector<Vec> &s) {
  const Vec &X = s[0], &v = s[1], &w = s[2], &f = s[3], &Y = s[4], &vn = s[5], &wn = s[6];
  double rho[3];
  for (int i = 0; i < 3; ++i) rho[i] = X[i] * v[0] + X[4 + i] * v[1] + X[8 + i] * v[2];
  const double step[6] = {dt * w[0], dt * w[1], dt * w[2], dt * rho[0], dt * rho[1], dt * rho[2]};
  Vec r = Log6(Mul4(Inv4(Mul4(X, Exp6(step))), Y));
  const double T = f[0] + f[1] + f[2] + f[3];
  const double a = p.arm_length / std::sqrt(2.0);
  // Rotor positions (x, y) and yaw signs of the X layout.
  const double px[4] = {a, -a, a, -a}, py[4] = {-a, a, a, -a}, sigma[4] = {-1, -1, 1, 1};
  double tau[3] = {0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    tau[0] += py[i] * f[i];
    tau[1] += -px[i] * f[i];
    tau[2] += sigma[i] * p.torque_coefficient * f[i];
  }
  const double J[3] = {p.inertia[0], p.inertia[1], p.inertia[2]};
  const double Jw[3] = {J[0] * w[0], J[1] * w[1], J[2] * w[2]};
  const double gyro[3] = {w[1] * Jw[2] - w[2] * Jw[1], w[2] * Jw[0] - w[0] * Jw[2],
                          w[0] * Jw[1] - w[1] * Jw[0]};
  for (int i = 0; i < 3; ++i) {
    const double accel =
        X[i * 4 + 2] * T / p.mass - p.linear_drag * v[i] - (i == 2 ? p.gravity : 0.0);
    r.push_back(vn[i] - v[i] - dt * accel);
  }
  for (int i = 0; i < 3; ++i) r.push_back(wn[i] - w[i] - dt * (tau[i] - gyro[i]) / J[i]);
  return r;
}

std::vector<float> RandomVector(std::mt19937 &rng, int n, double scale, double offset = 0) {
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<float> v(n);
  for (float &x : v) x = static_cast<float>(offset + scale * normal(rng));
  return v;
}

TEST(QuadrotorFactorBatch, MatchesReference) {
  constexpr int kItems = 300;  // several blocks, a partial warp at the end
  const QuadrotorParameters p = Params();
  std::mt19937 rng(31);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<float> dts(kItems);
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    dts[t] = 0.01f + 0.005f * static_cast<float>(t % 4);
    double x[6];
    for (double &c : x) c = normal(rng);
    const Vec X = Exp6(x);
    const auto v = RandomVector(rng, 3, 2.0), w = RandomVector(rng, 3, 3.0),
               f = RandomVector(rng, 4, 1.0, 3.0);
    // Next states: the Euler step (even items, residual ~0) or perturbed (odd).
    std::vector<Vec> s = {X,
                          Vec(v.begin(), v.end()),
                          Vec(w.begin(), w.end()),
                          Vec(f.begin(), f.end()),
                          X,
                          Vec(3, 0.0),
                          Vec(3, 0.0)};
    const Vec r = Reference(p, dts[t], s);  // with Y = X, vn = wn = 0
    double rho[3], step[6];
    for (int i = 0; i < 3; ++i) rho[i] = X[i] * v[0] + X[4 + i] * v[1] + X[8 + i] * v[2];
    for (int i = 0; i < 3; ++i) {
      step[i] = dts[t] * w[i];
      step[3 + i] = dts[t] * rho[i];
    }
    const double sigma = t % 2 == 0 ? 0.0 : 0.1;
    double noise[6];
    for (double &c : noise) c = sigma * normal(rng);
    const Vec Y = Mul4(Mul4(X, Exp6(step)), Exp6(noise));
    std::vector<float> vn(3), wn(3);
    for (int i = 0; i < 3; ++i) {
      vn[i] = static_cast<float>(-r[6 + i] + sigma * normal(rng));  // r = vn - (...) at vn = 0
      wn[i] = static_cast<float>(-r[9 + i] + sigma * normal(rng));
    }
    items.push_back({{SlotKind::kSE3, ToFloat(X)},
                     {SlotKind::kVector, v},
                     {SlotKind::kVector, w},
                     {SlotKind::kVector, f},
                     {SlotKind::kSE3, ToFloat(Y)},
                     {SlotKind::kVector, vn},
                     {SlotKind::kVector, wn}});
  }
  dvector<float> d_dts(dts);
  QuadrotorFactorBatch factor(d_dts.data(), p, kItems);
  EXPECT_EQ(factor.ResidualsSize(), 12u);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{6, 3, 3, 4, 6, 3, 3}));
  CheckFactor(
      factor, items, [&](int t, const std::vector<Vec> &s) { return Reference(p, dts[t], s); },
      "quadrotor", 5e-5, 1e-4);
}

/** Evaluates one item on the GPU; returns its 12 residuals. */
std::vector<float> EvaluateOne(const QuadrotorParameters &p, float dt,
                               const std::vector<std::vector<float>> &slots) {
  std::vector<float> storage;
  std::vector<size_t> offsets;
  for (const auto &s : slots) {
    offsets.push_back(storage.size());
    storage.insert(storage.end(), s.begin(), s.end());
  }
  CudaStream stream;
  dvector<float> d_storage(storage), d_dts(std::vector<float>{dt}), d_res(12);
  std::vector<float *> ptrs;
  for (size_t o : offsets) ptrs.push_back(d_storage.data() + o);
  dvector<float *> d_ptrs(ptrs);
  QuadrotorFactorBatch factor(d_dts.data(), p, 1);
  factor.SetNumActiveFactors(1);
  EXPECT_TRUE(factor.Evaluate(d_res.data(), nullptr, d_ptrs.data(), stream.GetStream()));
  std::vector<float> res(12);
  d_res.CopyToHost(res.data(), 12);
  return res;
}

TEST(QuadrotorFactorBatch, HoverIsAnEquilibrium) {
  QuadrotorParameters p = Params();
  p.linear_drag = 0.f;
  const std::vector<float> I = {1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1};
  const float hover = p.mass * p.gravity / 4.f;
  const auto res = EvaluateOne(
      p, 0.02f, {I, {0, 0, 0}, {0, 0, 0}, {hover, hover, hover, hover}, I, {0, 0, 0}, {0, 0, 0}});
  for (float r : res) EXPECT_NEAR(r, 0.f, 1e-6f);
}

TEST(QuadrotorFactorBatch, RotorLayoutSigns) {
  // More thrust on the left rotors (1: back-left, 2: front-left) rolls right
  // side down: positive torque about +x. More on the counter-clockwise rotors
  // (0, 1) yaws clockwise: negative torque about +z.
  QuadrotorParameters p = Params();
  const std::vector<float> I = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  const std::vector<float> zero = {0, 0, 0};
  auto rate_change = [&](const std::vector<float> &f) {
    const auto res = EvaluateOne(p, 0.01f, {I, zero, zero, f, I, zero, zero});
    return std::vector<float>{-res[9], -res[10], -res[11]};  // dt ω̇ (ω_{k+1} = 0)
  };
  const auto roll = rate_change({1, 2, 2, 1});
  EXPECT_GT(roll[0], 0.f);
  EXPECT_NEAR(roll[1], 0.f, 1e-7f);
  const auto yaw = rate_change({2, 2, 1, 1});
  EXPECT_LT(yaw[2], 0.f);
  EXPECT_NEAR(yaw[0], 0.f, 1e-7f);
}

TEST(QuadrotorFactorBatch, RejectsInvalidArguments) {
  dvector<float> dts(std::vector<float>{0.01f});
  EXPECT_THROW(QuadrotorFactorBatch(nullptr, Params(), 1), std::invalid_argument);
  QuadrotorParameters p = Params();
  p.mass = 0.f;
  EXPECT_THROW(QuadrotorFactorBatch(dts.data(), p, 1), std::invalid_argument);
}

}  // namespace
}  // namespace cunls
