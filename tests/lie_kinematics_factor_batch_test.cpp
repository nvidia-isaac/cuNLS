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

// SE2/SO3/SE3KinematicsFactorBatch (X_{k+1} = X_k Exp(dt ξ_k)) against a
// float64 reference (tests/dynamics_test_support.h), over several thread
// blocks, with small and large twists.

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "cunls/factor/dynamics/se2_kinematics_factor_batch.h"
#include "cunls/factor/dynamics/se3_kinematics_factor_batch.h"
#include "cunls/factor/dynamics/so3_kinematics_factor_batch.h"
#include "dynamics_test_support.h"

namespace cunls {
namespace {

using namespace dynamics_test;

constexpr int kItems = 300;  // several thread blocks and a partial one

/** Group operations of the float64 reference for one kinematics factor. */
struct Group {
  SlotKind kind;
  int n;
  std::function<Vec(const double *)> exp;
  std::function<Vec(const Vec &)> log;
  std::function<Vec(const Vec &, const Vec &)> mul;
  std::function<Vec(const Vec &)> inv;
};

Group SE2() { return {SlotKind::kSE2, 3, Exp3, Log3, Mul3, Inv3}; }
Group SO3() { return {SlotKind::kSO3, 3, ExpSO3, LogSO3, Mul3, TransposeSO3}; }
Group SE3() { return {SlotKind::kSE3, 6, Exp6, Log6, Mul4, Inv4}; }

void Check(FactorBatch &factor, const Group &g, const std::vector<float> &dts, uint32_t seed,
           const std::string &label) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    Vec d(g.n), twist(g.n), noise(g.n);
    for (double &v : d) v = normal(rng);
    // Twists from tiny (series branches) to large.
    const double scale = t % 3 == 0 ? 1e-4 : t % 3 == 1 ? 0.5 : 4.0;
    for (double &v : twist) v = scale * normal(rng);
    for (double &v : noise) v = t % 2 == 0 ? 0.0 : 0.2 * normal(rng);
    const Vec X = g.exp(d.data());
    Vec step(g.n);
    for (int i = 0; i < g.n; ++i) step[i] = dts[t] * twist[i];
    const Vec Y = g.mul(g.mul(X, g.exp(step.data())), g.exp(noise.data()));
    items.push_back(
        {{g.kind, ToFloat(X)}, {SlotKind::kVector, ToFloat(twist)}, {g.kind, ToFloat(Y)}});
  }
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        Vec step(g.n);
        for (int i = 0; i < g.n; ++i) step[i] = dts[t] * s[1][i];
        return g.log(g.mul(g.inv(g.mul(s[0], g.exp(step.data()))), s[2]));
      },
      label);
}

std::vector<float> Steps() {
  std::vector<float> dts(kItems);
  for (int t = 0; t < kItems; ++t) dts[t] = 0.05f + 0.05f * static_cast<float>(t % 4);
  return dts;
}

TEST(SE2KinematicsFactorBatch, MatchesReference) {
  const auto dts = Steps();
  dvector<float> d_dts(dts);
  SE2KinematicsFactorBatch factor(d_dts.data(), kItems);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{3, 3, 3}));
  Check(factor, SE2(), dts, 21, "se2 kinematics");
}

TEST(SO3KinematicsFactorBatch, MatchesReference) {
  const auto dts = Steps();
  dvector<float> d_dts(dts);
  SO3KinematicsFactorBatch factor(d_dts.data(), kItems);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{3, 3, 3}));
  Check(factor, SO3(), dts, 22, "so3 kinematics");
}

TEST(SE3KinematicsFactorBatch, MatchesReference) {
  const auto dts = Steps();
  dvector<float> d_dts(dts);
  SE3KinematicsFactorBatch factor(d_dts.data(), kItems);
  EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{6, 6, 6}));
  Check(factor, SE3(), dts, 23, "se3 kinematics");
}

TEST(LieKinematicsFactorBatch, RejectsNullTimeSteps) {
  EXPECT_THROW(SE2KinematicsFactorBatch(nullptr, 1), std::invalid_argument);
  EXPECT_THROW(SO3KinematicsFactorBatch(nullptr, 1), std::invalid_argument);
  EXPECT_THROW(SE3KinematicsFactorBatch(nullptr, 1), std::invalid_argument);
}

}  // namespace
}  // namespace cunls
