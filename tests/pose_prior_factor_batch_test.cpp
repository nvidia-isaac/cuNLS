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

// SE2PriorFactorBatch and SE3PriorFactorBatch against a float64 reference:
// residual Log(T_target⁻¹ T), Jacobians by central differences, several
// thread blocks.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "dynamics_test_support.h"

namespace cunls {
namespace {

using namespace dynamics_test;

constexpr int kItems = 300;

TEST(SE2PriorFactorBatch, MatchesReference) {
  std::mt19937 rng(61);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<float> targets;
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    const double x[3] = {3 * normal(rng), 3 * normal(rng), normal(rng)};
    const double d[3] = {0.3 * normal(rng), 0.3 * normal(rng), 0.3 * normal(rng)};
    const Vec target = Exp3(x);
    for (double v : target) targets.push_back(static_cast<float>(v));
    items.push_back({{SlotKind::kSE2, ToFloat(Mul3(target, Exp3(d)))}});
  }
  dvector<float> d_targets(targets);
  SE2PriorFactorBatch factor(reinterpret_cast<const SE2Transform *>(d_targets.data()), kItems);
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        const Vec target(targets.begin() + 9 * t, targets.begin() + 9 * t + 9);
        const auto r = Log3(Mul3(Inv3(target), s[0]));
        return Vec(r.begin(), r.end());
      },
      "se2 prior", 2e-5, 2e-4);
}

TEST(SE3PriorFactorBatch, MatchesReference) {
  std::mt19937 rng(62);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<float> targets;
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    double x[6], d[6];
    for (double &v : x) v = 2 * normal(rng);
    for (double &v : d) v = 0.3 * normal(rng);
    const Vec target = Exp6(x);
    for (double v : target) targets.push_back(static_cast<float>(v));
    items.push_back({{SlotKind::kSE3, ToFloat(Mul4(target, Exp6(d)))}});
  }
  dvector<float> d_targets(targets);
  SE3PriorFactorBatch factor(reinterpret_cast<const SE3Transform *>(d_targets.data()), kItems);
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        const Vec target(targets.begin() + 16 * t, targets.begin() + 16 * t + 16);
        return Log6(Mul4(Inv4(target), s[0]));
      },
      "se3 prior", 2e-5, 2e-4);
}

}  // namespace
}  // namespace cunls
