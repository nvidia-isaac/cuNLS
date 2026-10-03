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

// SE2DiskClearanceFactorBatch and SE3SphereClearanceFactorBatch against a
// float64 reference (residuals, Jacobians, several thread blocks).

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/factor/clearance/se2_disk_clearance_factor_batch.h"
#include "cunls/factor/clearance/se3_sphere_clearance_factor_batch.h"
#include "dynamics_test_support.h"

namespace cunls {
namespace {

using namespace dynamics_test;

constexpr int kItems = 300;
constexpr float kMargin = 0.25f;

TEST(SE2DiskClearanceFactorBatch, MatchesReference) {
  std::mt19937 rng(51);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<float> obstacles;
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    const double x[3] = {2 * normal(rng), 2 * normal(rng), normal(rng)};
    items.push_back({{SlotKind::kSE2, ToFloat(Exp3(x))}});
    obstacles.push_back(static_cast<float>(normal(rng)));
    obstacles.push_back(static_cast<float>(normal(rng)));
    obstacles.push_back(static_cast<float>(0.5 + 0.2 * normal(rng)));
  }
  dvector<float> d_obstacles(obstacles);
  SE2DiskClearanceFactorBatch factor(d_obstacles.data(), kMargin, kItems);
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        const float *o = obstacles.data() + 3 * t;
        const double dx = s[0][2] - o[0], dy = s[0][5] - o[1];
        return Vec{o[2] + kMargin - std::sqrt(dx * dx + dy * dy)};
      },
      "se2 disk");
}

TEST(SE3SphereClearanceFactorBatch, MatchesReference) {
  std::mt19937 rng(52);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<float> obstacles;
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    double x[6];
    for (double &c : x) c = 1.5 * normal(rng);
    items.push_back({{SlotKind::kSE3, ToFloat(Exp6(x))}});
    for (int i = 0; i < 3; ++i) obstacles.push_back(static_cast<float>(normal(rng)));
    obstacles.push_back(static_cast<float>(0.5 + 0.2 * normal(rng)));
  }
  dvector<float> d_obstacles(obstacles);
  SE3SphereClearanceFactorBatch factor(d_obstacles.data(), kMargin, kItems);
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        const float *o = obstacles.data() + 4 * t;
        const double d[3] = {s[0][3] - o[0], s[0][7] - o[1], s[0][11] - o[2]};
        return Vec{o[3] + kMargin - std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])};
      },
      "se3 sphere");
}

TEST(ClearanceFactorBatch, RejectsNullObstacles) {
  EXPECT_THROW(SE2DiskClearanceFactorBatch(nullptr, 0.f, 1), std::invalid_argument);
  EXPECT_THROW(SE3SphereClearanceFactorBatch(nullptr, 0.f, 1), std::invalid_argument);
}

}  // namespace
}  // namespace cunls
