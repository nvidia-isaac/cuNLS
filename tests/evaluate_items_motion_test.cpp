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
 * @file evaluate_items_motion_test.cpp
 * @brief Evaluate with items (num_items, factor_ids) of the constant-velocity
 * and constant-acceleration motion factors must give bitwise the same results
 * as the corresponding plain evaluations.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "cunls/common/device_vector.h"
#include "cunls/factor/motion/constant_acceleration_se2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_se3_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so3_factor_batch.h"
#include "tests/evaluate_items_check.h"

namespace cunls {
namespace {

using evaluate_items_test::CheckEvaluateItems;
using evaluate_items_test::ToDevice;

constexpr size_t kNumFactors = 23;
constexpr int kCopies = 4;

enum class Group { kSO2, kSO3, kSE2, kSE3 };

size_t PoseAmbient(Group g) {
  switch (g) {
    case Group::kSO2:
      return 4;
    case Group::kSO3:
    case Group::kSE2:
      return 9;
    case Group::kSE3:
      return 16;
  }
  return 0;
}

size_t Tangent(Group g) {
  switch (g) {
    case Group::kSO2:
      return 1;
    case Group::kSO3:
    case Group::kSE2:
      return 3;
    case Group::kSE3:
      return 6;
  }
  return 0;
}

/** Row-major 3x3 rotation from an axis-angle vector (Rodrigues). */
void Rotation3(float wx, float wy, float wz, float *r) {
  const float th = std::sqrt(wx * wx + wy * wy + wz * wz);
  const float kx = wx / th, ky = wy / th, kz = wz / th;
  const float c = std::cos(th), s = std::sin(th), v = 1.f - c;
  r[0] = c + kx * kx * v;
  r[1] = kx * ky * v - kz * s;
  r[2] = kx * kz * v + ky * s;
  r[3] = ky * kx * v + kz * s;
  r[4] = c + ky * ky * v;
  r[5] = ky * kz * v - kx * s;
  r[6] = kz * kx * v - ky * s;
  r[7] = kz * ky * v + kx * s;
  r[8] = c + kz * kz * v;
}

/** Writes a random valid group element (row-major) to out. */
void RandomPose(Group g, std::mt19937 &rng, float *out) {
  std::uniform_real_distribution<float> angle(-2.5f, 2.5f);
  std::normal_distribution<float> trans(0.f, 2.f);
  switch (g) {
    case Group::kSO2: {
      const float a = angle(rng);
      out[0] = std::cos(a);
      out[1] = -std::sin(a);
      out[2] = std::sin(a);
      out[3] = std::cos(a);
      break;
    }
    case Group::kSO3:
      Rotation3(angle(rng), angle(rng), angle(rng) + 0.1f, out);
      break;
    case Group::kSE2: {
      const float a = angle(rng);
      const float c = std::cos(a), s = std::sin(a);
      const float m[9] = {c, -s, trans(rng), s, c, trans(rng), 0.f, 0.f, 1.f};
      std::copy(m, m + 9, out);
      break;
    }
    case Group::kSE3: {
      float r[9];
      Rotation3(angle(rng), angle(rng), angle(rng) + 0.1f, r);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) out[i * 4 + j] = r[i * 3 + j];
        out[i * 4 + 3] = trans(rng);
      }
      out[12] = out[13] = out[14] = 0.f;
      out[15] = 1.f;
      break;
    }
  }
}

/**
 * Builds kCopies distinct state sets for a motion factor with two pose blocks
 * followed by (num_blocks - 2) tangent-vector blocks (velocity[, acceleration])
 * and checks Evaluate's item contract on it.
 */
template <typename Factor>
void CheckMotionFactor(Group g, int num_blocks, uint32_t seed) {
  std::mt19937 rng(seed);
  const size_t pose = PoseAmbient(g);
  const size_t tangent = Tangent(g);
  // Storage (ambient) size of each block; StateBlockSizes() reports tangent sizes.
  std::vector<size_t> block_sizes(num_blocks, tangent);
  block_sizes[0] = block_sizes[1] = pose;
  size_t per_factor = 0;
  for (size_t s : block_sizes) per_factor += s;

  std::uniform_real_distribution<float> dt_dist(0.05f, 1.5f);
  std::vector<float> dt(kNumFactors);
  for (float &x : dt) x = dt_dist(rng);
  auto d_dt = ToDevice(dt);

  std::normal_distribution<float> vec(0.f, 1.f);
  std::vector<float> states(kCopies * kNumFactors * per_factor);
  for (size_t f = 0; f < kCopies * kNumFactors; ++f) {
    float *p = states.data() + f * per_factor;
    RandomPose(g, rng, p);
    RandomPose(g, rng, p + pose);
    for (size_t i = 2 * pose; i < per_factor; ++i) p[i] = vec(rng);
  }
  auto d_states = ToDevice(states);

  Factor factor(d_dt.data(), kNumFactors);
  factor.SetNumFactors(kNumFactors);
  ASSERT_EQ(factor.StateBlockSizes(), std::vector<size_t>(num_blocks, tangent));
  CheckEvaluateItems(factor, kCopies, [&](int k) {
    std::vector<float *> pointers;
    for (size_t f = 0; f < kNumFactors; ++f) {
      float *p = d_states.data() + (k * kNumFactors + f) * per_factor;
      for (size_t s : block_sizes) {
        pointers.push_back(p);
        p += s;
      }
    }
    return pointers;
  });
}

TEST(EvaluateItemsMotion, ConstantVelocitySO2) {
  CheckMotionFactor<ConstantVelocitySO2FactorBatch>(Group::kSO2, 4, 11);
}

TEST(EvaluateItemsMotion, ConstantVelocitySO3) {
  CheckMotionFactor<ConstantVelocitySO3FactorBatch>(Group::kSO3, 4, 12);
}

TEST(EvaluateItemsMotion, ConstantVelocitySE2) {
  CheckMotionFactor<ConstantVelocitySE2FactorBatch>(Group::kSE2, 4, 13);
}

TEST(EvaluateItemsMotion, ConstantVelocitySE3) {
  CheckMotionFactor<ConstantVelocitySE3FactorBatch>(Group::kSE3, 4, 14);
}

TEST(EvaluateItemsMotion, ConstantAccelerationSO2) {
  CheckMotionFactor<ConstantAccelerationSO2FactorBatch>(Group::kSO2, 6, 21);
}

TEST(EvaluateItemsMotion, ConstantAccelerationSO3) {
  CheckMotionFactor<ConstantAccelerationSO3FactorBatch>(Group::kSO3, 6, 22);
}

TEST(EvaluateItemsMotion, ConstantAccelerationSE2) {
  CheckMotionFactor<ConstantAccelerationSE2FactorBatch>(Group::kSE2, 6, 23);
}

TEST(EvaluateItemsMotion, ConstantAccelerationSE3) {
  CheckMotionFactor<ConstantAccelerationSE3FactorBatch>(Group::kSE3, 6, 24);
}

}  // namespace
}  // namespace cunls
