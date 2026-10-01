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
 * @file evaluate_items_registration_test.cpp
 * @brief Evaluate with items (num_items, factor_ids) of the registration
 * factors and the weighting wrappers must give bitwise the same results as the
 * corresponding plain evaluations.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/information/information_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/point_to_plane_factor_batch.h"
#include "cunls/factor/point_to_point_factor_batch.h"
#include "cunls/factor/symmetric_point_to_plane_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "tests/evaluate_items_check.h"

namespace cunls {
namespace {

using evaluate_items_test::CheckEvaluateItems;
using evaluate_items_test::ToDevice;

constexpr int kNumFactors = 24;
constexpr int kCopies = 4;

cuBLASHandle &Cublas() {
  static cuBLASHandle handle;
  return handle;
}

/** Random SE(3) transform: rotation from a normalized random quaternion. */
SE3Transform RandomPose(std::mt19937 &rng, float rot, float trans) {
  std::normal_distribution<float> n(0.f, 1.f);
  float w = 1.f, x = rot * n(rng), y = rot * n(rng), z = rot * n(rng);
  const float norm = std::sqrt(w * w + x * x + y * y + z * z);
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;
  SE3Transform T{};
  T[0] = 1 - 2 * (y * y + z * z);
  T[1] = 2 * (x * y - w * z);
  T[2] = 2 * (x * z + w * y);
  T[4] = 2 * (x * y + w * z);
  T[5] = 1 - 2 * (x * x + z * z);
  T[6] = 2 * (y * z - w * x);
  T[8] = 2 * (x * z - w * y);
  T[9] = 2 * (y * z + w * x);
  T[10] = 1 - 2 * (x * x + y * y);
  T[3] = trans * n(rng);
  T[7] = trans * n(rng);
  T[11] = trans * n(rng);
  T[15] = 1.f;
  return T;
}

/** One distinct pose per (copy, factor) in device memory. */
struct Poses {
  dvector<SE3Transform> d;
  Poses(int count, uint32_t seed, float rot = 0.3f, float trans = 1.f) {
    std::mt19937 rng(seed);
    std::vector<SE3Transform> h;
    for (int i = 0; i < count; ++i) h.push_back(RandomPose(rng, rot, trans));
    d = ToDevice(h);
  }
  float *ptr(int i) { return reinterpret_cast<float *>(d.data() + i); }
  /** State pointers of copy k: one pose per factor. */
  std::vector<float *> ForCopy(int k, int num_factors) {
    std::vector<float *> p;
    for (int i = 0; i < num_factors; ++i) p.push_back(ptr(k * num_factors + i));
    return p;
  }
};

std::vector<Vector<3>> RandomPoints(int n, uint32_t seed, float scale = 2.f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.f, scale);
  std::vector<Vector<3>> v(n);
  for (auto &p : v) p = {d(rng), d(rng), d(rng)};
  return v;
}

std::vector<Vector<3>> RandomNormals(int n, uint32_t seed) {
  auto v = RandomPoints(n, seed, 1.f);
  for (auto &p : v) {
    const float norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    for (auto &c : p) c /= norm;
  }
  return v;
}

std::vector<float> RandomWeights(int n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(0.1f, 3.f);
  std::vector<float> w(n);
  for (float &x : w) x = d(rng);
  return w;
}

/** Random upper-triangular sqrt-information matrices with positive diagonal. */
template <int N>
std::vector<Matrix<N>> RandomSqrtInformation(int n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<Matrix<N>> v(n);
  for (auto &m : v) {
    m.fill(0.f);
    for (int i = 0; i < N; ++i) {
      m[i * N + i] = 0.5f + std::abs(d(rng)) * 2.f;
      for (int j = i + 1; j < N; ++j) m[i * N + j] = d(rng);
    }
  }
  return v;
}

/** PnP measurements: points in front of a camera near identity. */
struct PnPData {
  dvector<Vector<2>> obs;
  dvector<Vector<3>> pts;
  explicit PnPData(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xy(-1.f, 1.f), z(3.f, 8.f), o(-0.5f, 0.5f);
    std::vector<Vector<2>> ho(n);
    std::vector<Vector<3>> hp(n);
    for (int i = 0; i < n; ++i) {
      hp[i] = {xy(rng), xy(rng), z(rng)};
      ho[i] = {o(rng), o(rng)};
    }
    obs = ToDevice(ho);
    pts = ToDevice(hp);
  }
};

struct PointToPointData {
  dvector<Vector<3>> p, q;
  explicit PointToPointData(uint32_t seed)
      : p(ToDevice(RandomPoints(kNumFactors, seed))),
        q(ToDevice(RandomPoints(kNumFactors, seed + 1))) {}
};

TEST(EvaluateItemsRegistration, PointToPointMatchesEvaluate) {
  PointToPointData data(101);
  PointToPointFactorBatch factor(data.p.data(), data.q.data(), kNumFactors);
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 102);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

TEST(EvaluateItemsRegistration, PointToPlaneMatchesEvaluate) {
  auto p = ToDevice(RandomPoints(kNumFactors, 111));
  auto q = ToDevice(RandomPoints(kNumFactors, 112));
  auto nq = ToDevice(RandomNormals(kNumFactors, 113));
  PointToPlaneFactorBatch factor(p.data(), q.data(), nq.data(), kNumFactors);
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 114);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

TEST(EvaluateItemsRegistration, SymmetricPointToPlaneMatchesEvaluate) {
  auto p = ToDevice(RandomPoints(kNumFactors, 121));
  auto q = ToDevice(RandomPoints(kNumFactors, 122));
  auto np = ToDevice(RandomNormals(kNumFactors, 123));
  auto nq = ToDevice(RandomNormals(kNumFactors, 124));
  SymmetricPointToPlaneFactorBatch factor(p.data(), q.data(), np.data(), nq.data(), kNumFactors);
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 125);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

TEST(EvaluateItemsRegistration, WeightedUniformPointToPointMatchesEvaluate) {
  PointToPointData data(131);
  WeightedFactorBatch<PointToPointFactorBatch> factor(1.7f, data.p.data(), data.q.data(),
                                                      static_cast<size_t>(kNumFactors));
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 132);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

TEST(EvaluateItemsRegistration, WeightedPerFactorPointToPointMatchesEvaluate) {
  PointToPointData data(141);
  auto weights = ToDevice(RandomWeights(kNumFactors, 142));
  WeightedFactorBatch<PointToPointFactorBatch> factor(
      weights.data(), static_cast<size_t>(kNumFactors), data.p.data(), data.q.data(),
      static_cast<size_t>(kNumFactors));
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 143);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

TEST(EvaluateItemsRegistration, WeightedPerFactorPnPMatchesEvaluate) {
  PnPData data(kNumFactors, 151);
  auto weights = ToDevice(RandomWeights(kNumFactors, 152));
  WeightedFactorBatch<PnPFactorBatch> factor(weights.data(), static_cast<size_t>(kNumFactors),
                                             data.obs.data(), data.pts.data(),
                                             static_cast<size_t>(kNumFactors));
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 153, 0.05f, 0.1f);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

TEST(EvaluateItemsRegistration, InformationPointToPointMatchesEvaluate) {
  PointToPointData data(161);
  auto info = ToDevice(RandomSqrtInformation<3>(kNumFactors, 162));
  InformationFactorBatch<PointToPointFactorBatch> factor(
      Cublas(), info.data(), static_cast<size_t>(kNumFactors), data.p.data(), data.q.data(),
      static_cast<size_t>(kNumFactors));
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 163);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

TEST(EvaluateItemsRegistration, InformationPnPMatchesEvaluate) {
  PnPData data(kNumFactors, 171);
  auto info = ToDevice(RandomSqrtInformation<2>(kNumFactors, 172));
  InformationFactorBatch<PnPFactorBatch> factor(Cublas(), info.data(),
                                                static_cast<size_t>(kNumFactors), data.obs.data(),
                                                data.pts.data(), static_cast<size_t>(kNumFactors));
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 173, 0.05f, 0.1f);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

TEST(EvaluateItemsRegistration, InformationWeightedPointToPlaneMatchesEvaluate) {
  auto p = ToDevice(RandomPoints(kNumFactors, 181));
  auto q = ToDevice(RandomPoints(kNumFactors, 182));
  auto nq = ToDevice(RandomNormals(kNumFactors, 183));
  auto weights = ToDevice(RandomWeights(kNumFactors, 184));
  auto info = ToDevice(RandomSqrtInformation<1>(kNumFactors, 185));
  InformationFactorBatch<WeightedFactorBatch<PointToPlaneFactorBatch>> factor(
      Cublas(), info.data(), static_cast<size_t>(kNumFactors), weights.data(),
      static_cast<size_t>(kNumFactors), p.data(), q.data(), nq.data(),
      static_cast<size_t>(kNumFactors));
  factor.SetNumFactors(factor.Capacity());
  Poses poses(kCopies * kNumFactors, 186);
  CheckEvaluateItems(factor, kCopies, [&](int k) { return poses.ForCopy(k, kNumFactors); });
}

}  // namespace
}  // namespace cunls
