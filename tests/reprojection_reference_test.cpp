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

// ReprojectionFactorBatch and PnPFactorBatch against a float64 reference:
// world_from_rig pose states (right perturbation in the rig frame), with and
// without camera_from_rig extrinsics, residuals and Jacobians by central
// differences over several thread blocks; and invariance of the residual to
// where the world origin is.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "dynamics_test_support.h"

namespace cunls {
namespace {

using namespace dynamics_test;

constexpr int kItems = 300;

/** Normalized projection of world point P through world_from_rig T and camera_from_rig E. */
Vec Project(const Vec &T, const Vec &E, const double *P) {
  const Vec C = Mul4(E, Inv4(T));  // camera_from_world
  double c[3];
  for (int i = 0; i < 3; ++i)
    c[i] = C[i * 4] * P[0] + C[i * 4 + 1] * P[1] + C[i * 4 + 2] * P[2] + C[i * 4 + 3];
  return {c[0] / c[2], c[1] / c[2]};
}

/** A random scene item: rig pose, extrinsic, a point 2..8 m in front of the camera, observation. */
struct Item {
  Vec T, E, P, obs;
};

Item RandomItem(std::mt19937 &rng, bool extrinsic, double origin_offset) {
  std::normal_distribution<double> n(0.0, 1.0);
  std::uniform_real_distribution<double> depth(2.0, 8.0);
  double x[6];
  for (int i = 0; i < 3; ++i) x[i] = 0.7 * n(rng);
  for (int i = 3; i < 6; ++i) x[i] = 3.0 * n(rng) + origin_offset;
  Item it;
  it.T = Exp6(x);
  double e[6] = {0, 0, 0, 0, 0, 0};
  if (extrinsic)
    for (double &v : e) v = 0.3 * n(rng);
  it.E = Exp6(e);
  // Point in the camera frame, mapped to the world.
  const double pc[3] = {0.5 * n(rng), 0.5 * n(rng), depth(rng)};
  const Vec W = Mul4(it.T, Inv4(it.E));  // world_from_camera
  it.P.resize(3);
  for (int i = 0; i < 3; ++i)
    it.P[i] = W[i * 4] * pc[0] + W[i * 4 + 1] * pc[1] + W[i * 4 + 2] * pc[2] + W[i * 4 + 3];
  it.obs = Project(it.T, it.E, it.P.data());
  for (double &o : it.obs) o += 0.01 * n(rng);
  return it;
}

class ProjectionReferenceTest : public ::testing::TestWithParam<bool> {};

TEST_P(ProjectionReferenceTest, ReprojectionMatchesReference) {
  const bool extrinsic = GetParam();
  std::mt19937 rng(71);
  std::vector<Item> scene;
  std::vector<float> obs, extr;
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    scene.push_back(RandomItem(rng, extrinsic, 0.0));
    const Item &it = scene.back();
    obs.push_back(static_cast<float>(it.obs[0]));
    obs.push_back(static_cast<float>(it.obs[1]));
    for (double v : it.E) extr.push_back(static_cast<float>(v));
    items.push_back({{SlotKind::kSE3, ToFloat(it.T)}, {SlotKind::kVector, ToFloat(it.P)}});
  }
  dvector<float> d_obs(obs), d_extr(extr);
  auto *O = reinterpret_cast<const Vector<2> *>(d_obs.data());
  ReprojectionFactorBatch factor =
      extrinsic ? ReprojectionFactorBatch(O, reinterpret_cast<const SE3Transform *>(d_extr.data()),
                                          kItems)
                : ReprojectionFactorBatch(O, kItems);
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        const Vec p = Project(s[0], scene[t].E, s[1].data());
        return Vec{p[0] - obs[2 * t], p[1] - obs[2 * t + 1]};
      },
      std::string("reprojection") + (extrinsic ? " + extrinsic" : ""), 2e-5, 2e-4);
}

TEST_P(ProjectionReferenceTest, PnPMatchesReference) {
  const bool extrinsic = GetParam();
  std::mt19937 rng(72);
  std::vector<Item> scene;
  std::vector<float> obs, extr, points;
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    scene.push_back(RandomItem(rng, extrinsic, 0.0));
    const Item &it = scene.back();
    obs.push_back(static_cast<float>(it.obs[0]));
    obs.push_back(static_cast<float>(it.obs[1]));
    for (double v : it.E) extr.push_back(static_cast<float>(v));
    for (double v : it.P) points.push_back(static_cast<float>(v));
    items.push_back({{SlotKind::kSE3, ToFloat(it.T)}});
  }
  dvector<float> d_obs(obs), d_extr(extr), d_points(points);
  auto *O = reinterpret_cast<const Vector<2> *>(d_obs.data());
  auto *P = reinterpret_cast<const Vector<3> *>(d_points.data());
  PnPFactorBatch factor =
      extrinsic
          ? PnPFactorBatch(O, reinterpret_cast<const SE3Transform *>(d_extr.data()), P, kItems)
          : PnPFactorBatch(O, P, kItems);
  CheckFactor(
      factor, items,
      [&](int t, const std::vector<Vec> &s) {
        const double Pw[3] = {points[3 * t], points[3 * t + 1], points[3 * t + 2]};
        const Vec p = Project(s[0], scene[t].E, Pw);
        return Vec{p[0] - obs[2 * t], p[1] - obs[2 * t + 1]};
      },
      std::string("pnp") + (extrinsic ? " + extrinsic" : ""), 2e-5, 2e-4);
}

INSTANTIATE_TEST_SUITE_P(Extrinsic, ProjectionReferenceTest, ::testing::Bool());

// Moving the scene 1 km from the world origin changes neither the residuals
// nor the rig-frame Jacobian (beyond float32 round-off of the coordinates).
TEST(ReprojectionFactorBatch, InvariantToWorldOrigin) {
  std::mt19937 rng(73);
  const Item it = RandomItem(rng, false, 0.0);
  std::vector<float> res[2], jac[2];
  for (int moved = 0; moved < 2; ++moved) {
    Vec T = it.T, P = it.P;
    if (moved) {
      const double shift[3] = {1000.0, -700.0, 250.0};
      for (int i = 0; i < 3; ++i) {
        T[i * 4 + 3] += shift[i];
        P[i] += shift[i];
      }
    }
    std::vector<float> storage = ToFloat(T);
    const std::vector<float> p = ToFloat(P);
    storage.insert(storage.end(), p.begin(), p.end());
    dvector<float> d_storage(storage), d_obs(ToFloat(it.obs)), d_res(2), d_jac(18);
    dvector<float *> d_ptrs(std::vector<float *>{d_storage.data(), d_storage.data() + 16});
    ReprojectionFactorBatch factor(reinterpret_cast<const Vector<2> *>(d_obs.data()), 1);
    factor.SetNumActiveFactors(1);
    CudaStream stream;
    ASSERT_TRUE(factor.Evaluate(d_res.data(), d_jac.data(), d_ptrs.data(), stream.GetStream()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    res[moved].resize(2);
    jac[moved].resize(18);
    d_res.CopyToHost(res[moved].data(), 2);
    d_jac.CopyToHost(jac[moved].data(), 18);
  }
  // float32 coordinates near 1 km resolve ~6e-5 m; at 2..8 m depth that is ~3e-5 in the image.
  for (int i = 0; i < 2; ++i) EXPECT_NEAR(res[1][i], res[0][i], 1e-4f) << i;
  for (int i = 0; i < 18; ++i)
    EXPECT_NEAR(jac[1][i], jac[0][i], 1e-3f * (1 + std::fabs(jac[0][i]))) << i;
}

}  // namespace
}  // namespace cunls
