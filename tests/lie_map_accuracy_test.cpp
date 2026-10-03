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

// Accuracy of the batched exp / log kernels in float32 across scales, from
// near-identity elements (where series and closed forms switch) to large
// ones: SE2 exp and log against a double-precision closed form, and
// log(exp(xi)) round trips for SE2, Sim3 and SL4.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/math/sim_lie_math.h"
#include "cunls/math/sl_lie_math.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {
namespace {

constexpr size_t kN = 512;

std::vector<float> RandomTangents(size_t dim, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.f, scale);
  std::vector<float> xi(kN * dim);
  for (auto &v : xi) v = normal(rng);
  return xi;
}

/**
 * Scales rotation parts to angles below 2.8 rad, where log is unique (beyond
 * pi it wraps by 2 pi). `offset`/`count`: the rotation components of each
 * tangent of size `dim`.
 */
void ClampRotations(std::vector<float> &xi, size_t dim, size_t offset, size_t count) {
  for (size_t i = 0; i < xi.size() / dim; ++i) {
    float *w = xi.data() + i * dim + offset;
    float norm = 0.f;
    for (size_t k = 0; k < count; ++k) norm += w[k] * w[k];
    norm = std::sqrt(norm);
    if (norm > 2.8f) {
      for (size_t k = 0; k < count; ++k) w[k] *= 2.8f / norm;
    }
  }
}

std::vector<float> ToHost(const dvector<float> &d) {
  std::vector<float> h(d.size());
  d.CopyToHost(h.data(), h.size());
  return h;
}

/** Max |a - b| over the arrays. */
double MaxError(const std::vector<float> &a, const std::vector<float> &b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(double(a[i]) - b[i]));
  return worst;
}

class LieMapAccuracy : public ::testing::TestWithParam<float> {};

TEST_P(LieMapAccuracy, SE2ExpAndLogMatchDoublePrecision) {
  const float scale = GetParam();
  CudaStream stream;
  std::vector<float> xi = RandomTangents(3, scale, 1);
  ClampRotations(xi, 3, 2, 1);
  // Reference in double: T = [R(w), V(w) v; 0 1], V = [[s/w, -(1-c)/w], [(1-c)/w, s/w]].
  std::vector<float> ref(kN * 9);
  for (size_t i = 0; i < kN; ++i) {
    const double vx = xi[3 * i], vy = xi[3 * i + 1], w = xi[3 * i + 2];
    const double c = std::cos(w), s = std::sin(w);
    const double a = std::fabs(w) < 1e-8 ? 1.0 - w * w / 6.0 : s / w;
    const double b = std::fabs(w) < 1e-8 ? w / 2.0 : (1.0 - c) / w;
    const double t[9] = {c, -s, a * vx - b * vy, s, c, b * vx + a * vy, 0.0, 0.0, 1.0};
    std::copy(t, t + 9, ref.begin() + 9 * i);
  }
  dvector<float> d_xi(xi), d_t(kN * 9), d_log(kN * 3);
  ComputeExpSE2(stream.GetStream(), d_xi.data(), 3, 9, kN, d_t.data());
  ComputeLogSE2(stream.GetStream(), d_t.data(), 9, 3, kN, d_log.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const float tol = 2e-6f * std::max(1.f, scale);
  EXPECT_LT(MaxError(ToHost(d_t), ref), tol) << "exp, scale " << scale;
  EXPECT_LT(MaxError(ToHost(d_log), xi), 5 * tol) << "log(exp), scale " << scale;
}

TEST_P(LieMapAccuracy, Sim3LogInvertsExp) {
  const float scale = GetParam();
  CudaStream stream;
  std::vector<float> xi = RandomTangents(7, scale, 2);
  ClampRotations(xi, 7, 0, 3);
  dvector<float> d_xi(xi), d_t(kN * 16), d_log(kN * 7);
  ComputeExpSim3(stream.GetStream(), d_xi.data(), 7, 16, kN, d_t.data());
  ComputeLogSim3(stream.GetStream(), d_t.data(), 16, 7, kN, d_log.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_LT(MaxError(ToHost(d_log), xi), 2e-5 * std::max(1.f, scale)) << "scale " << scale;
}

TEST_P(LieMapAccuracy, SL4LogInvertsExp) {
  const float scale = std::min(GetParam(), 0.5f);  // keep exp(xi) well inside the log's domain
  CudaStream stream;
  std::vector<float> xi = RandomTangents(15, scale, 3);
  dvector<float> d_xi(xi), d_t(kN * 16), d_log(kN * 15);
  ComputeExpSL4(stream.GetStream(), d_xi.data(), 15, 4, 16, kN, d_t.data());
  ComputeLogSL4(stream.GetStream(), d_t.data(), 4, 16, 15, kN, d_log.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  // Before the Denman-Beavers fix the matrix square root (plain Newton) was
  // unstable and the log was off by up to 5 for ~2% of elements at scale 0.5.
  EXPECT_LT(MaxError(ToHost(d_log), xi), 5e-4) << "scale " << scale;
}

INSTANTIATE_TEST_SUITE_P(Scales, LieMapAccuracy, ::testing::Values(1e-4f, 1e-2f, 0.3f, 1.0f));

}  // namespace
}  // namespace cunls
