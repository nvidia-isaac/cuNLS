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

// Accuracy of the analytic Jacobians of the Lie-group prior and between factors
// (SO3, SE3, SE2, Sim2, Sim3) at small residuals, where closed forms such as
// (t - sin t) / t^3 lose float32 precision. Reference: central differences of
// the factor's own residual along the state's Plus.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/similarity2_between_factor_batch.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/prior/similarity2_prior_factor_batch.h"
#include "cunls/factor/prior/similarity3_prior_factor_batch.h"
#include "cunls/factor/prior/so3_prior_factor_batch.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/so3_state_batch.h"

namespace cunls {
namespace {

constexpr size_t kN = 64;

std::vector<float> ToHost(const dvector<float> &d) {
  std::vector<float> h(d.size());
  d.CopyToHost(h.data(), h.size());
  return h;
}

/** Random elements Exp(sigma * xi) of a state batch, via its own Plus. */
template <class States>
dvector<float> RandomElements(size_t ambient, size_t tangent, float sigma, uint32_t seed,
                              cudaStream_t stream) {
  std::vector<float> identity(kN * ambient, 0.f);
  const int d = ambient == 9 ? 3 : 4;
  for (size_t i = 0; i < kN; ++i) {
    for (int k = 0; k < d; ++k) identity[i * ambient + k * d + k] = 1.f;
  }
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.f, sigma);
  std::vector<float> xi(kN * tangent);
  for (auto &v : xi) v = normal(rng);
  dvector<float> x(identity), delta(xi), out(kN * ambient);
  States states(x.data(), kN);
  states.SetNumActiveStates(kN);
  states.Plus(x.data(), delta.data(), out.data(), stream);
  return out;
}

/**
 * Max relative error of the factor's Jacobian against central differences.
 * `states` are B slot buffers of kN elements; each item reads element f of each.
 */
template <class States, class Factor>
float JacobianError(const Factor &factor, std::vector<dvector<float> *> states, size_t ambient,
                    size_t tangent, size_t m, cudaStream_t stream) {
  const size_t b = states.size(), n = b * tangent;
  auto pointers = [&](std::vector<dvector<float> *> s) {
    std::vector<float *> p;
    for (size_t f = 0; f < kN; ++f) {
      for (size_t k = 0; k < b; ++k) p.push_back(s[k]->data() + f * ambient);
    }
    return dvector<float *>(p);
  };
  dvector<float> residuals(kN * m), jacobians(kN * m * n);
  auto ptrs = pointers(states);
  factor.Evaluate(residuals.data(), jacobians.data(), ptrs.data(), stream);
  const auto jac = ToHost(jacobians);
  float worst = 0.f;
  const float h = 1e-3f;
  for (size_t slot = 0; slot < b; ++slot) {
    for (size_t k = 0; k < tangent; ++k) {
      float r_plus_minus[2][kN * 8];
      for (int sgn = 0; sgn < 2; ++sgn) {
        std::vector<float> d(kN * tangent, 0.f);
        for (size_t f = 0; f < kN; ++f) d[f * tangent + k] = sgn == 0 ? h : -h;
        dvector<float> delta(d), moved(kN * ambient);
        States s(states[slot]->data(), kN);
        s.SetNumActiveStates(kN);
        s.Plus(states[slot]->data(), delta.data(), moved.data(), stream);
        auto perturbed = states;
        perturbed[slot] = &moved;
        auto p2 = pointers(perturbed);
        dvector<float> r(kN * m);
        factor.Evaluate(r.data(), nullptr, p2.data(), stream);
        const auto rh = ToHost(r);
        std::copy(rh.begin(), rh.end(), r_plus_minus[sgn]);
      }
      for (size_t f = 0; f < kN; ++f) {
        for (size_t row = 0; row < m; ++row) {
          const float fd = (r_plus_minus[0][f * m + row] - r_plus_minus[1][f * m + row]) / (2 * h);
          const float got = jac[(f * m + row) * n + slot * tangent + k];
          worst = std::max(worst, std::fabs(got - fd));
        }
      }
    }
  }
  return worst;
}

class LieJacobianAccuracy : public ::testing::TestWithParam<float> {};

/**
 * Prior and between (Delta = I) Jacobian errors for d x d group matrices with
 * `tangent` coordinates: states X ~ Exp(N(0, state_scale)), measurements
 * Z = X * Exp(noise) with noise of size residual_scale. (Sim states use a
 * smaller state_scale: scales e^{3 sigma} make float32 central differences
 * noisier than the tolerance.)
 */
template <class States, class Prior, class Between, class Transform>
std::pair<float, float> PriorAndBetweenErrors(int d, size_t tangent, float residual_scale,
                                              float state_scale, uint32_t seed,
                                              cudaStream_t stream) {
  const size_t ambient = d * d;
  auto x = RandomElements<States>(ambient, tangent, state_scale, seed, stream);
  auto noise = RandomElements<States>(ambient, tangent, residual_scale, seed + 1, stream);
  dvector<float> z(kN * ambient);
  {
    auto xh = ToHost(x), nh = ToHost(noise);
    std::vector<float> zh(kN * ambient, 0.f);
    for (size_t f = 0; f < kN; ++f) {
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
          float acc = 0.f;
          for (int k = 0; k < d; ++k)
            acc += xh[f * ambient + i * d + k] * nh[f * ambient + k * d + j];
          zh[f * ambient + i * d + j] = acc;
        }
      }
    }
    z.CopyFromHost(zh.data(), zh.size());
  }
  Prior prior(reinterpret_cast<const Transform *>(z.data()), kN);
  prior.SetNumActiveFactors(kN);
  const float prior_error = JacobianError<States>(prior, {&x}, ambient, tangent, tangent, stream);
  std::vector<float> eye(kN * ambient, 0.f);
  for (size_t f = 0; f < kN; ++f) {
    for (int k = 0; k < d; ++k) eye[f * ambient + k * (d + 1)] = 1.f;
  }
  dvector<float> deltas(eye);
  Between between(reinterpret_cast<const Transform *>(deltas.data()), kN);
  between.SetNumActiveFactors(kN);
  const float between_error =
      JacobianError<States>(between, {&x, &z}, ambient, tangent, tangent, stream);
  return {prior_error, between_error};
}

TEST_P(LieJacobianAccuracy, SE2PriorAndBetween) {
  CudaStream stream;
  auto [prior, between] =
      PriorAndBetweenErrors<SE2StateBatch, SE2PriorFactorBatch, SE2BetweenFactorBatch,
                            SE2Transform>(3, 3, GetParam(), 1.0f, 11, stream.GetStream());
  EXPECT_LT(prior, 1e-3f) << "residual scale " << GetParam();
  EXPECT_LT(between, 1e-3f) << "residual scale " << GetParam();
}

TEST_P(LieJacobianAccuracy, Sim2PriorAndBetween) {
  CudaStream stream;
  auto [prior, between] =
      PriorAndBetweenErrors<Similarity2StateBatch, Similarity2PriorFactorBatch,
                            Similarity2BetweenFactorBatch, Similarity2Transform>(
          3, 4, GetParam(), 0.3f, 13, stream.GetStream());
  EXPECT_LT(prior, 1e-3f) << "residual scale " << GetParam();
  EXPECT_LT(between, 1e-3f) << "residual scale " << GetParam();
}

TEST_P(LieJacobianAccuracy, Sim3PriorAndBetween) {
  CudaStream stream;
  auto [prior, between] =
      PriorAndBetweenErrors<Similarity3StateBatch, Similarity3PriorFactorBatch,
                            Similarity3BetweenFactorBatch, Similarity3Transform>(
          4, 7, GetParam(), 0.3f, 15, stream.GetStream());
  EXPECT_LT(prior, 1e-3f) << "residual scale " << GetParam();
  EXPECT_LT(between, 1e-3f) << "residual scale " << GetParam();
}

TEST_P(LieJacobianAccuracy, SE3PriorAndBetween) {
  const float residual_scale = GetParam();
  CudaStream stream;
  auto x = RandomElements<SE3StateBatch>(16, 6, 1.0f, 1, stream.GetStream());
  auto noise = RandomElements<SE3StateBatch>(16, 6, residual_scale, 2, stream.GetStream());
  // Measurements close to the states: residuals of size ~residual_scale.
  dvector<float> z(kN * 16), y(kN * 16), identity_delta(kN * 6);
  {
    auto xh = ToHost(x), nh = ToHost(noise);
    std::vector<float> zh(kN * 16, 0.f);
    for (size_t f = 0; f < kN; ++f) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          float acc = 0.f;
          for (int k = 0; k < 4; ++k) acc += xh[f * 16 + i * 4 + k] * nh[f * 16 + k * 4 + j];
          zh[f * 16 + i * 4 + j] = acc;
        }
      }
    }
    z.CopyFromHost(zh.data(), zh.size());
  }
  SE3PriorFactorBatch prior(reinterpret_cast<const SE3Transform *>(z.data()), kN);
  prior.SetNumActiveFactors(kN);
  const float prior_error = JacobianError<SE3StateBatch>(prior, {&x}, 16, 6, 6, stream.GetStream());
  // Between with Delta = I: r = Log(L^-1 R), R = L * noise.
  std::vector<float> eye(kN * 16, 0.f);
  for (size_t f = 0; f < kN; ++f) {
    for (int k = 0; k < 4; ++k) eye[f * 16 + k * 5] = 1.f;
  }
  dvector<float> deltas(eye);
  SE3BetweenFactorBatch between(reinterpret_cast<const SE3Transform *>(deltas.data()), kN);
  between.SetNumActiveFactors(kN);
  const float between_error =
      JacobianError<SE3StateBatch>(between, {&x, &z}, 16, 6, 6, stream.GetStream());
  EXPECT_LT(prior_error, 1e-3f) << "residual scale " << residual_scale;
  EXPECT_LT(between_error, 1e-3f) << "residual scale " << residual_scale;
}

TEST_P(LieJacobianAccuracy, SO3PriorAndBetween) {
  const float residual_scale = GetParam();
  CudaStream stream;
  auto x = RandomElements<SO3StateBatch>(9, 3, 1.0f, 3, stream.GetStream());
  auto noise = RandomElements<SO3StateBatch>(9, 3, residual_scale, 4, stream.GetStream());
  dvector<float> z(kN * 9);
  {
    auto xh = ToHost(x), nh = ToHost(noise);
    std::vector<float> zh(kN * 9, 0.f);
    for (size_t f = 0; f < kN; ++f) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          float acc = 0.f;
          for (int k = 0; k < 3; ++k) acc += xh[f * 9 + i * 3 + k] * nh[f * 9 + k * 3 + j];
          zh[f * 9 + i * 3 + j] = acc;
        }
      }
    }
    z.CopyFromHost(zh.data(), zh.size());
  }
  SO3PriorFactorBatch prior(reinterpret_cast<const SO3Rotation *>(z.data()), kN);
  prior.SetNumActiveFactors(kN);
  const float prior_error = JacobianError<SO3StateBatch>(prior, {&x}, 9, 3, 3, stream.GetStream());
  std::vector<float> eye(kN * 9, 0.f);
  for (size_t f = 0; f < kN; ++f) {
    for (int k = 0; k < 3; ++k) eye[f * 9 + k * 4] = 1.f;
  }
  dvector<float> deltas(eye);
  SO3BetweenFactorBatch between(reinterpret_cast<const SO3Rotation *>(deltas.data()), kN);
  between.SetNumActiveFactors(kN);
  const float between_error =
      JacobianError<SO3StateBatch>(between, {&x, &z}, 9, 3, 3, stream.GetStream());
  EXPECT_LT(prior_error, 1e-3f) << "residual scale " << residual_scale;
  EXPECT_LT(between_error, 1e-3f) << "residual scale " << residual_scale;
}

INSTANTIATE_TEST_SUITE_P(ResidualScales, LieJacobianAccuracy,
                         ::testing::Values(1e-4f, 1e-3f, 1e-2f, 1e-1f, 3e-1f));

}  // namespace
}  // namespace cunls
