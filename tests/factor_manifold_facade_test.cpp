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
 * @file factor_manifold_facade_test.cpp
 * @brief Unit tests for the compile-time manifold facades
 * (BetweenFactorBatch<Manifold>, PriorFactorBatch<Manifold>,
 * ConstantVelocityFactorBatch<Manifold>,
 * ConstantAccelerationFactorBatch<Manifold>).
 *
 * Covers: zero-cost specialization (sizeof equivalence), CTAD deduction for
 * BetweenFactorBatch/PriorFactorBatch (no <Manifold> needed), and
 * behavioral equivalence (facade Evaluate() output matches the direct
 * class's).
 */

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/between_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_factor_batch.h"
#include "cunls/factor/prior/prior_factor_batch.h"

namespace cunls {
namespace {

// --- sizeof equivalence: the facade adds no members/vtable entries. ------

static_assert(sizeof(BetweenFactorBatch<manifold::SE3>) == sizeof(SE3BetweenFactorBatch));
static_assert(sizeof(BetweenFactorBatch<manifold::SO3>) == sizeof(SO3BetweenFactorBatch));
static_assert(sizeof(BetweenFactorBatch<manifold::SE2>) == sizeof(SE2BetweenFactorBatch));
static_assert(sizeof(BetweenFactorBatch<manifold::SO2>) == sizeof(SO2BetweenFactorBatch));
static_assert(sizeof(BetweenFactorBatch<manifold::Similarity2>) ==
              sizeof(Similarity2BetweenFactorBatch));
static_assert(sizeof(BetweenFactorBatch<manifold::Similarity3>) ==
              sizeof(Similarity3BetweenFactorBatch));
static_assert(sizeof(BetweenFactorBatch<manifold::SL4>) == sizeof(SL4BetweenFactorBatch));
static_assert(sizeof(BetweenFactorBatch<manifold::Vector<3>>) ==
              sizeof(VectorBetweenFactorBatch<3>));

static_assert(sizeof(PriorFactorBatch<manifold::SE3>) == sizeof(SE3PriorFactorBatch));
static_assert(sizeof(PriorFactorBatch<manifold::SO3>) == sizeof(SO3PriorFactorBatch));
static_assert(sizeof(PriorFactorBatch<manifold::SE2>) == sizeof(SE2PriorFactorBatch));
static_assert(sizeof(PriorFactorBatch<manifold::SO2>) == sizeof(SO2PriorFactorBatch));
static_assert(sizeof(PriorFactorBatch<manifold::Similarity2>) ==
              sizeof(Similarity2PriorFactorBatch));
static_assert(sizeof(PriorFactorBatch<manifold::Similarity3>) ==
              sizeof(Similarity3PriorFactorBatch));
static_assert(sizeof(PriorFactorBatch<manifold::SL4>) == sizeof(SL4PriorFactorBatch));

static_assert(sizeof(ConstantVelocityFactorBatch<manifold::SE3>) ==
              sizeof(ConstantVelocitySE3FactorBatch));
static_assert(sizeof(ConstantAccelerationFactorBatch<manifold::SE3>) ==
              sizeof(ConstantAccelerationSE3FactorBatch));

// --- CTAD deduction compiles for every manifold (no <Manifold> written). -

static_assert(
    std::is_same_v<decltype(BetweenFactorBatch(std::declval<const SE3Transform *>(), size_t{})),
                   BetweenFactorBatch<manifold::SE3>>);
static_assert(
    std::is_same_v<decltype(BetweenFactorBatch(std::declval<const SO3Rotation *>(), size_t{})),
                   BetweenFactorBatch<manifold::SO3>>);
static_assert(
    std::is_same_v<decltype(BetweenFactorBatch(std::declval<const SE2Transform *>(), size_t{})),
                   BetweenFactorBatch<manifold::SE2>>);
static_assert(
    std::is_same_v<decltype(BetweenFactorBatch(std::declval<const SO2Rotation *>(), size_t{})),
                   BetweenFactorBatch<manifold::SO2>>);
static_assert(std::is_same_v<
              decltype(BetweenFactorBatch(std::declval<const Similarity2Transform *>(), size_t{})),
              BetweenFactorBatch<manifold::Similarity2>>);
static_assert(
    std::is_same_v<decltype(BetweenFactorBatch(std::declval<const SL4Transform *>(), size_t{})),
                   BetweenFactorBatch<manifold::SL4>>);
// No CTAD for manifold::Vector<Dim> (see between_factor_batch.h) — explicit
// template argument only; covered by the sizeof static_assert above.

static_assert(
    std::is_same_v<decltype(PriorFactorBatch(std::declval<const SE3Transform *>(), size_t{})),
                   PriorFactorBatch<manifold::SE3>>);
static_assert(
    std::is_same_v<decltype(PriorFactorBatch(std::declval<const SO2Rotation *>(), size_t{})),
                   PriorFactorBatch<manifold::SO2>>);

// --- Behavioral equivalence: facade Evaluate() matches the direct class. -

TEST(FactorManifoldFacadeTest, BetweenFactorBatchSE3DeducedMatchesDirectClass) {
  constexpr size_t kN = 8;
  SE3Transform identity{};
  identity[0] = identity[5] = identity[10] = identity[15] = 1.0f;
  std::vector<SE3Transform> poses(kN + 1, identity);
  std::vector<SE3Transform> deltas(kN, identity);

  dvector<SE3Transform> poses_dev(poses);
  dvector<SE3Transform> deltas_dev(deltas);

  std::vector<const float *> ptrs(2 * kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs[2 * i] = reinterpret_cast<const float *>(poses_dev.data() + (i + 1));
    ptrs[2 * i + 1] = reinterpret_cast<const float *>(poses_dev.data() + i);
  }
  dvector<const float *> ptrs_dev(ptrs);

  CudaStream stream;

  SE3BetweenFactorBatch direct(deltas_dev.data(), kN);
  dvector<float> res_direct(6 * kN), jac_direct(72 * kN);
  direct.Evaluate(res_direct.data(), jac_direct.data(),
                  reinterpret_cast<const float *const *>(ptrs_dev.data()), stream.GetStream());

  // CTAD: no <manifold::SE3> written anywhere.
  BetweenFactorBatch facade(deltas_dev.data(), kN);
  dvector<float> res_facade(6 * kN), jac_facade(72 * kN);
  facade.Evaluate(res_facade.data(), jac_facade.data(),
                  reinterpret_cast<const float *const *>(ptrs_dev.data()), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_direct(6 * kN), host_facade(6 * kN);
  res_direct.CopyToHost(host_direct.data(), 6 * kN);
  res_facade.CopyToHost(host_facade.data(), 6 * kN);
  for (size_t i = 0; i < 6 * kN; ++i) {
    EXPECT_FLOAT_EQ(host_direct[i], host_facade[i]) << "residual index " << i;
  }
}

TEST(FactorManifoldFacadeTest, PriorFactorBatchSO2DeducedMatchesDirectClass) {
  constexpr size_t kN = 8;
  SO2Rotation identity{};
  identity[0] = identity[3] = 1.0f;
  std::vector<SO2Rotation> poses(kN, identity);
  std::vector<SO2Rotation> observations(kN, identity);

  dvector<SO2Rotation> poses_dev(poses);
  dvector<SO2Rotation> observations_dev(observations);

  std::vector<const float *> ptrs(kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs[i] = reinterpret_cast<const float *>(poses_dev.data() + i);
  }
  dvector<const float *> ptrs_dev(ptrs);

  CudaStream stream;

  SO2PriorFactorBatch direct(observations_dev.data(), kN);
  dvector<float> res_direct(kN), jac_direct(2 * kN);
  direct.Evaluate(res_direct.data(), jac_direct.data(),
                  reinterpret_cast<const float *const *>(ptrs_dev.data()), stream.GetStream());

  // CTAD: no <manifold::SO2> written anywhere.
  PriorFactorBatch facade(observations_dev.data(), kN);
  dvector<float> res_facade(kN), jac_facade(2 * kN);
  facade.Evaluate(res_facade.data(), jac_facade.data(),
                  reinterpret_cast<const float *const *>(ptrs_dev.data()), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_direct(kN), host_facade(kN);
  res_direct.CopyToHost(host_direct.data(), kN);
  res_facade.CopyToHost(host_facade.data(), kN);
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(host_direct[i], host_facade[i]) << "residual index " << i;
  }
}

// Similarity3's between-factor deduction guide also disambiguates a leading
// cuBLASHandle& argument correctly.
TEST(FactorManifoldFacadeTest, BetweenFactorBatchSimilarity3DeducedCompiles) {
  constexpr size_t kN = 1;
  Similarity3Transform identity{};
  identity[0] = identity[5] = identity[10] = identity[15] = 1.0f;
  std::vector<Similarity3Transform> deltas(kN, identity);
  dvector<Similarity3Transform> deltas_dev(deltas);

  cuBLASHandle cublas;
  BetweenFactorBatch facade(cublas, deltas_dev.data(), kN);
  static_assert(std::is_same_v<decltype(facade), BetweenFactorBatch<manifold::Similarity3>>);
  EXPECT_EQ(facade.NumFactors(), kN);
}

// Motion-prior facades: explicit template argument only (no CTAD).
TEST(FactorManifoldFacadeTest, ConstantVelocityFactorBatchExplicitTemplateArgument) {
  const float dt = 0.1f;
  dvector<float> dt_dev(std::vector<float>{dt});
  ConstantVelocityFactorBatch<manifold::SE3> factor(dt_dev.data(), 1);
  EXPECT_EQ(factor.NumFactors(), 1u);
}

TEST(FactorManifoldFacadeTest, ConstantAccelerationFactorBatchExplicitTemplateArgument) {
  const float dt = 0.1f;
  dvector<float> dt_dev(std::vector<float>{dt});
  ConstantAccelerationFactorBatch<manifold::SE3> factor(dt_dev.data(), 1);
  EXPECT_EQ(factor.NumFactors(), 1u);
}

}  // namespace
}  // namespace cunls
