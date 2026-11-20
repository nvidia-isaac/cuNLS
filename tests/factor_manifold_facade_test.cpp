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
 * behavioral equivalence (the facade's Evaluate() is the direct class's).
 */

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>
#include <vector>

#include "cunls/common/device_vector.h"
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

// --- Behavioral equivalence: the facade is the concrete class (it derives
// from it and does not redeclare Evaluate), so its Evaluate() is the direct
// class's.

static_assert(std::is_base_of_v<SE3BetweenFactorBatch, BetweenFactorBatch<manifold::SE3>>);
static_assert(std::is_same_v<decltype(&BetweenFactorBatch<manifold::SE3>::Evaluate),
                             decltype(&SE3BetweenFactorBatch::Evaluate)>);
static_assert(std::is_base_of_v<SO2PriorFactorBatch, PriorFactorBatch<manifold::SO2>>);
static_assert(std::is_same_v<decltype(&PriorFactorBatch<manifold::SO2>::Evaluate),
                             decltype(&SO2PriorFactorBatch::Evaluate)>);

// Similarity3's between-factor deduction guide resolves from the measurement
// pointer type alone, like the other manifolds.
TEST(FactorManifoldFacadeTest, BetweenFactorBatchSimilarity3DeducedCompiles) {
  constexpr size_t kN = 1;
  Similarity3Transform identity{};
  identity[0] = identity[5] = identity[10] = identity[15] = 1.0f;
  std::vector<Similarity3Transform> deltas(kN, identity);
  dvector<Similarity3Transform> deltas_dev(deltas);

  BetweenFactorBatch facade(deltas_dev.data(), kN);
  facade.SetNumActiveFactors(facade.Capacity());
  static_assert(std::is_same_v<decltype(facade), BetweenFactorBatch<manifold::Similarity3>>);
  EXPECT_EQ(facade.NumActiveFactors(), kN);
}

// Motion-prior facades: explicit template argument only (no CTAD).
TEST(FactorManifoldFacadeTest, ConstantVelocityFactorBatchExplicitTemplateArgument) {
  const float dt = 0.1f;
  dvector<float> dt_dev(std::vector<float>{dt});
  ConstantVelocityFactorBatch<manifold::SE3> factor(dt_dev.data(), 1);
  factor.SetNumActiveFactors(factor.Capacity());
  EXPECT_EQ(factor.NumActiveFactors(), 1u);
}

TEST(FactorManifoldFacadeTest, ConstantAccelerationFactorBatchExplicitTemplateArgument) {
  const float dt = 0.1f;
  dvector<float> dt_dev(std::vector<float>{dt});
  ConstantAccelerationFactorBatch<manifold::SE3> factor(dt_dev.data(), 1);
  factor.SetNumActiveFactors(factor.Capacity());
  EXPECT_EQ(factor.NumActiveFactors(), 1u);
}

}  // namespace
}  // namespace cunls
