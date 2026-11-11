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
 * @file evaluate_items_prior_between_test.cpp
 * @brief Evaluate with items (num_items, factor_ids) of the prior and between
 * factor batches must give bitwise the same results as plain evaluations.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/between/similarity2_between_factor_batch.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/factor/between/sl4_between_factor_batch.h"
#include "cunls/factor/between/so2_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/prior/similarity2_prior_factor_batch.h"
#include "cunls/factor/prior/similarity3_prior_factor_batch.h"
#include "cunls/factor/prior/sl4_prior_factor_batch.h"
#include "cunls/factor/prior/so2_prior_factor_batch.h"
#include "cunls/factor/prior/so3_prior_factor_batch.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "tests/evaluate_items_check.h"

namespace cunls {
namespace {

using evaluate_items_test::CheckEvaluateItems;
using evaluate_items_test::ToDevice;

constexpr int kNumFactors = 23;
constexpr int kCopies = 4;

std::vector<float> RandomVector(size_t n, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = dist(rng);
  return v;
}

/**
 * @brief `count` valid elements of a matrix Lie group in device memory, built
 * as identity (+) random tangent with the state batch's Plus.
 */
template <typename StateT>
dvector<float> LieElements(int count, int ambient, int tangent, float scale, uint32_t seed) {
  const int dim = static_cast<int>(std::lround(std::sqrt(ambient)));
  std::vector<float> identity(static_cast<size_t>(count) * ambient, 0.f);
  for (int b = 0; b < count; ++b) {
    for (int i = 0; i < dim; ++i) identity[static_cast<size_t>(b) * ambient + i * dim + i] = 1.f;
  }
  auto base = ToDevice(identity);
  auto delta = ToDevice(RandomVector(static_cast<size_t>(count) * tangent, scale, seed));
  dvector<float> out(static_cast<size_t>(count) * ambient);
  CudaStream stream;
  StateT states(base.data(), count);
  states.SetNumActiveStates(states.Capacity(), states.ConstCapacity());
  states.Plus(base.data(), delta.data(), out.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

/**
 * @brief Checks a factor whose items read `arity` states (1: prior, 2: between)
 * of `ambient` floats each: copy k uses distinct states per (copy, factor),
 * from a pool of arity * kCopies * kNumFactors elements.
 */
void CheckFactor(FactorBatch &factor, dvector<float> &states, int arity, int ambient) {
  factor.SetNumActiveFactors(kNumFactors);
  CheckEvaluateItems(factor, kCopies, [&](int k) {
    std::vector<float *> p;
    for (int i = 0; i < arity * kNumFactors; ++i) {
      p.push_back(states.data() + static_cast<size_t>(arity * k * kNumFactors + i) * ambient);
    }
    return p;
  });
}

/** Builds kNumFactors measurements and arity * kCopies * kNumFactors states, then checks. */
template <typename FactorT, typename ObsT, typename StateT>
void CheckLie(int arity, int ambient, int tangent, uint32_t seed, float scale = 0.5f) {
  auto measurements = LieElements<StateT>(kNumFactors, ambient, tangent, scale, seed);
  FactorT factor(reinterpret_cast<const ObsT *>(measurements.data()), kNumFactors);
  auto states =
      LieElements<StateT>(arity * kCopies * kNumFactors, ambient, tangent, scale, seed + 1);
  CheckFactor(factor, states, arity, ambient);
}

template <typename FactorT, int kDim>
void CheckVector(int arity, uint32_t seed) {
  auto measurements = ToDevice(RandomVector(kNumFactors * kDim, 1.f, seed));
  FactorT factor(reinterpret_cast<const Vector<kDim> *>(measurements.data()), kNumFactors);
  auto states = ToDevice(RandomVector(arity * kCopies * kNumFactors * kDim, 1.f, seed + 1));
  CheckFactor(factor, states, arity, kDim);
}

// ----------------------------------------------------------------------------
// Priors (one state per factor)
// ----------------------------------------------------------------------------

TEST(EvaluateItemsPrior, PriorVectorMatchesEvaluate) {
  CheckVector<PriorVectorFactorBatch<5>, 5>(1, 30);
}

TEST(EvaluateItemsPrior, SE2PriorMatchesEvaluate) {
  CheckLie<SE2PriorFactorBatch, SE2Transform, SE2StateBatch>(1, 9, 3, 40);
}

TEST(EvaluateItemsPrior, SO2PriorMatchesEvaluate) {
  CheckLie<SO2PriorFactorBatch, SO2Rotation, SO2StateBatch>(1, 4, 1, 50);
}

TEST(EvaluateItemsPrior, SO3PriorMatchesEvaluate) {
  CheckLie<SO3PriorFactorBatch, SO3Rotation, SO3StateBatch>(1, 9, 3, 60);
}

TEST(EvaluateItemsPrior, Similarity2PriorMatchesEvaluate) {
  CheckLie<Similarity2PriorFactorBatch, Similarity2Transform, Similarity2StateBatch>(1, 9, 4, 70);
}

TEST(EvaluateItemsPrior, Similarity3PriorMatchesEvaluate) {
  CheckLie<Similarity3PriorFactorBatch, Similarity3Transform, Similarity3StateBatch>(1, 16, 7, 80);
}

TEST(EvaluateItemsPrior, SL4PriorMatchesEvaluate) {
  // SL(4)'s log is only well-defined near the identity: keep the elements close.
  CheckLie<SL4PriorFactorBatch, SL4Transform, SL4StateBatch>(1, 16, 15, 90, 0.1f);
}

// ----------------------------------------------------------------------------
// Betweens (left and right state per factor)
// ----------------------------------------------------------------------------

TEST(EvaluateItemsBetween, VectorBetweenMatchesEvaluate) {
  CheckVector<VectorBetweenFactorBatch<5>, 5>(2, 130);
}

TEST(EvaluateItemsBetween, SE2BetweenMatchesEvaluate) {
  CheckLie<SE2BetweenFactorBatch, SE2Transform, SE2StateBatch>(2, 9, 3, 140);
}

TEST(EvaluateItemsBetween, SO2BetweenMatchesEvaluate) {
  CheckLie<SO2BetweenFactorBatch, SO2Rotation, SO2StateBatch>(2, 4, 1, 150);
}

TEST(EvaluateItemsBetween, SO3BetweenMatchesEvaluate) {
  CheckLie<SO3BetweenFactorBatch, SO3Rotation, SO3StateBatch>(2, 9, 3, 160);
}

TEST(EvaluateItemsBetween, Similarity2BetweenMatchesEvaluate) {
  CheckLie<Similarity2BetweenFactorBatch, Similarity2Transform, Similarity2StateBatch>(2, 9, 4,
                                                                                       170);
}

TEST(EvaluateItemsBetween, Similarity3BetweenMatchesEvaluate) {
  CheckLie<Similarity3BetweenFactorBatch, Similarity3Transform, Similarity3StateBatch>(2, 16, 7,
                                                                                       180);
}

TEST(EvaluateItemsBetween, SL4BetweenMatchesEvaluate) {
  // SL(4)'s log is only well-defined near the identity: keep the elements close.
  CheckLie<SL4BetweenFactorBatch, SL4Transform, SL4StateBatch>(2, 16, 15, 190, 0.1f);
}

}  // namespace
}  // namespace cunls
