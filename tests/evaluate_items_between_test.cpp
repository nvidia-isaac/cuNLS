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
 * @file evaluate_items_between_test.cpp
 * @brief Evaluate with items (num_items, factor_ids) of the between factor
 * batches must give bitwise the same results as plain evaluations.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
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

cuBLASHandle &Cublas() {
  static cuBLASHandle handle;
  return handle;
}

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
struct LieElements {
  int ambient;
  dvector<float> d;

  LieElements(int count, int ambient_size, int tangent_size, float scale, uint32_t seed)
      : ambient(ambient_size), d(static_cast<size_t>(count) * ambient_size) {
    const int dim = static_cast<int>(std::lround(std::sqrt(ambient_size)));
    std::vector<float> identity(static_cast<size_t>(count) * ambient_size, 0.f);
    for (int b = 0; b < count; ++b) {
      for (int i = 0; i < dim; ++i) {
        identity[static_cast<size_t>(b) * ambient_size + i * dim + i] = 1.f;
      }
    }
    auto base = ToDevice(identity);
    auto delta = ToDevice(RandomVector(static_cast<size_t>(count) * tangent_size, scale, seed));
    CudaStream stream;
    StateT states(Cublas(), base.data(), count);
    states.Plus(base.data(), delta.data(), d.data(), stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  float *ptr(int i) { return d.data() + static_cast<size_t>(i) * ambient; }
};

/**
 * @brief State pointers of copy k: distinct (left, right) elements per
 * (copy, factor), taken from a pool of 2 * kCopies * kNumFactors elements.
 */
template <typename PtrFn>
std::vector<float *> PairPointersForCopy(PtrFn ptr, int k) {
  std::vector<float *> p;
  for (int i = 0; i < kNumFactors; ++i) {
    p.push_back(ptr(2 * (k * kNumFactors + i)));
    p.push_back(ptr(2 * (k * kNumFactors + i) + 1));
  }
  return p;
}

/** Builds kNumFactors deltas and 2 * kCopies * kNumFactors states, then checks. */
template <typename FactorT, typename ObsT, typename StateT>
void CheckLieBetween(int ambient, int tangent, uint32_t seed, float scale = 0.5f) {
  LieElements<StateT> deltas(kNumFactors, ambient, tangent, scale, seed);
  FactorT between(reinterpret_cast<const ObsT *>(deltas.d.data()), kNumFactors);
  LieElements<StateT> states(2 * kCopies * kNumFactors, ambient, tangent, scale, seed + 1);
  CheckEvaluateItems(between, kCopies, [&](int k) {
    return PairPointersForCopy([&](int i) { return states.ptr(i); }, k);
  });
}

TEST(EvaluateItemsBetween, VectorBetweenMatchesEvaluate) {
  constexpr int kDim = 5;
  auto deltas = ToDevice(RandomVector(kNumFactors * kDim, 1.f, 130));
  VectorBetweenFactorBatch<kDim> between(reinterpret_cast<const Vector<kDim> *>(deltas.data()),
                                         kNumFactors);
  auto states = ToDevice(RandomVector(2 * kCopies * kNumFactors * kDim, 1.f, 131));
  CheckEvaluateItems(between, kCopies, [&](int k) {
    return PairPointersForCopy(
        [&](int i) { return states.data() + static_cast<size_t>(i) * kDim; }, k);
  });
}

TEST(EvaluateItemsBetween, SE2BetweenMatchesEvaluate) {
  CheckLieBetween<SE2BetweenFactorBatch, SE2Transform, SE2StateBatch>(9, 3, 140);
}

TEST(EvaluateItemsBetween, SO2BetweenMatchesEvaluate) {
  CheckLieBetween<SO2BetweenFactorBatch, SO2Rotation, SO2StateBatch>(4, 1, 150);
}

TEST(EvaluateItemsBetween, SO3BetweenMatchesEvaluate) {
  CheckLieBetween<SO3BetweenFactorBatch, SO3Rotation, SO3StateBatch>(9, 3, 160);
}

TEST(EvaluateItemsBetween, Similarity2BetweenMatchesEvaluate) {
  CheckLieBetween<Similarity2BetweenFactorBatch, Similarity2Transform, Similarity2StateBatch>(
      9, 4, 170);
}

TEST(EvaluateItemsBetween, Similarity3BetweenMatchesEvaluate) {
  LieElements<Similarity3StateBatch> deltas(kNumFactors, 16, 7, 0.5f, 180);
  Similarity3BetweenFactorBatch between(
      Cublas(), reinterpret_cast<const Similarity3Transform *>(deltas.d.data()), kNumFactors);
  LieElements<Similarity3StateBatch> states(2 * kCopies * kNumFactors, 16, 7, 0.5f, 181);
  CheckEvaluateItems(between, kCopies, [&](int k) {
    return PairPointersForCopy([&](int i) { return states.ptr(i); }, k);
  });
}

TEST(EvaluateItemsBetween, SL4BetweenMatchesEvaluate) {
  // SL(4)'s log is only well-defined near the identity: keep the elements close.
  CheckLieBetween<SL4BetweenFactorBatch, SL4Transform, SL4StateBatch>(16, 15, 190, 0.1f);
}

}  // namespace
}  // namespace cunls
