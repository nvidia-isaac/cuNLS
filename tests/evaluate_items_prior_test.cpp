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
 * @file evaluate_items_prior_test.cpp
 * @brief Evaluate with items (num_items, factor_ids) of the prior factor
 * batches must give bitwise the same results as plain evaluations.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
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
    states.SetNumStateBlocks(states.Capacity(), states.ConstCapacity());
    states.Plus(base.data(), delta.data(), d.data(), stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  float *ptr(int i) { return d.data() + static_cast<size_t>(i) * ambient; }
};

/** State pointers of copy k: distinct element per (copy, factor). */
template <typename Elements>
std::vector<float *> PointersForCopy(Elements &states, int k) {
  std::vector<float *> p;
  for (int i = 0; i < kNumFactors; ++i) p.push_back(states.ptr(k * kNumFactors + i));
  return p;
}

/** Builds kNumFactors targets and kCopies * kNumFactors states, then checks. */
template <typename FactorT, typename ObsT, typename StateT>
void CheckLiePrior(int ambient, int tangent, uint32_t seed, float scale = 0.5f) {
  LieElements<StateT> targets(kNumFactors, ambient, tangent, scale, seed);
  FactorT prior(reinterpret_cast<const ObsT *>(targets.d.data()), kNumFactors);
  prior.SetNumFactors(kNumFactors);
  LieElements<StateT> states(kCopies * kNumFactors, ambient, tangent, scale, seed + 1);
  CheckEvaluateItems(prior, kCopies, [&](int k) { return PointersForCopy(states, k); });
}

TEST(EvaluateItemsPrior, PriorVectorMatchesEvaluate) {
  constexpr int kDim = 5;
  auto targets = ToDevice(RandomVector(kNumFactors * kDim, 1.f, 30));
  PriorVectorFactorBatch<kDim> prior(reinterpret_cast<const Vector<kDim> *>(targets.data()),
                                     kNumFactors);
  prior.SetNumFactors(prior.Capacity());
  auto states = ToDevice(RandomVector(kCopies * kNumFactors * kDim, 1.f, 31));
  CheckEvaluateItems(prior, kCopies, [&](int k) {
    std::vector<float *> p;
    for (int i = 0; i < kNumFactors; ++i) {
      p.push_back(states.data() + static_cast<size_t>(k * kNumFactors + i) * kDim);
    }
    return p;
  });
}

TEST(EvaluateItemsPrior, SE2PriorMatchesEvaluate) {
  CheckLiePrior<SE2PriorFactorBatch, SE2Transform, SE2StateBatch>(9, 3, 40);
}

TEST(EvaluateItemsPrior, SO2PriorMatchesEvaluate) {
  CheckLiePrior<SO2PriorFactorBatch, SO2Rotation, SO2StateBatch>(4, 1, 50);
}

TEST(EvaluateItemsPrior, SO3PriorMatchesEvaluate) {
  CheckLiePrior<SO3PriorFactorBatch, SO3Rotation, SO3StateBatch>(9, 3, 60);
}

TEST(EvaluateItemsPrior, Similarity2PriorMatchesEvaluate) {
  CheckLiePrior<Similarity2PriorFactorBatch, Similarity2Transform, Similarity2StateBatch>(9, 4, 70);
}

TEST(EvaluateItemsPrior, Similarity3PriorMatchesEvaluate) {
  CheckLiePrior<Similarity3PriorFactorBatch, Similarity3Transform, Similarity3StateBatch>(16, 7,
                                                                                          80);
}

TEST(EvaluateItemsPrior, SL4PriorMatchesEvaluate) {
  // SL(4)'s log is only well-defined near the identity: keep the elements close.
  CheckLiePrior<SL4PriorFactorBatch, SL4Transform, SL4StateBatch>(16, 15, 90, 0.1f);
}

}  // namespace
}  // namespace cunls
