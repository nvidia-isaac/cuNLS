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

#pragma once

/**
 * @file evaluate_items_check.h
 * @brief Shared check of FactorBatch::Evaluate's item parameters: evaluating
 * items (factor_ids, num_factor_ids) must give bitwise the same rows as plain
 * evaluations of the corresponding factors and state sets.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/factor_batch.h"

namespace cunls {
namespace evaluate_items_test {

template <typename T>
std::vector<T> ToHost(const dvector<T> &d) {
  std::vector<T> h(d.size());
  if (!h.empty()) d.CopyToHost(h.data(), h.size());
  return h;
}

template <typename T>
dvector<T> ToDevice(const std::vector<T> &h) {
  dvector<T> d(h.size());
  if (!h.empty()) d.CopyFromHost(h.data(), h.size());
  return d;
}

/**
 * @brief Checks one factor batch.
 *
 * pointers_for_copy(k) returns the NumFactors() * B state pointers of state
 * set k (B = StateBlockSizes().size()), as a plain evaluation would use them.
 * Compares, bitwise:
 *  1. Evaluate(..., nullptr, copies * N) against one plain Evaluate per set;
 *  2. Evaluate(..., factor_ids, M) for random (factor, set) items with
 *     repeats against the matching rows of 1.
 * Jacobians are checked too when `jacobians` is true.
 */
inline void CheckEvaluateItems(const FactorBatch &factor, int copies,
                               const std::function<std::vector<float *>(int)> &pointers_for_copy,
                               bool jacobians = true) {
  CudaStream stream;
  const int n_f = static_cast<int>(factor.NumFactors());
  const int m = static_cast<int>(factor.ResidualsSize());
  const auto sizes = factor.StateBlockSizes();
  const int nb = static_cast<int>(sizes.size());
  const int n = static_cast<int>(std::accumulate(sizes.begin(), sizes.end(), size_t{0}));
  std::vector<float *> table;
  for (int k = 0; k < copies; ++k) {
    const auto p = pointers_for_copy(k);
    ASSERT_EQ(p.size(), static_cast<size_t>(n_f) * nb);
    table.insert(table.end(), p.begin(), p.end());
  }
  auto d_table = ToDevice(table);
  const size_t items = static_cast<size_t>(copies) * n_f;

  dvector<float> ref_r(items * m), ref_j(jacobians ? items * m * n : 0);
  for (int k = 0; k < copies; ++k) {
    ASSERT_TRUE(factor.Evaluate(ref_r.data() + k * n_f * m,
                                jacobians ? ref_j.data() + k * n_f * m * n : nullptr,
                                d_table.data() + k * n_f * nb, stream.GetStream()));
  }
  dvector<float> r(items * m), j(jacobians ? items * m * n : 0);
  ASSERT_TRUE(factor.Evaluate(r.data(), jacobians ? j.data() : nullptr, d_table.data(),
                              stream.GetStream(), nullptr, items));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hr = ToHost(ref_r);
  const auto hj = ToHost(ref_j);
  // The test data must be valid for the factor, or the comparison is vacuous.
  for (float x : hr) ASSERT_TRUE(std::isfinite(x)) << "reference residuals are not finite";
  for (float x : hj) ASSERT_TRUE(std::isfinite(x)) << "reference Jacobians are not finite";
  EXPECT_EQ(hr, ToHost(r)) << "replicated residuals";
  if (jacobians) {
    EXPECT_EQ(hj, ToHost(j)) << "replicated Jacobians";
  }

  std::mt19937 rng(7);
  const int num_items = 3 * n_f + 5;
  std::vector<int> ids(num_items), copy(num_items);
  std::vector<float *> item_table;
  for (int t = 0; t < num_items; ++t) {
    ids[t] = static_cast<int>(rng() % n_f);
    copy[t] = static_cast<int>(rng() % copies);
    for (int b = 0; b < nb; ++b) item_table.push_back(table[(copy[t] * n_f + ids[t]) * nb + b]);
  }
  auto d_ids = ToDevice(ids);
  auto d_items = ToDevice(item_table);
  dvector<float> ir(static_cast<size_t>(num_items) * m);
  dvector<float> ij(jacobians ? static_cast<size_t>(num_items) * m * n : 0);
  ASSERT_TRUE(factor.Evaluate(ir.data(), jacobians ? ij.data() : nullptr, d_items.data(),
                              stream.GetStream(), d_ids.data(), num_items));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hir = ToHost(ir);
  const auto hij = ToHost(ij);
  for (int t = 0; t < num_items; ++t) {
    const size_t row = static_cast<size_t>(copy[t]) * n_f + ids[t];
    ASSERT_EQ(0, std::memcmp(&hir[t * m], &hr[row * m], m * sizeof(float))) << "item " << t;
    if (jacobians) {
      ASSERT_EQ(0, std::memcmp(&hij[t * m * n], &hj[row * m * n], m * n * sizeof(float)))
          << "item " << t;
    }
  }
}

}  // namespace evaluate_items_test
}  // namespace cunls
