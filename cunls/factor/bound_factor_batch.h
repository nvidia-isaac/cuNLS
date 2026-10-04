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

#include <cuda_runtime.h>

#include <vector>

#include "cunls/factor/constraint_factor_batch.h"

namespace cunls {

/**
 * @brief Launches the raw bound rows of BoundFactorBatch for `dim`-dimensional
 * states (see BoundFactorBatch for the layout).
 */
void LaunchBoundConstraintKernel(const float *lower, const float *upper,
                                 float const *const *state_pointers, float *values,
                                 float *jacobians, int dim, size_t num_items, const int *factor_ids,
                                 size_t num_factors, cudaStream_t stream);

/**
 * @brief Box constraints `lower <= x <= upper` on the components of a
 * Dim-dimensional vector state (VectorStateBatch<Dim>).
 *
 * An inequality constraint batch by itself (no ConstraintFactorBatch wrapper
 * needed): factor f has 2 * Dim rows,
 *
 * @verbatim
 *   rows [0, Dim):        x_i - upper_i <= 0
 *   rows [Dim, 2 Dim):    lower_i - x_i <= 0
 * @endverbatim
 *
 * Bounds are per factor and per component; ±infinity leaves a side
 * unbounded (the row is never active).
 *
 * @tparam Dim Dimension of the bounded vector state.
 */
template <int Dim>
class BoundFactorBatch : public ConstraintFactorBatchBase {
 public:
  /**
   * @param lower Device array of capacity * Dim lower bounds (-inf: none).
   * @param upper Device array of capacity * Dim upper bounds (+inf: none).
   * @param capacity Number of factors the buffers hold. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @param scale Positive row scale (see ConstraintFactorBatch).
   */
  BoundFactorBatch(const float *lower, const float *upper, size_t capacity, float scale = 1.f)
      : ConstraintFactorBatchBase(ConstraintKind::kInequality, scale, capacity),
        lower_(lower),
        upper_(upper) {
    AllocateMultipliers();
  }

  size_t ResidualsSize() const override { return 2 * Dim; }
  std::vector<size_t> StateSizes() const override { return {Dim}; }

 protected:
  bool EvaluateConstraintRows(float *values, float *jacobians, float const *const *state_pointers,
                              cudaStream_t stream, const int *factor_ids,
                              size_t num_factor_ids) const override {
    LaunchBoundConstraintKernel(lower_, upper_, state_pointers, values, jacobians, Dim,
                                num_factor_ids, factor_ids, this->NumActiveFactors(), stream);
    return true;
  }

 private:
  const float *lower_;
  const float *upper_;
};

}  // namespace cunls
