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
#include "cunls/factor/sized_factor_batch.h"

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
 * @brief Box constraints `lower <= x <= upper` on the components of a `dim`-dimensional
 * vector state, the size set at run time (the Python BoundFactorBatch; BoundFactorBatch<Dim>
 * fixes it at compile time).
 *
 * An inequality constraint batch by itself (no ConstraintFactorBatch wrapper needed): factor f
 * has 2 * dim rows,
 *
 * @verbatim
 *   rows [0, dim):        x_i - upper_i <= 0
 *   rows [dim, 2 dim):    lower_i - x_i <= 0
 * @endverbatim
 *
 * Bounds are per factor and per component; ±infinity leaves a side unbounded (the row is never
 * active).
 */
class DynamicBoundFactorBatch : public ConstraintFactorBatchBase {
 public:
  /**
   * @param dim Components of the bounded vector state (>= 1).
   * @param lower Device array of capacity * dim lower bounds (-inf: none).
   * @param upper Device array of capacity * dim upper bounds (+inf: none).
   * @param capacity Number of factors the buffers hold. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @param scale Positive row scale (see ConstraintFactorBatch).
   * @throws std::invalid_argument if dim < 1.
   */
  DynamicBoundFactorBatch(int dim, const float *lower, const float *upper, size_t capacity,
                          float scale = 1.f)
      : ConstraintFactorBatchBase(ConstraintKind::kInequality, scale, capacity),
        dim_(CheckedVectorDim("BoundFactorBatch", dim)),
        lower_(lower),
        upper_(upper) {
    AllocateMultipliers();
  }

  /** @brief Components of the bounded vector state. */
  int Dimension() const { return static_cast<int>(dim_); }
  size_t ResidualsSize() const override { return 2 * dim_; }
  std::vector<size_t> StateSizes() const override { return {dim_}; }

 protected:
  bool EvaluateConstraintRows(float *values, float *jacobians, float const *const *state_pointers,
                              cudaStream_t stream, const int *factor_ids,
                              size_t num_factor_ids) const override {
    LaunchBoundConstraintKernel(lower_, upper_, state_pointers, values, jacobians,
                                static_cast<int>(dim_), num_factor_ids, factor_ids,
                                NumActiveFactors(), stream);
    return true;
  }

 private:
  size_t dim_;          ///< Components of the bounded vector state.
  const float *lower_;  ///< Device lower bounds, capacity * dim floats.
  const float *upper_;  ///< Device upper bounds, capacity * dim floats.
};

/**
 * @brief DynamicBoundFactorBatch on a Dim-dimensional vector state (VectorStateBatch<Dim>),
 * the size fixed at compile time.
 *
 * @tparam Dim Dimension of the bounded vector state.
 */
template <int Dim>
class BoundFactorBatch : public DynamicBoundFactorBatch {
 public:
  /** @brief See DynamicBoundFactorBatch (capacity * Dim bounds each). */
  BoundFactorBatch(const float *lower, const float *upper, size_t capacity, float scale = 1.f)
      : DynamicBoundFactorBatch(Dim, lower, upper, capacity, scale) {}
};

}  // namespace cunls
