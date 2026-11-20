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

#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/** @brief Launches HalfspaceFactorBatch for `dim`-dimensional states. */
void LaunchHalfspaceKernel(const float *normals, const float *offsets,
                           float const *const *state_pointers, float *residuals, float *jacobians,
                           int dim, size_t num_items, const int *factor_ids, size_t num_factors,
                           cudaStream_t stream);

/**
 * @brief Signed halfspace value `r = aᵀ x - b` of a `dim`-dimensional vector state, with
 * per-factor normal a and offset b, the size set at run time (the Python HalfspaceFactorBatch;
 * HalfspaceFactorBatch<Dim> fixes it at compile time).
 *
 * A constraint function: wrap it in
 * `ConstraintFactorBatch(&halfspaces, ConstraintKind::kInequality)` for
 * `aᵀ x <= b` (lanes, polygonal free space as several halfspaces), or as an
 * equality for a hyperplane. Jacobian: aᵀ (1 x dim).
 */
class DynamicHalfspaceFactorBatch : public FactorBatch {
 public:
  /**
   * @param dim Components of the state (>= 1).
   * @param normals Device array of capacity * dim floats (a of factor f at f * dim).
   * @param offsets Device array of capacity floats (b).
   * @param capacity Number of factors the buffers hold.
   * @throws std::invalid_argument if dim < 1.
   */
  DynamicHalfspaceFactorBatch(int dim, const float *normals, const float *offsets, size_t capacity)
      : FactorBatch(capacity),
        dim_(CheckedVectorDim("HalfspaceFactorBatch", dim)),
        normals_(normals),
        offsets_(offsets) {}

  /** @brief Components of the state. */
  int Dimension() const { return static_cast<int>(dim_); }
  size_t ResidualsSize() const final { return 1; }
  std::vector<size_t> StateSizes() const final { return {dim_}; }

  /** @brief Evaluates `aᵀ x - b` and aᵀ; follows FactorBatch::Evaluate's item contract. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const final {
    const size_t num_factors = NumActiveFactors();
    const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
    if (num_items == 0 || num_factors == 0) return true;
    LaunchHalfspaceKernel(normals_, offsets_, state_pointers, residuals, jacobians,
                          static_cast<int>(dim_), num_items, factor_ids, num_factors, stream);
    return true;
  }

 private:
  size_t dim_;            ///< Components of the state.
  const float *normals_;  ///< Device normals, capacity * dim floats.
  const float *offsets_;  ///< Device offsets, capacity floats.
};

/**
 * @brief DynamicHalfspaceFactorBatch on a Dim-dimensional vector state (VectorStateBatch<Dim>),
 * the size fixed at compile time.
 *
 * @tparam Dim Dimension of the state.
 */
template <int Dim>
class HalfspaceFactorBatch : public DynamicHalfspaceFactorBatch {
 public:
  /** @brief The compile-time layout (one row, one state of Dim). */
  using sized_layout = SizedFactorBatch<1, Dim>;
  /** @brief Residual rows per factor. */
  static constexpr size_t residual_size_ = 1;

  /** @brief See DynamicHalfspaceFactorBatch (capacity * Dim normals, capacity offsets). */
  HalfspaceFactorBatch(const float *normals, const float *offsets, size_t capacity)
      : DynamicHalfspaceFactorBatch(Dim, normals, offsets, capacity) {}
};

}  // namespace cunls
