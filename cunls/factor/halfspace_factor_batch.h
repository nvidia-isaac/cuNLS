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

#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/** @brief Launches HalfspaceFactorBatch for `dim`-dimensional states. */
void LaunchHalfspaceKernel(const float *normals, const float *offsets,
                           float const *const *state_pointers, float *residuals, float *jacobians,
                           int dim, size_t num_items, const int *factor_ids, size_t num_factors,
                           cudaStream_t stream);

/**
 * @brief Signed halfspace value `r = aᵀ x - b` of a Dim-dimensional vector
 * state (VectorStateBatch<Dim>), with per-factor normal a and offset b.
 *
 * A constraint function: wrap it in
 * `ConstraintFactorBatch(&halfspaces, ConstraintKind::kInequality)` for
 * `aᵀ x <= b` (lanes, polygonal free space as several halfspaces), or as an
 * equality for a hyperplane. Jacobian: aᵀ (1 x Dim).
 *
 * @tparam Dim Dimension of the state.
 */
template <int Dim>
class HalfspaceFactorBatch : public SizedFactorBatch<1, Dim> {
 public:
  /**
   * @param normals Device array of capacity * Dim floats (a of factor f at f * Dim).
   * @param offsets Device array of capacity floats (b).
   * @param capacity Number of factors the buffers hold.
   */
  HalfspaceFactorBatch(const float *normals, const float *offsets, size_t capacity)
      : SizedFactorBatch<1, Dim>(capacity), normals_(normals), offsets_(offsets) {}

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override {
    const size_t num_factors = this->NumActiveFactors();
    const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
    if (num_items == 0 || num_factors == 0) return true;
    LaunchHalfspaceKernel(normals_, offsets_, state_pointers, residuals, jacobians, Dim, num_items,
                          factor_ids, num_factors, stream);
    return true;
  }

 private:
  const float *normals_;
  const float *offsets_;
};

}  // namespace cunls
