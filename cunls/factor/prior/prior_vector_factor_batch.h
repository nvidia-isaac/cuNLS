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

#include <cuda/std/array>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Launches the prior vector factor kernel.
 *
 * @param observations Pointer to observation data (flattened array of vectors)
 * @param state_pointers Array of state pointers
 * @param residuals Output residuals (can be nullptr)
 * @param jacobians Output jacobians (can be nullptr)
 * @param dim Dimension of each vector
 * @param num_vectors Number of items (vectors) to process
 * @param stream CUDA stream for kernel execution
 * @param factor_ids Optional per-item observation indices (device pointer);
 *                   nullptr means item t reads observation t % num_factors
 * @param num_factors Number of observations; 0 means num_vectors
 */
void LaunchPriorVectorFactorKernel(const float *observations, float const *const *state_pointers,
                                   float *residuals, float *jacobians, int dim, int num_vectors,
                                   cudaStream_t stream, const int *factor_ids = nullptr,
                                   int num_factors = 0);

/**
 * @brief Prior on `dim`-dimensional vector states, the size set at run time (the Python
 * PriorVectorFactorBatch; PriorVectorFactorBatch<Dim> fixes it at compile time):
 * residual = state - observation, Jacobian the identity.
 *
 * @note `observations` is device memory of capacity * dim floats
 *       ([obs0: dim][obs1: dim]...), valid for the batch's lifetime.
 */
class DynamicPriorVectorFactorBatch : public FactorBatch {
 public:
  /**
   * @param dim Components per vector (>= 1).
   * @param observations Device array of capacity * dim floats.
   * @param capacity Factors the buffer holds; the active count starts at 0
   *        (SetNumActiveFactors).
   * @throws std::invalid_argument if dim < 1.
   */
  DynamicPriorVectorFactorBatch(int dim, const float *observations, size_t capacity)
      : FactorBatch(capacity),
        dim_(CheckedVectorDim("PriorVectorFactorBatch", dim)),
        observations_(observations) {}

  /** @brief Components per vector. */
  int Dimension() const { return static_cast<int>(dim_); }
  size_t ResidualsSize() const final { return dim_; }
  std::vector<size_t> StateSizes() const final { return {dim_}; }

  /**
   * @brief residual = state - observation per item, Jacobian the identity (dim x dim);
   * follows FactorBatch::Evaluate's item contract. Either output may be nullptr.
   */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const final {
    const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
    if (num_items == 0 || NumActiveFactors() == 0) return true;
    LaunchPriorVectorFactorKernel(observations_, state_pointers, residuals, jacobians,
                                  static_cast<int>(dim_), static_cast<int>(num_items), stream,
                                  factor_ids, static_cast<int>(NumActiveFactors()));
    return true;
  }

 private:
  size_t dim_;                 ///< Components per vector.
  const float *observations_;  ///< Device observations, capacity * dim floats.
};

/**
 * @brief DynamicPriorVectorFactorBatch of Dim components, the size fixed at compile time
 * (also what WeightedFactorBatch / InformationFactorBatch need: `sized_layout`).
 *
 * @tparam Dim Dimension of each vector.
 */
template <int Dim>
class PriorVectorFactorBatch : public DynamicPriorVectorFactorBatch {
 public:
  /** @brief The compile-time layout (residual Dim, one state of Dim). */
  using sized_layout = SizedFactorBatch<Dim, Dim>;
  /** @brief Residual rows per factor. */
  static constexpr size_t residual_size_ = Dim;

  /**
   * @param observations_ptr Device array of capacity vectors.
   * @param capacity Factors the buffer holds; the active count starts at 0.
   */
  PriorVectorFactorBatch(const Vector<Dim> *observations_ptr, size_t capacity)
      : DynamicPriorVectorFactorBatch(Dim, reinterpret_cast<const float *>(observations_ptr),
                                      capacity) {}
};

}  // namespace cunls
