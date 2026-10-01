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

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Launches the prior vector factor kernel.
 *
 * @param observations Pointer to observation data (flattened array of vectors)
 * @param state_pointers Array of state block pointers
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
 * @brief Batch factor for prior vector constraints.
 *
 * Computes the residual between state vectors and observation vectors:
 * residual = state - observation
 *
 * @tparam Dim Dimension of each vector
 *
 * @note The observations_ptr must point to GPU device memory and remain
 *       valid for the lifetime of this object. The memory layout is:
 *       [obs0: Dim floats][obs1: Dim floats]...[obsN-1: Dim floats]
 */
template <int Dim>
class PriorVectorFactorBatch : public SizedFactorBatch<Dim, Dim> {
  using Base = SizedFactorBatch<Dim, Dim>;
  using VectorType = Vector<Dim>;

 public:
  /**
   * @brief Constructs a batch of prior vector factors.
   *
   * @param observations_ptr Pointer to GPU device memory containing
   * observations. Must point to at least capacity * Dim floats of allocated
   * memory.
   * @param capacity Number of factors the measurement buffers hold. The active
   *        count starts at 0: call SetNumFactors(n) before evaluating or solving.
   */
  PriorVectorFactorBatch(const VectorType *observations_ptr, size_t capacity)
      : SizedFactorBatch<Dim, Dim>(capacity), observations_ptr_(observations_ptr) {}

  /**
   * @brief Evaluates prior vector residuals and optionally Jacobians.
   *
   * Computes residual = state - observation for each vector in the batch.
   * The Jacobian is the identity matrix for each vector.
   *
   * @param residuals Output device pointer for residuals (Dim floats per
   *                  factor). Can be nullptr to skip residual computation.
   * @param jacobians Output device pointer for Jacobians (Dim x Dim floats per
   *                  factor). Can be nullptr to skip Jacobian computation.
   * @param state_pointers Device pointer to state block pointers. Each entry
   *                   points to a Dim-dimensional vector on the device.
   * @param stream CUDA stream for asynchronous execution.
   * @param factor_ids Optional per-item factor indices (device pointer).
   * @param num_factor_ids Number of items (the length of factor_ids when it
   *        is given); 0 means NumFactors().
   * @return true on success.
   */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const final {
    const size_t num_items = num_factor_ids == 0 ? this->NumFactors() : num_factor_ids;
    if (num_items == 0 || this->NumFactors() == 0) {
      return true;
    }
    auto data_ptr = reinterpret_cast<const float *>(observations_ptr_);

    LaunchPriorVectorFactorKernel(data_ptr, state_pointers, residuals, jacobians, Dim,
                                  static_cast<int>(num_items), stream, factor_ids,
                                  static_cast<int>(this->NumFactors()));
    return true;
  }

 private:
  PriorVectorFactorBatch() = default;

  /// Pointer to user-managed device memory containing observations.
  const VectorType *observations_ptr_;
};

}  // namespace cunls
