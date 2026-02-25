/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * @param num_vectors Number of vectors to process
 * @param stream CUDA stream for kernel execution
 */
void LaunchPriorVectorFactorKernel(const float* observations,
                                 float const* const* state_pointers,
                                 float* residuals, float* jacobians, int dim,
                                 int num_vectors, cudaStream_t stream);

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
   * @param observations_ptr Pointer to GPU device memory containing observations.
   *                         Must point to at least num_factors * Dim floats of allocated memory.
   * @param num_factors Number of factors in the batch.
   */
  PriorVectorFactorBatch(const VectorType* observations_ptr,
                               size_t num_factors)
      : observations_ptr_(observations_ptr),
        num_factors_(num_factors) {}

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
   * @return true on success.
   */
  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final {
    auto data_ptr = reinterpret_cast<const float*>(observations_ptr_);
    size_t num_factors = this->NumFactors();

    LaunchPriorVectorFactorKernel(data_ptr, state_pointers, residuals, jacobians, Dim,
                                num_factors, stream);
    return true;
  }

  /**
   * @brief Returns the number of prior vector factors in this batch.
   * @return Number of factors.
   */
  size_t NumFactors() const final { return num_factors_; }

 private:
  PriorVectorFactorBatch() = default;

  /// Pointer to user-managed device memory containing observations.
  const VectorType* observations_ptr_;

  /// Number of factors in the batch.
  size_t num_factors_;
};

}  // namespace cunls
