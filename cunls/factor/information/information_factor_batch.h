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

#include <numeric>
#include <sstream>
#include <type_traits>
#include <utility>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/log.h"
#include "cunls/common/type_traits.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Applies sqrt-information matrices to a batch of residual vectors.
 *
 * Computes residuals[i] = sqrt_information[i] * residuals[i] in-place
 * for each factor in the batch.
 *
 * @param cublas_handle Opaque cuBLAS handle (void*).
 * @param sqrt_information Batched square-root information matrices (device).
 * @param residuals Residual vectors, modified in-place (device).
 * @param residual_size Dimension of each residual / information matrix.
 * @param num_factors Number of factors in the batch.
 */
void ApplyInformationToResiduals(void *cublas_handle, const float *sqrt_information,
                                 float *residuals, size_t residual_size, size_t num_factors);

/**
 * @brief Applies sqrt-information matrices to a batch of Jacobian matrices.
 *
 * Computes jacobians[i] = sqrt_information[i] * jacobians[i] in-place
 * for each factor in the batch.
 *
 * @param cublas_handle Opaque cuBLAS handle (void*).
 * @param sqrt_information Batched square-root information matrices (device).
 * @param jacobians Jacobian matrices, modified in-place (device).
 * @param residual_size Row dimension of the Jacobian / information matrix.
 * @param jacobian_pitch Column dimension (total state-block width) of each
 * Jacobian.
 * @param num_factors Number of factors in the batch.
 */
void ApplyInformationToJacobians(void *cublas_handle, const float *sqrt_information,
                                 float *jacobians, size_t residual_size, size_t jacobian_pitch,
                                 size_t num_factors);

/**
 * @brief Applies per-item sqrt-information matrices to residual vectors.
 *
 * Computes residuals[t] = sqrt_information[f(t)] * residuals[t] in-place for
 * each of num_items items, where f(t) is the item's factor
 * (FactorBatch::Evaluate's item contract): factor_ids[t], or t modulo
 * num_factors when factor_ids is null. Each item uses a fixed summation order,
 * so its result does not depend on num_items or factor_ids.
 *
 * @param sqrt_information Row-major sqrt-information matrices, one per factor
 * (device).
 * @param residuals Residual vectors, modified in-place (device).
 * @param residual_size Dimension of each residual / information matrix.
 * @param num_items Number of residual vectors.
 * @param factor_ids Optional device array of per-item factor indices.
 * @param num_factors Number of sqrt-information matrices; 0 means num_items.
 * @param stream CUDA stream for asynchronous execution.
 */
void ApplyInformationToResidualItems(const float *sqrt_information, float *residuals,
                                     size_t residual_size, size_t num_items, const int *factor_ids,
                                     size_t num_factors, cudaStream_t stream);

/**
 * @brief Applies per-item sqrt-information matrices to Jacobian matrices.
 *
 * Computes jacobians[t] = sqrt_information[f(t)] * jacobians[t] in-place, with
 * f(t) as in ApplyInformationToResidualItems.
 *
 * @param sqrt_information Row-major sqrt-information matrices, one per factor
 * (device).
 * @param jacobians Row-major Jacobians, modified in-place (device).
 * @param residual_size Row dimension of the Jacobian / information matrix.
 * @param jacobian_pitch Column dimension (total state-block width) of each
 * Jacobian.
 * @param num_items Number of Jacobians.
 * @param factor_ids Optional device array of per-item factor indices.
 * @param num_factors Number of sqrt-information matrices; 0 means num_items.
 * @param stream CUDA stream for asynchronous execution.
 */
void ApplyInformationToJacobianItems(const float *sqrt_information, float *jacobians,
                                     size_t residual_size, size_t jacobian_pitch, size_t num_items,
                                     const int *factor_ids, size_t num_factors,
                                     cudaStream_t stream);

/**
 * @brief Wrapper factor that applies square-root information matrices.
 *
 * This class wraps a SizedFactorBatch and applies square-root information
 * matrices to both residuals and Jacobians. The information matrix represents
 * the inverse covariance of the measurement noise.
 *
 * For residuals: r_weighted = sqrt(Information) * r
 * For Jacobians: J_weighted = sqrt(Information) * J
 *
 * @tparam T The wrapped factor type, must derive from
 * SizedFactorBatch
 *
 * @note The sqrt_information_matrices_ptr must point to GPU device memory and
 * remain valid for the lifetime of this object. The memory layout is: [mat0:
 * residual_size^2 floats][mat1: residual_size^2 floats]...
 */
template <class T, typename std::enable_if_t<IsDerivedFromAnySizedFactorBatch<T>::value, int> = 0>
class InformationFactorBatch : public T::sized_layout {
 public:
  using InformationMatrix = Matrix<T::residual_size_>;

  /**
   * @brief Constructs an InformationFactorBatch wrapper.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param sqrt_information_matrices_ptr Pointer to GPU device memory
   * containing square-root information matrices, one per factor slot. Must
   * point to at least capacity * residual_size^2 floats of allocated memory.
   * @param capacity Number of square-root information matrices the buffer
   *        holds; must equal the wrapped batch's ``Capacity()``. The active
   *        count starts at 0: call SetNumFactors(n) before evaluating or solving.
   * @param sized_factor_batch_args Arguments forwarded to the wrapped
   * factor batch constructor verbatim (same order as ``T``'s constructor;
   * e.g. ``SE3BetweenFactorBatch`` and ``Similarity3BetweenFactorBatch``
   * still take a leading ``cuBLASHandle``, while factors such as
   * ``ReprojectionFactorBatch`` do not. For ``WeightedFactorBatch<U>``, pass
   * ``weight`` (or per-factor weights) then ``U``'s constructor arguments).
   */
  template <class... Args>
  InformationFactorBatch(cuBLASHandle &cublas_handle,
                         const InformationMatrix *sqrt_information_matrices_ptr, size_t capacity,
                         Args &&...sized_factor_batch_args)
      : cublas_handle_(cublas_handle),
        sqrt_information_matrices_ptr_(sqrt_information_matrices_ptr),
        num_matrices_(capacity),
        factor_batch_(std::forward<Args>(sized_factor_batch_args)...) {
    if (num_matrices_ != factor_batch_.Capacity()) {
      std::stringstream ss;
      ss << "Capacity of the sqrt information matrices (" << num_matrices_
         << ") must match the wrapped factor batch's capacity (" << factor_batch_.Capacity() << ")";
      LogError(ss.str());
      throw std::invalid_argument(ss.str());
    }
  }

  /**
   * @brief Returns the number of factors in the batch.
   *
   * @return Number of factors (same as number of information matrices)
   */
  size_t NumFactors() const override { return factor_batch_.NumFactors(); }

  /**
   * @brief Capacity of the wrapped factor batch. The sqrt-information matrices buffer must
   * hold one entry per factor of this capacity.
   */
  size_t Capacity() const override { return factor_batch_.Capacity(); }

  /**
   * @brief Sets the number of active factors of the wrapped batch (see
   * FactorBatch::SetNumFactors). Factor f keeps using entry f of the
   * sqrt-information matrices.
   *
   * @param num_factors Active count, at most Capacity().
   * @throws std::invalid_argument if num_factors > Capacity().
   */
  void SetNumFactors(size_t num_factors) override { factor_batch_.SetNumFactors(num_factors); }

  /**
   * @brief Evaluates the factor with information matrix weighting.
   *
   * First evaluates the wrapped factor, then applies the square-root
   * information matrices to residuals and Jacobians:
   * - residuals = sqrt(Information) * residuals
   * - jacobians = sqrt(Information) * jacobians
   *
   * @param residuals Output residuals (device pointer, modified in-place)
   * @param jacobians Output Jacobians (device pointer, modified in-place).
   *                  Can be nullptr if Jacobians are not needed.
   * @param state_pointers Array of state block pointers (device pointer to
   * device pointers)
   * @param stream CUDA stream for asynchronous execution
   * @param factor_ids Optional per-item factor indices (device), forwarded to
   * the wrapped batch unchanged.
   * @param num_factor_ids Number of items (the length of factor_ids when it
   *        is given); 0 means NumFactors().
   * @return true if evaluation succeeded, false otherwise
   */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override {
    const size_t num_items = num_factor_ids == 0 ? this->NumFactors() : num_factor_ids;
    if (num_items == 0 || NumFactors() == 0) {
      return true;
    }
    if (!factor_batch_.Evaluate(residuals, jacobians, state_pointers, stream, factor_ids,
                                num_items)) {
      return false;
    }

    auto info_ptr = reinterpret_cast<const float *>(sqrt_information_matrices_ptr_);
    const size_t rsize = T::residual_size_;
    const size_t num_factors = factor_batch_.NumFactors();

    if (residuals != nullptr) {
      ApplyInformationToResidualItems(info_ptr, residuals, rsize, num_items, factor_ids,
                                      num_factors, stream);
    }

    if (jacobians == nullptr) {
      return true;
    }

    auto state_block_sizes = this->StateBlockSizes();
    const size_t jacobian_pitch =
        std::accumulate(state_block_sizes.begin(), state_block_sizes.end(), 0);

    ApplyInformationToJacobianItems(info_ptr, jacobians, rsize, jacobian_pitch, num_items,
                                    factor_ids, num_factors, stream);

    return true;
  }

 private:
  T factor_batch_;  ///< Wrapped factor batch

  /// Pointer to user-managed device memory containing square-root information
  /// matrices.
  const InformationMatrix *sqrt_information_matrices_ptr_;

  /// Number of per-factor square-root information matrices (equals batch size).
  size_t num_matrices_;

  /// cuBLAS handle passed at construction. Kept for API compatibility; Evaluate
  /// applies the matrices with per-item kernels (ApplyInformationTo*Items) so
  /// that item results are independent of the item count.
  cuBLASHandle &cublas_handle_;
};

}  // namespace cunls
