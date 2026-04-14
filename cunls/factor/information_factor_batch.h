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
void ApplyInformationToResiduals(void *cublas_handle,
                                 const float *sqrt_information,
                                 float *residuals, size_t residual_size,
                                 size_t num_factors);

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
void ApplyInformationToJacobians(void *cublas_handle,
                                 const float *sqrt_information,
                                 float *jacobians, size_t residual_size,
                                 size_t jacobian_pitch, size_t num_factors);

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
template <class T, typename std::enable_if_t<
                       IsDerivedFromAnySizedFactorBatch<T>::value, int> = 0>
class InformationFactorBatch : public T::sized_layout {
public:
  using InformationMatrix = Matrix<T::residual_size_>;

  /**
   * @brief Constructs an InformationFactorBatch wrapper.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param sqrt_information_matrices_ptr Pointer to GPU device memory
   * containing square-root information matrices. Must point to at least
   * num_matrices * residual_size^2 floats of allocated memory.
   * @param num_matrices Number of square-root information matrices; must equal
   *                     the wrapped batch's ``NumFactors()``.
   * @param sized_factor_batch_args Arguments forwarded to the wrapped
   * factor batch constructor verbatim (same order as ``T``'s constructor;
   * e.g. ``SE3BetweenFactorBatch`` and ``Similarity3BetweenFactorBatch``
   * still take a leading ``cuBLASHandle``, while factors such as
   * ``ReprojectionFactorBatch`` do not. For ``WeightedFactorBatch<U>``, pass
   * ``weight`` (or per-factor weights) then ``U``'s constructor arguments).
   */
  template <class... Args>
  InformationFactorBatch(cuBLASHandle &cublas_handle,
                         const InformationMatrix *sqrt_information_matrices_ptr,
                         size_t num_matrices, Args &&...sized_factor_batch_args)
      : cublas_handle_(cublas_handle),
        sqrt_information_matrices_ptr_(sqrt_information_matrices_ptr),
        num_matrices_(num_matrices),
        factor_batch_(std::forward<Args>(sized_factor_batch_args)...) {
    if (num_matrices_ != factor_batch_.NumFactors()) {
      std::stringstream ss;
      ss << "Number of sqrt information matrices (" << num_matrices_
         << ") must match wrapped factor batch size ("
         << factor_batch_.NumFactors() << ")";
      LogError(ss.str());
      throw std::invalid_argument(ss.str());
    }
  }

  /**
   * @brief Returns the number of factors in the batch.
   *
   * @return Number of factors (same as number of information matrices)
   */
  size_t NumFactors() const final { return factor_batch_.NumFactors(); }

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
   * @return true if evaluation succeeded, false otherwise
   */
  bool Evaluate(float *residuals, float *jacobians,
                float const *const *state_pointers,
                cudaStream_t stream) const final {
    factor_batch_.Evaluate(residuals, jacobians, state_pointers, stream);

    auto handle = cublas_handle_.GetHandle(stream);
    auto info_ptr =
        reinterpret_cast<const float *>(sqrt_information_matrices_ptr_);
    const size_t rsize = T::residual_size_;
    const size_t num_factors = factor_batch_.NumFactors();

    ApplyInformationToResiduals(handle, info_ptr, residuals, rsize,
                                num_factors);

    if (jacobians == nullptr) {
      return true;
    }

    auto state_block_sizes = this->StateBlockSizes();
    const size_t jacobian_pitch =
        std::accumulate(state_block_sizes.begin(), state_block_sizes.end(), 0);

    ApplyInformationToJacobians(handle, info_ptr, jacobians, rsize,
                                jacobian_pitch, num_factors);

    return true;
  }

private:
  T factor_batch_; ///< Wrapped factor batch

  /// Pointer to user-managed device memory containing square-root information
  /// matrices.
  const InformationMatrix *sqrt_information_matrices_ptr_;

  /// Number of per-factor square-root information matrices (equals batch size).
  size_t num_matrices_;

  cuBLASHandle &cublas_handle_; ///< cuBLAS handle for matrix operations
};

} // namespace cunls
