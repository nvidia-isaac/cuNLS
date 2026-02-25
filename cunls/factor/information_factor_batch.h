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

#include <numeric>
#include <sstream>
#include <type_traits>
#include <utility>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/log.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

namespace {
/**
 * @brief Helper struct for SFINAE-based type checking.
 *
 * Used to detect if a type T derives from SizedFactorBatch.
 */
struct DerivedFromAnySizedFactorBatchHelper {
  template <int Dim, int... StateBlockSizes>
  static std::true_type test(
      const SizedFactorBatch<Dim, StateBlockSizes...>*);

  /// Fallback overload for types that don't derive from SizedFactorBatch
  static std::false_type test(...);
};

/**
 * @brief Type trait that checks if T derives from SizedFactorBatch.
 *
 * Uses SFINAE to determine if a type is derived from any instantiation
 * of SizedFactorBatch. C++17 compatible alternative to concepts.
 */
template <class T>
struct IsDerivedFromAnySizedFactorBatch
    : decltype(DerivedFromAnySizedFactorBatchHelper::test(
          std::declval<T*>())){};

}  // namespace

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
 * @note The sqrt_information_matrices_ptr must point to GPU device memory and remain
 *       valid for the lifetime of this object. The memory layout is:
 *       [mat0: residual_size^2 floats][mat1: residual_size^2 floats]...
 */
template <class T,
          typename std::enable_if_t<
              IsDerivedFromAnySizedFactorBatch<T>::value, int> = 0>
class InformationFactorBatch : public FactorBatch {
 public:
  using InformationMatrix = Matrix<T::residual_size_>;

  /**
   * @brief Constructs an InformationFactorBatch wrapper.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param sqrt_information_matrices_ptr Pointer to GPU device memory containing
   *                                      square-root information matrices.
   *                                      Must point to at least num_factors * residual_size^2
   *                                      floats of allocated memory.
   * @param num_factors Number of factors in the batch.
   * @param sized_factor_batch_args Arguments forwarded to the wrapped
   * factor batch constructor.
   */
  template <class... Args>
  InformationFactorBatch(
      cuBLASHandle& cublas_handle,
      const InformationMatrix* sqrt_information_matrices_ptr,
      size_t num_factors,
      Args&&... sized_factor_batch_args)
      : cublas_handle_(cublas_handle),
        sqrt_information_matrices_ptr_(sqrt_information_matrices_ptr),
        num_factors_(num_factors),
        factor_batch_(
            cublas_handle, std::forward<Args>(sized_factor_batch_args)...) {
    if (num_factors_ != factor_batch_.NumFactors()) {
      std::stringstream ss;
      ss << "Number of sqrt information matrices must match number of factors";
      LogError(ss.str());
      throw std::invalid_argument(ss.str());
    }
  }

  /**
   * @brief Returns the size of residuals for each factor.
   *
   * @return Residual size (same as the wrapped factor)
   */
  size_t ResidualsSize() const final { return T::residual_size_; }

  /**
   * @brief Returns the number of factors in the batch.
   *
   * @return Number of factors (same as number of information matrices)
   */
  size_t NumFactors() const final {
    return num_factors_;
  }

  /**
   * @brief Returns the sizes of state blocks.
   *
   * @return Vector of state block sizes (delegated to wrapped factor).
   */
  std::vector<size_t> StateBlockSizes() const final {
    return factor_batch_.StateBlockSizes();
  }

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
  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final {
    factor_batch_.Evaluate(residuals, jacobians, state_pointers, stream);

    auto handle = cublas_handle_.GetHandle(stream);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    auto information_matrices_ptr = reinterpret_cast<const float*>(
        sqrt_information_matrices_ptr_);

    const size_t rsize = T::residual_size_;

    const size_t stride = rsize * rsize;
    constexpr size_t inc = 1;

    const size_t num_factors = factor_batch_.NumFactors();

    THROW_ON_CUBLAS_ERROR(cublasSgemvStridedBatched(
        handle, CUBLAS_OP_N, rsize, rsize, &alpha, information_matrices_ptr,
        rsize, stride, residuals, inc, rsize, &beta, residuals, inc, rsize,
        num_factors));

    if (jacobians == nullptr) {
      return true;
    }

    auto state_block_sizes = StateBlockSizes();

    const size_t jacobian_pitch = std::accumulate(
        state_block_sizes.begin(), state_block_sizes.end(), 0);

    const size_t jacobian_stride = jacobian_pitch * ResidualsSize();

    THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, jacobian_pitch, rsize, rsize, &alpha,
        jacobians, jacobian_pitch, jacobian_stride, information_matrices_ptr,
        rsize, stride, &beta, jacobians, jacobian_pitch, jacobian_stride,
        num_factors));

    return true;
  }

 private:
  T factor_batch_;  ///< Wrapped factor batch

  /// Pointer to user-managed device memory containing square-root information matrices.
  const InformationMatrix* sqrt_information_matrices_ptr_;

  /// Number of factors in the batch.
  size_t num_factors_;

  cuBLASHandle& cublas_handle_;  ///< cuBLAS handle for matrix operations
};

}  // namespace cunls
