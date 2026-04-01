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

#include "cunls/common/log.h"
#include "cunls/common/type_traits.h"

namespace cunls {

/**
 * @brief Applies a uniform scalar weight to batched residual vectors.
 *
 * Computes residuals[i] *= weight for every element across all factors.
 *
 * @param weight Scalar weight to apply.
 * @param residuals Residual vectors, modified in-place (device).
 * @param total_elements Total number of float elements (num_factors * residual_size).
 * @param stream CUDA stream for asynchronous execution.
 */
void ApplyUniformWeightToResiduals(float weight, float* residuals,
                                   size_t total_elements,
                                   cudaStream_t stream);

/**
 * @brief Applies a uniform scalar weight to batched Jacobian matrices.
 *
 * Computes jacobians[i] *= weight for every element across all factors.
 *
 * @param weight Scalar weight to apply.
 * @param jacobians Jacobian matrices, modified in-place (device).
 * @param total_elements Total number of float elements
 *        (num_factors * residual_size * jacobian_pitch).
 * @param stream CUDA stream for asynchronous execution.
 */
void ApplyUniformWeightToJacobians(float weight, float* jacobians,
                                   size_t total_elements,
                                   cudaStream_t stream);

/**
 * @brief Applies per-factor scalar weights to batched residual vectors.
 *
 * Computes residuals[i * residual_size + j] *= weights[i] for each factor i.
 *
 * @param weights Device pointer to per-factor weights (num_factors floats).
 * @param residuals Residual vectors, modified in-place (device).
 * @param residual_size Dimension of each residual vector.
 * @param num_factors Number of factors in the batch.
 * @param stream CUDA stream for asynchronous execution.
 */
void ApplyPerFactorWeightToResiduals(const float* weights, float* residuals,
                                     size_t residual_size, size_t num_factors,
                                     cudaStream_t stream);

/**
 * @brief Applies per-factor scalar weights to batched Jacobian matrices.
 *
 * Computes jacobians[i * stride + j] *= weights[i] for each factor i,
 * where stride = residual_size * jacobian_pitch.
 *
 * @param weights Device pointer to per-factor weights (num_factors floats).
 * @param jacobians Jacobian matrices, modified in-place (device).
 * @param residual_size Row dimension of the Jacobian.
 * @param jacobian_pitch Column dimension (total state-block width) of each Jacobian.
 * @param num_factors Number of factors in the batch.
 * @param stream CUDA stream for asynchronous execution.
 */
void ApplyPerFactorWeightToJacobians(const float* weights, float* jacobians,
                                     size_t residual_size,
                                     size_t jacobian_pitch, size_t num_factors,
                                     cudaStream_t stream);

/**
 * @brief Wrapper factor that applies scalar weights to residuals and Jacobians.
 *
 * This class wraps a SizedFactorBatch and multiplies both residuals and
 * Jacobians by a scalar weight. Two construction modes are supported:
 *
 * 1. **Uniform weight**: A single float is applied to every factor equally.
 * 2. **Per-factor weights**: A device pointer to an array of floats provides
 *    one weight per factor, allowing heterogeneous weighting.
 *
 * For residuals: r_weighted = weight * r
 * For Jacobians: J_weighted = weight * J
 *
 * @tparam T The wrapped factor type, must derive from SizedFactorBatch.
 *
 * @note For per-factor weights, the weights pointer must point to GPU device
 *       memory and remain valid for the lifetime of this object.
 */
template <class T, typename std::enable_if_t<
                       IsDerivedFromAnySizedFactorBatch<T>::value, int> = 0>
class WeightedFactorBatch : public T::sized_layout {
 public:
  /**
   * @brief Constructs a WeightedFactorBatch with a uniform scalar weight.
   *
   * Every factor in the batch has its residual and Jacobian multiplied by
   * the same weight value. The number of factors is
   * ``factor_batch_.NumFactors()`` after construction of the inner batch.
   *
   * @param weight Scalar weight applied to all factors.
   * @param sized_factor_batch_args Arguments forwarded to the wrapped
   *        factor batch constructor.
   */
  template <class... Args>
  WeightedFactorBatch(float weight, Args&&... sized_factor_batch_args)
      : factor_batch_(std::forward<Args>(sized_factor_batch_args)...),
        uniform_weight_(weight),
        per_factor_weights_(nullptr) {}

  /**
   * @brief Constructs a WeightedFactorBatch with per-factor weights.
   *
   * Each factor i has its residual and Jacobian multiplied by weights[i].
   *
   * @param per_factor_weights Device pointer to an array of per-factor weights.
   *                           Must point to at least num_weights floats.
   * @param num_weights Number of weights; must equal the wrapped batch's
   *                    ``NumFactors()``.
   * @param sized_factor_batch_args Arguments forwarded to the wrapped
   *        factor batch constructor.
   */
  template <class... Args>
  WeightedFactorBatch(const float* per_factor_weights, size_t num_weights,
                      Args&&... sized_factor_batch_args)
      : factor_batch_(std::forward<Args>(sized_factor_batch_args)...),
        uniform_weight_(0.0f),
        per_factor_weights_(per_factor_weights) {
    if (per_factor_weights_ == nullptr) {
      std::stringstream ss;
      ss << "WeightedFactorBatch: per_factor_weights must not be null";
      LogError(ss.str());
      throw std::invalid_argument(ss.str());
    }
    if (num_weights != factor_batch_.NumFactors()) {
      std::stringstream ss;
      ss << "WeightedFactorBatch: num_weights (" << num_weights
         << ") must match wrapped factor batch size ("
         << factor_batch_.NumFactors() << ")";
      LogError(ss.str());
      throw std::invalid_argument(ss.str());
    }
  }

  size_t NumFactors() const final { return factor_batch_.NumFactors(); }

  /**
   * @brief Evaluates the factor with weight scaling.
   *
   * First evaluates the wrapped factor, then applies the weight(s)
   * to residuals and Jacobians:
   * - residuals = weight * residuals
   * - jacobians = weight * jacobians
   *
   * @param residuals Output residuals (device pointer, modified in-place)
   * @param jacobians Output Jacobians (device pointer, modified in-place).
   *                  Can be nullptr if Jacobians are not needed.
   * @param state_pointers Array of state block pointers (device pointer to
   *        device pointers)
   * @param stream CUDA stream for asynchronous execution
   * @return true if evaluation succeeded, false otherwise
   */
  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final {
    factor_batch_.Evaluate(residuals, jacobians, state_pointers, stream);

    const size_t rsize = T::residual_size_;
    const size_t num_factors = factor_batch_.NumFactors();

    if (per_factor_weights_ != nullptr) {
      ApplyPerFactorWeightToResiduals(per_factor_weights_, residuals, rsize,
                                      num_factors, stream);
    } else {
      ApplyUniformWeightToResiduals(uniform_weight_, residuals,
                                    num_factors * rsize, stream);
    }

    if (jacobians == nullptr) {
      return true;
    }

    auto state_block_sizes = this->StateBlockSizes();
    const size_t jacobian_pitch = std::accumulate(
        state_block_sizes.begin(), state_block_sizes.end(), size_t{0});

    if (per_factor_weights_ != nullptr) {
      ApplyPerFactorWeightToJacobians(per_factor_weights_, jacobians, rsize,
                                      jacobian_pitch, num_factors, stream);
    } else {
      ApplyUniformWeightToJacobians(uniform_weight_, jacobians,
                                    num_factors * rsize * jacobian_pitch,
                                    stream);
    }

    return true;
  }

 private:
  T factor_batch_;

  float uniform_weight_;

  const float* per_factor_weights_;
};

}  // namespace cunls
