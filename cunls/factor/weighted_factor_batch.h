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

#include <sstream>
#include <type_traits>
#include <utility>

#include "cunls/common/log.h"
#include "cunls/common/type_traits.h"

namespace cunls {

/**
 * @brief Evaluates `inner` under FactorBatch::Evaluate's item contract and
 * scales each item's residual and Jacobian rows by a scalar weight: the weight
 * of the item's factor f (factor_ids[t], or t % inner.NumActiveFactors()),
 * per_factor_weights[f], when per_factor_weights is not null, otherwise
 * uniform_weight. The evaluation of WeightedFactorBatch and of the Python
 * weighted wrapper.
 *
 * @param inner The weighted factor batch.
 * @param uniform_weight Weight of every item when per_factor_weights is null.
 * @param per_factor_weights Device array of inner.NumActiveFactors() weights,
 *        or nullptr.
 * @return false (nothing scaled) if inner's evaluation fails, true otherwise.
 */
bool EvaluateWeighted(const FactorBatch &inner, float uniform_weight,
                      const float *per_factor_weights, float *residuals, float *jacobians,
                      float const *const *state_pointers, cudaStream_t stream,
                      const int *factor_ids, size_t num_factor_ids);

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
template <class T, typename std::enable_if_t<HasSizedLayout<T>::value, int> = 0>
class WeightedFactorBatch : public T::sized_layout {
 public:
  /**
   * @brief Constructs a WeightedFactorBatch with a uniform scalar weight.
   *
   * Every factor in the batch has its residual and Jacobian multiplied by
   * the same weight value. The number of factors is
   * ``factor_batch_.NumActiveFactors()`` after construction of the inner batch.
   *
   * @param weight Scalar weight applied to all factors.
   * @param sized_factor_batch_args Arguments forwarded to the wrapped
   *        factor batch constructor.
   */
  template <class... Args>
  WeightedFactorBatch(float weight, Args &&...sized_factor_batch_args)
      : factor_batch_(std::forward<Args>(sized_factor_batch_args)...),
        uniform_weight_(weight),
        per_factor_weights_(nullptr) {}

  /**
   * @brief Constructs a WeightedFactorBatch with per-factor weights.
   *
   * Each factor i has its residual and Jacobian multiplied by weights[i].
   *
   * @param per_factor_weights Device pointer to an array of per-factor weights,
   *                           one per factor slot (at least `capacity` floats).
   * @param capacity Number of weights the buffer holds; must equal the wrapped
   *        batch's ``Capacity()``. The active count starts at 0: call
   *        SetNumActiveFactors(n) before evaluating or solving.
   * @param sized_factor_batch_args Arguments forwarded to the wrapped
   *        factor batch constructor.
   */
  template <class... Args>
  WeightedFactorBatch(const float *per_factor_weights, size_t capacity,
                      Args &&...sized_factor_batch_args)
      : factor_batch_(std::forward<Args>(sized_factor_batch_args)...),
        uniform_weight_(0.0f),
        per_factor_weights_(per_factor_weights) {
    if (per_factor_weights_ == nullptr) {
      std::stringstream ss;
      ss << "WeightedFactorBatch: per_factor_weights must not be null";
      LogError(ss.str());
      throw std::invalid_argument(ss.str());
    }
    if (capacity != factor_batch_.Capacity()) {
      std::stringstream ss;
      ss << "WeightedFactorBatch: capacity (" << capacity
         << ") must match the wrapped factor batch's capacity (" << factor_batch_.Capacity() << ")";
      LogError(ss.str());
      throw std::invalid_argument(ss.str());
    }
  }

  size_t NumActiveFactors() const override { return factor_batch_.NumActiveFactors(); }

  /**
   * @brief Capacity of the wrapped factor batch. The per-factor weights buffer (if used) must
   * hold one entry per factor of this capacity.
   */
  size_t Capacity() const override { return factor_batch_.Capacity(); }

  /**
   * @brief Sets the number of active factors of the wrapped batch (see
   * FactorBatch::SetNumActiveFactors). Factor f keeps using entry f of the
   * per-factor weights (if any).
   *
   * @param num_factors Active count, at most Capacity().
   * @throws std::invalid_argument if num_factors > Capacity().
   */
  void SetNumActiveFactors(size_t num_active_factors) override {
    factor_batch_.SetNumActiveFactors(num_active_factors);
  }

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
   * @param state_pointers Array of state pointers (device pointer to
   *        device pointers)
   * @param stream CUDA stream for asynchronous execution
   * @param factor_ids Optional per-item factor indices (device), forwarded
   *        to the wrapped batch unchanged.
   * @param num_factor_ids Number of items (the length of factor_ids when it
   *        is given); 0 means NumActiveFactors().
   * @return true if evaluation succeeded, false otherwise
   */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override {
    return EvaluateWeighted(factor_batch_, uniform_weight_, per_factor_weights_, residuals,
                            jacobians, state_pointers, stream, factor_ids, num_factor_ids);
  }

 private:
  T factor_batch_;

  float uniform_weight_;

  const float *per_factor_weights_;
};

}  // namespace cunls
