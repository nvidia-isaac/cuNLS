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

#include <stdexcept>

#include "sized_state_batch.h"

namespace cunls {

/**
 * @brief Computes element-wise addition of vectors: x_plus_delta = x + delta.
 *
 * @param x Pointer to input vectors on GPU.
 * @param delta Pointer to delta vectors on GPU.
 * @param x_plus_delta Pointer to output vectors on GPU.
 * @param num_params Number of states.
 * @param dim Dimension of each vector.
 * @param stream CUDA stream for async execution.
 */
void CalculateVectorPlus(const float *x, const float *delta, float *x_plus_delta, size_t num_params,
                         int dim, cudaStream_t stream);

/**
 * @brief Clamps x to [lower, upper] element-wise where free[i] != 0.
 *
 * @param x Device array of num_values floats, updated in place.
 * @param free Device array of num_values flags (0: leave the entry as it is).
 * @param lower, upper Device arrays of num_values bounds (±inf: unbounded).
 */
void ProjectVectorToBounds(float *x, const float *free, const float *lower, const float *upper,
                           size_t num_values, cudaStream_t stream);

/**
 * @brief mask[i] = 0 where x[i] sits at a bound and direction[i] points outward.
 */
void MaskVectorActiveBounds(const float *x, const float *direction, const float *lower,
                            const float *upper, float *mask, size_t num_values,
                            cudaStream_t stream);

/**
 * @brief Batch of Euclidean vector states with compile-time dimension.
 *
 * For Euclidean states, the tangent and ambient spaces are identical
 * (both have dimension Dim), and the Plus operation reduces to element-wise
 * vector addition: x_plus_delta = x + delta.
 *
 * @tparam Dim The dimension of each vector state.
 */
template <int Dim>
class VectorStateBatch : public SizedStateBatch<Dim, Dim> {
 public:
  using Base = SizedStateBatch<Dim, Dim>;

  /**
   * @brief Constructs a batch of vector states.
   *
   * @param device_ptr Pointer to GPU device memory containing the vectors.
   *                   Must point to at least capacity * Dim floats of
   * allocated memory.
   * @param capacity Number of states the buffer holds. The active count
   *        starts at 0: call SetNumActiveStates(n) before solving.
   */
  VectorStateBatch(const float *device_ptr, size_t capacity) : Base(device_ptr, capacity) {}

  /**
   * @brief Constructs a batch of vector states with constant state
   * constraints.
   *
   * @param device_ptr Pointer to GPU device memory containing the vectors.
   *                   Must point to at least capacity * Dim floats of
   * allocated memory.
   * @param capacity Number of states the buffer holds. The active count
   *        starts at 0: call SetNumActiveStates(n) before solving.
   * @param device_constant_state_ids Pointer to GPU device memory containing
   * the indices of states that should remain constant.
   * @param const_capacity Number of ids the constant-id buffer holds.
   */
  VectorStateBatch(const float *device_ptr, size_t capacity, const int *device_constant_state_ids,
                   size_t const_capacity)
      : Base(device_ptr, capacity, device_constant_state_ids, const_capacity) {}

  /**
   * @brief Computes x_plus_delta = x + delta element-wise for all states.
   *
   * @param x             Device pointer to current state values.
   * @param delta         Device pointer to tangent-space updates.
   * @param x_plus_delta  Device pointer to output state values.
   * @param stream        CUDA stream for asynchronous execution.
   */
  void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
            size_t num_replicas = 1) override {
    CalculateVectorPlus(x, delta, x_plus_delta, this->num_active_states_ * num_replicas, Dim,
                        stream);
  }

  /**
   * @brief Box bounds lower <= x <= upper per component, enforced by the
   * Gauss-Newton and Levenberg-Marquardt minimizers by projection (see
   * StateBatch::HasBounds): the iterates stay inside the box, and no
   * constraint rows or penalties are involved.
   *
   * @param lower, upper Device arrays of Capacity() * Dim floats (state i at
   *        `i * Dim`); ±inf leaves a side unbounded. Not owned: they must
   *        outlive the solves. Both nullptr removes the bounds.
   * @throws std::invalid_argument if exactly one of them is nullptr.
   */
  void SetBounds(const float *lower, const float *upper) {
    if ((lower == nullptr) != (upper == nullptr)) {
      throw std::invalid_argument("VectorStateBatch::SetBounds: need both bounds or neither");
    }
    lower_ = lower;
    upper_ = upper;
  }

  bool HasBounds() const override { return lower_ != nullptr; }

  void ProjectToBounds(float *x, const float *free, cudaStream_t stream) const override {
    if (lower_ == nullptr) return;
    ProjectVectorToBounds(x, free, lower_, upper_, this->num_active_states_ * Dim, stream);
  }

  void MaskActiveBounds(const float *x, const float *direction, float *mask,
                        cudaStream_t stream) const override {
    if (lower_ == nullptr) return;
    MaskVectorActiveBounds(x, direction, lower_, upper_, mask, this->num_active_states_ * Dim,
                           stream);
  }

 private:
  /** @brief Default constructor (private, not for external use). */
  VectorStateBatch() = default;

  const float *lower_ = nullptr;
  const float *upper_ = nullptr;
};

}  // namespace cunls
