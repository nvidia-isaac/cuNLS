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
 * @brief x_plus_delta = x + delta element-wise, for num_params states of `dim`
 * components; with bounds, every component with a nonzero delta is then
 * clamped into [lower, upper] (a component the step does not move stays as
 * it is).
 *
 * @param x, delta, x_plus_delta Device arrays of num_params * dim floats.
 * @param lower, upper Device bounds (both nullptr: none), `bound_values`
 *        floats; component k uses entry k % bound_values (replicas share the
 *        bounds).
 */
void CalculateVectorPlus(const float *x, const float *delta, float *x_plus_delta, size_t num_params,
                         int dim, cudaStream_t stream, const float *lower = nullptr,
                         const float *upper = nullptr, size_t bound_values = 0);

/**
 * @brief Box bounds of a state batch: lower <= x <= upper per component
 * (VectorStateBatch::SetBounds). Bounds are a constraint, enforced only by
 * AugmentedLagrangianMinimizer; read through this interface (dynamic_cast
 * from StateBatch), so StateBatch itself knows nothing about bounds.
 */
class BoxBoundedStates {
 public:
  virtual ~BoxBoundedStates() = default;

  /** @brief Device lower bounds, Capacity() * dimension floats; nullptr: no bounds. */
  virtual const float *LowerBounds() const = 0;

  /** @brief Device upper bounds, Capacity() * dimension floats; nullptr: no bounds. */
  virtual const float *UpperBounds() const = 0;
};

/** @brief Whether `batch` has box bounds (see BoxBoundedStates). */
inline bool HasBoxBounds(const StateBatch *batch) {
  const auto *bounded = dynamic_cast<const BoxBoundedStates *>(batch);
  return bounded != nullptr && bounded->LowerBounds() != nullptr;
}

/**
 * @brief Batch of Euclidean vector states with compile-time dimension.
 *
 * For Euclidean states, the tangent and ambient spaces are identical
 * (both have dimension Dim), and the Plus operation reduces to element-wise
 * vector addition: x_plus_delta = x + delta. With box bounds (SetBounds),
 * Plus keeps every component it moves inside the box.
 *
 * @tparam Dim The dimension of each vector state.
 */
template <int Dim>
class VectorStateBatch : public SizedStateBatch<Dim, Dim>, public BoxBoundedStates {
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
   * @brief x_plus_delta = x + delta element-wise for all states; with bounds,
   * every component with a nonzero delta is clamped into the box (constant
   * states, whose delta is 0, are never moved or clamped).
   *
   * @param x             Device pointer to current state values.
   * @param delta         Device pointer to tangent-space updates.
   * @param x_plus_delta  Device pointer to output state values.
   * @param stream        CUDA stream for asynchronous execution.
   * @param num_replicas  Replicas of the batch (see StateBatch::Plus); they share the bounds.
   */
  void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
            size_t num_replicas = 1) override {
    CalculateVectorPlus(x, delta, x_plus_delta, this->num_active_states_ * num_replicas, Dim,
                        stream, lower_, upper_, this->num_active_states_ * Dim);
  }

  /**
   * @brief Box bounds lower <= x <= upper per component. Bounds are a
   * constraint: solve a problem with bounded states with
   * AugmentedLagrangianMinimizer, which keeps the iterates inside the box
   * (projected Gauss-Newton); GaussNewtonMinimizer, LevenbergMarquardtMinimizer
   * and the RANSAC minimizers reject it. Constant states are never moved.
   *
   * @param lower, upper Device arrays of Capacity() * Dim floats (state i at
   *        `i * Dim`); ±inf leaves a side unbounded. Not owned: they must
   *        outlive the solves, and are read at every solve (they may be
   *        rewritten, or rebound with another call, between solves). Both
   *        nullptr removes the bounds.
   * @throws std::invalid_argument if exactly one of them is nullptr.
   */
  void SetBounds(const float *lower, const float *upper) {
    if ((lower == nullptr) != (upper == nullptr)) {
      throw std::invalid_argument("VectorStateBatch::SetBounds: need both bounds or neither");
    }
    lower_ = lower;
    upper_ = upper;
  }

  /** @brief See BoxBoundedStates. */
  const float *LowerBounds() const override { return lower_; }
  /** @brief See BoxBoundedStates. */
  const float *UpperBounds() const override { return upper_; }

 private:
  /** @brief Default constructor (private, not for external use). */
  VectorStateBatch() = default;

  const float *lower_ = nullptr;
  const float *upper_ = nullptr;
};

}  // namespace cunls
