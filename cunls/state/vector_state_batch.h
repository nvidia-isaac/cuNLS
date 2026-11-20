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
#include <string>

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
 * @brief Box bounds of a state batch: lower <= value <= upper per tangent
 * component (VectorStateBatch::SetBounds, ArticulationStateBatch::SetBounds).
 * Bounds are a constraint, enforced only by AugmentedLagrangianMinimizer; read
 * through this interface (dynamic_cast from StateBatch), so StateBatch itself
 * knows nothing about bounds.
 *
 * The bounded value of tangent component k of state s is the ambient float
 * `x[s * AmbientSize() + offset[k]]`, with offset = BoundedComponentOffsets()
 * (the identity, offset[k] = k, for vector states).
 */
class BoxBoundedStates {
 public:
  virtual ~BoxBoundedStates() = default;

  /** @brief Device lower bounds, Capacity() * TangentSize() floats; nullptr: no bounds. */
  virtual const float *LowerBounds() const = 0;

  /** @brief Device upper bounds, Capacity() * TangentSize() floats; nullptr: no bounds. */
  virtual const float *UpperBounds() const = 0;

  /**
   * @brief Where each tangent component's bounded value is stored: a device
   * array of TangentSize() ints, the ambient offset within one state, or -1
   * for a component that has no bounds (e.g. the rotation of a ball joint).
   * nullptr (the default): the identity, for states whose ambient and
   * tangent coordinates coincide (VectorStateBatch).
   */
  virtual const int *BoundedComponentOffsets() const { return nullptr; }

  /**
   * @brief x ⊕ δ for the active states without clamping into the bounds,
   * for numeric differentiation (a clamped perturbation would give a
   * one-sided derivative at a bound). Same arrays as StateBatch::Plus with
   * one replica.
   */
  virtual void PlusIgnoringBounds(const float *x, const float *delta, float *x_plus_delta,
                                  cudaStream_t stream) = 0;
};

/** @brief Whether `batch` has box bounds (see BoxBoundedStates). */
inline bool HasBoxBounds(const StateBatch *batch) {
  const auto *bounded = dynamic_cast<const BoxBoundedStates *>(batch);
  return bounded != nullptr && bounded->LowerBounds() != nullptr;
}

/**
 * @brief Batch of Euclidean vector states of `dim` components, the size set at run time (the
 * Python VectorStateBatch; VectorStateBatch<Dim> fixes it at compile time).
 *
 * Tangent and ambient spaces coincide (`dim` floats each), and Plus is element-wise addition,
 * x_plus_delta = x + delta. With box bounds (SetBounds), Plus keeps every component it moves
 * inside the box.
 */
class DynamicVectorStateBatch : public DynamicSizedStateBatch, public BoxBoundedStates {
 public:
  /**
   * @param dim Components per state (>= 1).
   * @param device_ptr Device storage of capacity * dim floats.
   * @param capacity States the storage holds; the active count starts at 0.
   * @param device_constant_state_ids Optional device ids of constant states.
   * @param const_capacity Entries of device_constant_state_ids.
   * @throws std::invalid_argument if dim < 1.
   */
  DynamicVectorStateBatch(int dim, const float *device_ptr, size_t capacity,
                          const int *device_constant_state_ids = nullptr, size_t const_capacity = 0)
      : DynamicSizedStateBatch(CheckedDim(dim), CheckedDim(dim), device_ptr, capacity,
                               device_constant_state_ids, const_capacity) {}

  /** @brief Components per state. */
  int Dimension() const { return static_cast<int>(AmbientSize()); }

  /** @brief x + delta for `num_replicas` copies of the active states, clamped into the bounds. */
  void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
            size_t num_replicas = 1) override {
    CalculateVectorPlus(x, delta, x_plus_delta, num_active_states_ * num_replicas, Dimension(),
                        stream, lower_, upper_, num_active_states_ * AmbientSize());
  }

  /**
   * @brief Box bounds: device arrays of Capacity() * dim floats (±inf: unbounded), not
   * owned; both nullptr removes them.
   * @throws std::invalid_argument if only one is given.
   */
  void SetBounds(const float *lower, const float *upper) {
    if ((lower == nullptr) != (upper == nullptr)) {
      throw std::invalid_argument(
          "VectorStateBatch::SetBounds: got one bound and nullptr (expected both bounds or "
          "neither). Fix: pass both arrays, or nullptr for both.");
    }
    lower_ = lower;
    upper_ = upper;
  }

  const float *LowerBounds() const override { return lower_; }
  const float *UpperBounds() const override { return upper_; }

  /** @brief x + delta for the active states without clamping (numeric differentiation). */
  void PlusIgnoringBounds(const float *x, const float *delta, float *x_plus_delta,
                          cudaStream_t stream) override {
    CalculateVectorPlus(x, delta, x_plus_delta, num_active_states_, Dimension(), stream);
  }

 private:
  static size_t CheckedDim(int dim) {
    if (dim < 1) {
      throw std::invalid_argument("VectorStateBatch: dim must be at least 1 (got " +
                                  std::to_string(dim) + "). Fix: pass the vector size.");
    }
    return static_cast<size_t>(dim);
  }

  const float *lower_ = nullptr;
  const float *upper_ = nullptr;
};

/**
 * @brief DynamicVectorStateBatch of Dim components, the size fixed at compile time.
 *
 * @tparam Dim The dimension of each vector state.
 */
template <int Dim>
class VectorStateBatch : public DynamicVectorStateBatch {
 public:
  /** @brief Storage of capacity * Dim floats; no constant states. */
  VectorStateBatch(const float *device_ptr, size_t capacity)
      : DynamicVectorStateBatch(Dim, device_ptr, capacity) {}

  /** @brief Storage of capacity * Dim floats and constant state ids. */
  VectorStateBatch(const float *device_ptr, size_t capacity, const int *device_constant_state_ids,
                   size_t const_capacity)
      : DynamicVectorStateBatch(Dim, device_ptr, capacity, device_constant_state_ids,
                                const_capacity) {}
};

}  // namespace cunls
