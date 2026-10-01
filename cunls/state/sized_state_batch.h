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

#include "state_batch.h"

namespace cunls {

/**
 * @brief Template class for batch processing of states with compile-time
 * known dimensions.
 *
 * This class provides a concrete implementation of StateBatch for states
 * where both the ambient dimension (storage size) and tangent dimension
 * (optimization space) are known at compile time. This enables compile-time
 * optimizations and type safety.
 *
 * The class manages a contiguous array of states stored on the GPU device
 * memory. Each state occupies AmbientDim floats, and states are stored
 * sequentially.
 *
 * @tparam AmbientDim The dimension of the ambient space (storage size per
 * state). This is the number of floats needed to store one state.
 * @tparam TangentDim The dimension of the tangent space (optimization space per
 * state). This is the number of floats needed to represent an
 * update/delta.
 *
 * @note This is a base class that provides storage and access methods. Derived
 * classes (such as SE3StateBatch or VectorStateBatch) must implement the Plus()
 * operation specific to their manifold structure.
 *
 * @note The data pointed to by device_ptr must be allocated on the GPU device
 * and remain valid for the lifetime of this object. The memory layout is:
 *       [state0: AmbientDim floats][state1: AmbientDim floats]...[stateN-1:
 * AmbientDim floats]
 */
template <int AmbientDim, int TangentDim>
class SizedStateBatch : public StateBatch {
 public:
  /**
   * @brief Constructs a batch of states with no constant states.
   *
   * The batch wraps user-owned GPU memory holding up to `capacity` states of
   * AmbientDim floats each: [state0][state1]...[state(capacity-1)]. The active
   * count starts at 0: call SetNumActiveStates(n) before solving. Only the first
   * n states are read and written; the rest stay addressable through
   * StateDevicePtr().
   *
   * @param device_ptr GPU memory of at least capacity * AmbientDim floats.
   * @param capacity Number of states the buffer holds.
   */
  SizedStateBatch(const float *device_ptr, size_t capacity)
      : ptr_(device_ptr),
        num_active_states_(0),
        capacity_(capacity),
        constant_state_ids_(nullptr),
        num_const_states_(0),
        const_capacity_(0) {}

  /**
   * @brief Constructs a batch of states with a buffer of constant-state
   * ids.
   *
   * As above, plus a user-owned GPU array of up to `const_capacity` indices of
   * states held constant. The active counts start at 0: call
   * SetNumActiveStates(n, num_const) before solving; the first num_const ids
   * (each below n) are then constant.
   *
   * @param device_ptr GPU memory of at least capacity * AmbientDim floats.
   * @param capacity Number of states the buffer holds.
   * @param device_constant_state_ids GPU array of at least const_capacity
   *        ints, or nullptr if const_capacity is 0. Not copied: it must stay
   *        valid for the lifetime of this object.
   * @param const_capacity Number of ids the constant-id buffer holds.
   */
  SizedStateBatch(const float *device_ptr, size_t capacity, const int *device_constant_state_ids,
                  size_t const_capacity)
      : ptr_(device_ptr),
        num_active_states_(0),
        capacity_(capacity),
        constant_state_ids_(device_constant_state_ids),
        num_const_states_(0),
        const_capacity_(const_capacity) {}

  /**
   * @brief Returns the number of states in this batch.
   *
   * @return The total number of states managed by this batch.
   */
  size_t NumActiveStates() const final { return num_active_states_; }

  /**
   * @brief Returns the dimension of the tangent space.
   *
   * The tangent space dimension determines the size of update vectors (deltas)
   * used in optimization. This is a compile-time constant equal to TangentDim.
   *
   * @return The tangent space dimension (TangentDim).
   */
  size_t TangentSize() const final { return TangentDim; };

  /**
   * @brief Returns the dimension of the ambient space.
   *
   * The ambient space dimension determines the storage size of each
   * state. This is a compile-time constant equal to AmbientDim.
   *
   * @return The ambient space dimension (AmbientDim).
   */
  size_t AmbientSize() const final { return AmbientDim; };

  /**
   * @brief Returns a mutable device pointer to a specific state.
   *
   * Computes the device memory address of the state at the given index.
   * The pointer can be used to read or modify the state data on the GPU.
   *
   * @param state_idx The zero-based index of the state, in
   *        [0, Capacity()). States at or above NumActiveStates() are inactive but
   *        addressable, so connectivity for a coming solve can be built before
   *        SetNumActiveStates() is called.
   * @return Device pointer to the state data (AmbientDim floats), or
   *         nullptr if state_idx >= Capacity().
   *
   * @note The returned pointer points to GPU device memory. Use CUDA memory
   * operations or kernels to access/modify the data.
   */
  float *StateDevicePtr(size_t state_idx) final {
    if (state_idx >= capacity_) {
      return nullptr;
    }

    return const_cast<float *>(ptr_ + state_idx * AmbientDim);
  }

  /**
   * @brief Returns a const device pointer to a specific state.
   *
   * Computes the device memory address of the state at the given index.
   * The pointer provides read-only access to the state data on the GPU.
   *
   * @param state_idx The zero-based index of the state, in
   *        [0, Capacity()) (see the non-const overload).
   * @return Const device pointer to the state data (AmbientDim floats),
   *         or nullptr if state_idx >= Capacity().
   *
   * @note The returned pointer points to GPU device memory. Use CUDA memory
   * operations or kernels to read the data.
   */
  const float *StateDevicePtr(size_t state_idx) const final {
    if (state_idx >= capacity_) {
      return nullptr;
    }

    return ptr_ + state_idx * AmbientDim;
  }

  /**
   * @brief Returns a pointer to the array of constant state indices.
   *
   * Returns the device pointer to the array containing indices of states
   * that should remain constant (not optimized) during the optimization
   * process.
   *
   * @return Device pointer to an array of integer indices, or nullptr if no
   * states are marked as constant. The array should contain sorted, unique
   * indices in the range [0, NumActiveStates()).
   *
   * @note The returned pointer is valid only if the batch was constructed with
   *       constant_state_ids. Otherwise, it may be nullptr or uninitialized.
   */
  const int *ConstStateIds() const final { return constant_state_ids_; }

  /**
   * @brief Returns the number of states marked as constant.
   * @return The number of constant (non-optimized) states.
   */
  size_t NumConstStates() const final { return num_const_states_; }

  /**
   * @brief Capacity of the state buffer, in states: the constructor's capacity.
   */
  size_t Capacity() const final { return capacity_; }

  /**
   * @brief Entry capacity of the constant-id buffer: the constructor's
   * const_capacity.
   */
  size_t ConstCapacity() const final { return const_capacity_; }

  /**
   * @brief Sets the active state and constant-id counts (see
   * StateBatch::SetNumActiveStates). Host-only; takes effect at the next Plus /
   * Minimize.
   *
   * @param num_active_states Active states: the first `num_active_states`
   *        states of the buffer, at most Capacity().
   * @param num_const_states Active constant ids: the first entries of the
   *        constant-id buffer, at most ConstCapacity(); each must be below
   *        `num_active_states`.
   * @throws std::invalid_argument if a count exceeds its capacity.
   */
  void SetNumActiveStates(size_t num_active_states, size_t num_const_states = 0) override {
    if (num_active_states > capacity_ || num_const_states > const_capacity_) {
      throw std::invalid_argument("SetNumActiveStates(" + std::to_string(num_active_states) + ", " +
                                  std::to_string(num_const_states) + ") exceeds the capacity (" +
                                  std::to_string(capacity_) + ", " +
                                  std::to_string(const_capacity_) + ")");
    }
    num_active_states_ = num_active_states;
    num_const_states_ = num_const_states;
  }

 protected:
  /**
   * @brief Device pointer to the contiguous array of states.
   *
   * Points to GPU device memory containing num_active_states_ states stored
   * sequentially. Each state occupies AmbientDim floats.
   *
   * Memory layout: [state0: AmbientDim floats][state1: AmbientDim floats]...
   *                [stateN-1: AmbientDim floats]
   *
   * Total memory size: num_active_states_ * AmbientDim * sizeof(float) bytes.
   */
  const float *ptr_;

  /**
   * @brief The number of states in this batch.
   *
   * This value determines the total number of states managed by this
   * batch and is used for bounds checking when accessing individual states.
   */
  size_t num_active_states_;

  /** @brief Capacity of the buffer in states (the constructor's capacity). */
  size_t capacity_;

  /**
   * @brief Device pointer to array of constant state indices.
   *
   * Points to GPU device memory containing the indices of states that
   * should remain constant during optimization. The array should contain
   * sorted, unique indices in the range [0, num_active_states_).
   *
   * Can be nullptr if no states are marked as constant.
   *
   * @note This pointer is not owned by this object; the caller is responsible
   *       for managing the lifetime of the array.
   */
  const int *constant_state_ids_ = nullptr;

  /** @brief Number of states that are held constant during optimization.
   */
  size_t num_const_states_ = 0;

  /** @brief Constant-id buffer capacity (the constructor's count). */
  size_t const_capacity_ = 0;
};

}  // namespace cunls
