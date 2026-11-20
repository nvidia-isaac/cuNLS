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
 * @brief Storage and bookkeeping of a state batch whose ambient and tangent sizes are set at
 * run time: a contiguous device array of Capacity() states of AmbientSize() floats each
 * ([state0][state1]..., caller-owned, valid for the batch's lifetime), the optional constant
 * state ids, and the active counts. SizedStateBatch fixes the sizes at compile time; derived
 * classes implement Plus() for their manifold.
 */
class DynamicSizedStateBatch : public StateBatch {
 public:
  /**
   * @param ambient_size Floats per state in the storage.
   * @param tangent_size Floats per state update (delta).
   * @param device_ptr Device storage of capacity * ambient_size floats.
   * @param capacity States the storage holds; the active count starts at 0
   *        (SetNumActiveStates).
   * @param device_constant_state_ids Optional device array of the ids of constant states
   *        (const_capacity entries).
   * @param const_capacity Entries of device_constant_state_ids.
   */
  DynamicSizedStateBatch(size_t ambient_size, size_t tangent_size, const float *device_ptr,
                         size_t capacity, const int *device_constant_state_ids = nullptr,
                         size_t const_capacity = 0)
      : ptr_(device_ptr),
        capacity_(capacity),
        constant_state_ids_(device_constant_state_ids),
        const_capacity_(const_capacity),
        ambient_size_(ambient_size),
        tangent_size_(tangent_size) {}

  /** @brief Active states (SetNumActiveStates). */
  size_t NumActiveStates() const final { return num_active_states_; }
  /** @brief Floats per state update. */
  size_t TangentSize() const final { return tangent_size_; }
  /** @brief Floats per stored state. */
  size_t AmbientSize() const final { return ambient_size_; }
  /** @brief Device pointer to state `state_idx`, or nullptr past the capacity. */
  float *StateDevicePtr(size_t state_idx) final {
    return state_idx < capacity_ ? const_cast<float *>(ptr_ + state_idx * ambient_size_) : nullptr;
  }
  /** @brief Device pointer to state `state_idx`, or nullptr past the capacity. */
  const float *StateDevicePtr(size_t state_idx) const final {
    return state_idx < capacity_ ? ptr_ + state_idx * ambient_size_ : nullptr;
  }
  /** @brief Device array of the constant state ids, or nullptr. */
  const int *ConstStateIds() const final { return constant_state_ids_; }
  /** @brief Active constant states (SetNumActiveStates). */
  size_t NumConstStates() const final { return num_const_states_; }
  /** @brief States the storage holds. */
  size_t Capacity() const final { return capacity_; }
  /** @brief Entries of the constant-id array. */
  size_t ConstCapacity() const final { return const_capacity_; }

  /**
   * @brief Sets the active state count and the active constant-id count (host only).
   * @throws std::invalid_argument if either exceeds its capacity.
   */
  void SetNumActiveStates(size_t num_active_states, size_t num_const_states = 0) override {
    if (num_active_states > capacity_ || num_const_states > const_capacity_) {
      throw std::invalid_argument("SetNumActiveStates(" + std::to_string(num_active_states) + ", " +
                                  std::to_string(num_const_states) + ") exceeds the capacity (" +
                                  std::to_string(capacity_) + ", " +
                                  std::to_string(const_capacity_) +
                                  "). Fix: construct the batch with larger capacities.");
    }
    num_active_states_ = num_active_states;
    num_const_states_ = num_const_states;
  }

 protected:
  const float *ptr_;                         ///< Device storage.
  size_t num_active_states_ = 0;             ///< Active states.
  size_t capacity_;                          ///< States the storage holds.
  const int *constant_state_ids_ = nullptr;  ///< Device ids of constant states, or nullptr.
  size_t num_const_states_ = 0;              ///< Active constant ids.
  size_t const_capacity_ = 0;                ///< Entries of constant_state_ids_.

 private:
  size_t ambient_size_;  ///< Floats per stored state.
  size_t tangent_size_;  ///< Floats per state update.
};

/**
 * @brief DynamicSizedStateBatch with the ambient size (floats per stored state) and tangent
 * size (floats per update) fixed at compile time; the base of the built-in manifold states.
 */
template <int AmbientDim, int TangentDim>
class SizedStateBatch : public DynamicSizedStateBatch {
 public:
  /** @brief Storage of capacity * AmbientDim floats; no constant states. */
  SizedStateBatch(const float *device_ptr, size_t capacity)
      : DynamicSizedStateBatch(AmbientDim, TangentDim, device_ptr, capacity) {}

  /** @brief Storage of capacity * AmbientDim floats and constant state ids. */
  SizedStateBatch(const float *device_ptr, size_t capacity, const int *device_constant_state_ids,
                  size_t const_capacity)
      : DynamicSizedStateBatch(AmbientDim, TangentDim, device_ptr, capacity,
                               device_constant_state_ids, const_capacity) {}
};

}  // namespace cunls
