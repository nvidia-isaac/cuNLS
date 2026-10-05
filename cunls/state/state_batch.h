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

namespace cunls {

/**
 * @brief Abstract base class for a batch of states on the GPU.
 *
 * StateBatch defines the interface for managing a contiguous batch of
 * states stored in GPU device memory. Each state lives on a
 * (possibly non-Euclidean) manifold with a tangent space used for optimization
 * updates and an ambient space used for storage.
 *
 * Derived classes must implement the Plus() operation that applies a
 * tangent-space update to the states according to the manifold structure.
 */
class StateBatch {
 public:
  /** @brief Virtual destructor. */
  virtual ~StateBatch() = default;

  /**
   * @brief Returns the dimension of the tangent (optimization) space per state.
   * @return The tangent space dimension.
   */
  virtual size_t TangentSize() const = 0;

  /**
   * @brief Returns the dimension of the ambient (storage) space per state.
   * @return The ambient space dimension.
   */
  virtual size_t AmbientSize() const = 0;

  /**
   * @brief Returns the total number of states in this batch.
   * @return The number of states.
   */
  virtual size_t NumActiveStates() const = 0;

  /**
   * @brief Applies a tangent-space update to every state:
   * x_plus_delta = x (+) delta, where (+) is the manifold Plus (vector
   * addition for Euclidean spaces, right-multiplication by the exponential map
   * for Lie groups).
   *
   * <b>Terms</b>
   *
   * - N = NumActiveStates(): states in the batch.
   * - A = AmbientSize(): floats stored per state (e.g. 16 for an SE(3) matrix).
   * - T = TangentSize(): floats per update vector (e.g. 6 for SE(3)).
   * - R = num_replicas: the arrays hold R contiguous copies ("replicas") of
   *   the batch. Replica r is states [r * N, (r + 1) * N). The RANSAC
   *   minimizers keep one replica per hypothesis and update all of them in one
   *   call. The regular minimizers pass R = 1.
   *
   * Every one of the R * N states is updated independently: state i of the
   * output depends only on state i of x and state i of delta.
   *
   * <b>Parameters</b>
   *
   * @param x [in] Device array of R * N * A floats. State i is
   *        `x[i * A .. (i + 1) * A)`.
   * @param delta [in] Device array of R * N * T floats. State i's update is
   *        `delta[i * T .. (i + 1) * T)`.
   * @param x_plus_delta [out] Device array of R * N * A floats, same layout
   *        as x. Must not overlap x or delta.
   * @param stream CUDA stream on which all work is enqueued. The call may
   *        return before the work completes.
   * @param num_replicas R >= 1 (default 1).
   *
   * <b>Example</b> (N = 2 states, R = 3 replicas: 6 states in every array,
   * state i of x at `x + i * A`, of delta at `delta + i * T`)
   *
   * @verbatim
   *     global state i     0     1  |  2     3  |  4     5
   *     replica r          0     0  |  1     1  |  2     2
   *     state within r     0     1  |  0     1  |  0     1
   * @endverbatim
   *
   * <b>Implementing it</b>: treat the arrays as one batch of R * N states,
   * e.g. launch one thread per state with `i < R * N`. Size any internal
   * scratch for R * N states, not N.
   */
  virtual void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
                    size_t num_replicas = 1) = 0;

  /**
   * @brief Returns a mutable device pointer to a specific state.
   *
   * @param state_idx Zero-based index of the state.
   * @return Device pointer to the state data, or nullptr if out of
   * bounds.
   */
  virtual float *StateDevicePtr(size_t state_idx) = 0;

  /**
   * @brief Returns a const device pointer to a specific state.
   *
   * @param state_idx Zero-based index of the state.
   * @return Const device pointer to the state data, or nullptr if out of
   * bounds.
   */
  virtual const float *StateDevicePtr(size_t state_idx) const = 0;

  /**
   * @brief Returns a device pointer to the array of constant state
   * indices.
   * @return Device pointer to integer indices of constant states, or nullptr if
   * none.
   */
  virtual const int *ConstStateIds() const = 0;

  /**
   * @brief Returns the number of states marked as constant.
   * @return The number of constant (non-optimized) states.
   */
  virtual size_t NumConstStates() const = 0;

  /**
   * @brief Number of states the batch's buffer holds: the capacity
   * passed to the constructor. SetNumActiveStates accepts any value up to it.
   */
  virtual size_t Capacity() const { return NumActiveStates(); }

  /**
   * @brief Number of entries the constant-id buffer holds: the const_capacity
   * passed to the constructor (0 without a constant-id buffer).
   */
  virtual size_t ConstCapacity() const { return NumConstStates(); }

  /**
   * @brief Sets the number of active states and of active constant ids,
   * for buffers that are allocated once and rewritten in place between solves.
   *
   * The active states are the first `num_active_states` states of the state
   * buffer; the active constant ids are the first `num_const_states` entries of
   * the constant-id buffer (each must be below `num_active_states`). States
   * past `num_active_states` are neither read nor written. Built-in batches
   * start with 0 active states: call this before the first solve.
   *
   * A host-only assignment: no allocation, no device work. Takes effect at the
   * next Plus / Minimize. Must not be called while a minimization that uses
   * this batch is running. Factors may only reference active states.
   *
   * SizedStateBatch (every built-in state batch) implements it. The default
   * accepts only the current sizes.
   *
   * @param num_active_states Active state count, at most Capacity().
   * @param num_const_states Active constant count, at most ConstCapacity().
   * @throws std::invalid_argument if a count exceeds its capacity.
   * @throws std::logic_error if the batch cannot be resized (default
   *         implementation with different sizes).
   */
  virtual void SetNumActiveStates(size_t num_active_states, size_t num_const_states = 0) {
    if (num_active_states != NumActiveStates() || num_const_states != NumConstStates()) {
      throw std::logic_error("SetNumActiveStates: this state batch cannot be resized");
    }
  }
};

}  // namespace cunls
