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
 * @brief Abstract base class for a batch of state blocks on the GPU.
 *
 * StateBatch defines the interface for managing a contiguous batch of
 * state blocks stored in GPU device memory. Each state block lives on a
 * (possibly non-Euclidean) manifold with a tangent space used for optimization
 * updates and an ambient space used for storage.
 *
 * Derived classes must implement the Plus() operation that applies a
 * tangent-space update to the state blocks according to the manifold structure.
 */
class StateBatch {
 public:
  /** @brief Virtual destructor. */
  virtual ~StateBatch() = default;

  /**
   * @brief Returns the dimension of the tangent (optimization) space per block.
   * @return The tangent space dimension.
   */
  virtual size_t TangentSize() const = 0;

  /**
   * @brief Returns the dimension of the ambient (storage) space per block.
   * @return The ambient space dimension.
   */
  virtual size_t AmbientSize() const = 0;

  /**
   * @brief Returns the total number of state blocks in this batch.
   * @return The number of state blocks.
   */
  virtual size_t NumStateBlocks() const = 0;

  /**
   * @brief Applies a tangent-space update to every state block:
   * x_plus_delta = x (+) delta, where (+) is the manifold Plus (vector
   * addition for Euclidean spaces, right-multiplication by the exponential map
   * for Lie groups).
   *
   * <b>Terms</b>
   *
   * - N = NumStateBlocks(): blocks in the batch.
   * - A = AmbientSize(): floats stored per block (e.g. 16 for an SE(3) matrix).
   * - T = TangentSize(): floats per update vector (e.g. 6 for SE(3)).
   * - R = num_replicas: the arrays hold R contiguous copies ("replicas") of
   *   the batch. Replica r is blocks [r * N, (r + 1) * N). The RANSAC
   *   minimizers keep one replica per hypothesis and update all of them in one
   *   call. The regular minimizers pass R = 1.
   *
   * Every one of the R * N blocks is updated independently: block i of the
   * output depends only on block i of x and block i of delta.
   *
   * <b>Parameters</b>
   *
   * @param x [in] Device array of R * N * A floats. Block i is
   *        `x[i * A .. (i + 1) * A)`.
   * @param delta [in] Device array of R * N * T floats. Block i's update is
   *        `delta[i * T .. (i + 1) * T)`.
   * @param x_plus_delta [out] Device array of R * N * A floats, same layout
   *        as x. Must not overlap x or delta.
   * @param stream CUDA stream on which all work is enqueued. The call may
   *        return before the work completes.
   * @param num_replicas R >= 1 (default 1).
   *
   * <b>Example</b> (N = 2 blocks, R = 3 replicas: 6 blocks in every array,
   * block i of x at `x + i * A`, of delta at `delta + i * T`)
   *
   * @verbatim
   *     global block i     0     1  |  2     3  |  4     5
   *     replica r          0     0  |  1     1  |  2     2
   *     block within r     0     1  |  0     1  |  0     1
   * @endverbatim
   *
   * <b>Implementing it</b>: treat the arrays as one batch of R * N blocks,
   * e.g. launch one thread per block with `i < R * N`. Size any internal
   * scratch for R * N blocks, not N.
   */
  virtual void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
                    size_t num_replicas = 1) = 0;

  /**
   * @brief Returns a mutable device pointer to a specific state block.
   *
   * @param state_block_idx Zero-based index of the state block.
   * @return Device pointer to the state block data, or nullptr if out of
   * bounds.
   */
  virtual float *StateBlockDevicePtr(size_t state_block_idx) = 0;

  /**
   * @brief Returns a const device pointer to a specific state block.
   *
   * @param state_block_idx Zero-based index of the state block.
   * @return Const device pointer to the state block data, or nullptr if out of
   * bounds.
   */
  virtual const float *StateBlockDevicePtr(size_t state_block_idx) const = 0;

  /**
   * @brief Returns a device pointer to the array of constant state block
   * indices.
   * @return Device pointer to integer indices of constant blocks, or nullptr if
   * none.
   */
  virtual const int *ConstStateIds() const = 0;

  /**
   * @brief Returns the number of state blocks marked as constant.
   * @return The number of constant (non-optimized) state blocks.
   */
  virtual size_t NumConstStateBlocks() const = 0;

  /**
   * @brief Number of state blocks the batch's buffer holds: the capacity
   * passed to the constructor. SetNumStateBlocks accepts any value up to it.
   */
  virtual size_t Capacity() const { return NumStateBlocks(); }

  /**
   * @brief Number of entries the constant-id buffer holds: the const_capacity
   * passed to the constructor (0 without a constant-id buffer).
   */
  virtual size_t ConstCapacity() const { return NumConstStateBlocks(); }

  /**
   * @brief Sets the number of active state blocks and of active constant ids,
   * for buffers that are allocated once and rewritten in place between solves.
   *
   * The active blocks are the first `num_blocks` blocks of the state buffer;
   * the active constant ids are the first `num_const_state_blocks` entries of
   * the constant-id buffer (each must be below `num_blocks`). Blocks past
   * `num_blocks` are neither read nor written. Built-in batches start with 0
   * active blocks: call this before the first solve.
   *
   * A host-only assignment: no allocation, no device work. Takes effect at the
   * next Plus / Minimize. Must not be called while a minimization that uses
   * this batch is running. Factors may only reference active blocks.
   *
   * SizedStateBatch (every built-in state batch) implements it. The default
   * accepts only the current sizes.
   *
   * @param num_blocks Active block count, at most Capacity().
   * @param num_const_state_blocks Active constant count, at most ConstCapacity().
   * @throws std::invalid_argument if a count exceeds its capacity.
   * @throws std::logic_error if the batch cannot be resized (default
   *         implementation with different sizes).
   */
  virtual void SetNumStateBlocks(size_t num_blocks, size_t num_const_state_blocks = 0) {
    if (num_blocks != NumStateBlocks() || num_const_state_blocks != NumConstStateBlocks()) {
      throw std::logic_error("SetNumStateBlocks: this state batch cannot be resized");
    }
  }
};

}  // namespace cunls
