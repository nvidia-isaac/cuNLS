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
   * @brief Applies a tangent-space update to the state blocks.
   *
   * Computes x_plus_delta = x (+) delta, where (+) is the manifold Plus
   * operation (e.g., vector addition for Euclidean spaces, or
   * right-multiplication by the exponential map for Lie groups).
   *
   * @param x           Device pointer to the current state values
   *                    (NumStateBlocks * AmbientSize floats).
   * @param delta       Device pointer to the tangent-space updates
   *                    (NumStateBlocks * TangentSize floats).
   * @param x_plus_delta Device pointer to the output state values
   *                    (NumStateBlocks * AmbientSize floats).
   * @param stream      CUDA stream for asynchronous execution.
   */
  virtual void Plus(const float *x, const float *delta, float *x_plus_delta,
                    cudaStream_t stream) = 0;

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
};

} // namespace cunls
