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

#include "state_batch.h"
#include <cuda_runtime.h>

namespace cunls {

/**
 * @brief Template class for batch processing of state blocks with compile-time
 * known dimensions.
 *
 * This class provides a concrete implementation of StateBatch for state blocks
 * where both the ambient dimension (storage size) and tangent dimension
 * (optimization space) are known at compile time. This enables compile-time
 * optimizations and type safety.
 *
 * The class manages a contiguous array of state blocks stored on the GPU device
 * memory. Each state block occupies AmbientDim floats, and blocks are stored
 * sequentially.
 *
 * @tparam AmbientDim The dimension of the ambient space (storage size per state
 * block). This is the number of floats needed to store one state block.
 * @tparam TangentDim The dimension of the tangent space (optimization space per
 * state block). This is the number of floats needed to represent an
 * update/delta.
 *
 * @note This is a base class that provides storage and access methods. Derived
 * classes (such as SE3StateBatch or VectorStateBatch) must implement the Plus()
 * operation specific to their manifold structure.
 *
 * @note The data pointed to by device_ptr must be allocated on the GPU device
 * and remain valid for the lifetime of this object. The memory layout is:
 *       [block0: AmbientDim floats][block1: AmbientDim floats]...[blockN-1:
 * AmbientDim floats]
 */
template <int AmbientDim, int TangentDim>
class SizedStateBatch : public StateBatch {
public:
  /**
   * @brief Constructs a batch of state blocks without constant state
   * constraints.
   *
   * Creates a batch that manages num_blocks state blocks, all of which can be
   * optimized. The data is stored contiguously in GPU device memory.
   *
   * @param device_ptr Pointer to GPU device memory containing the state blocks.
   *                   Must point to at least num_blocks * AmbientDim floats of
   * allocated memory. The memory layout is: [block0][block1]...[blockN-1],
   * where each block is AmbientDim floats.
   * @param num_blocks The number of state blocks in this batch.
   */
  SizedStateBatch(const float *device_ptr, size_t num_blocks)
      : ptr_(device_ptr), num_blocks_(num_blocks), constant_state_ids_(nullptr),
        num_const_state_blocks_(0) {}

  /**
   * @brief Constructs a batch of state blocks with constant state constraints.
   *
   * Creates a batch that manages num_blocks state blocks, where some blocks may
   * be marked as constant (not optimized). The constant_state_ids array
   * contains the indices of state blocks that should remain fixed during
   * optimization.
   *
   * @param device_ptr Pointer to GPU device memory containing the state blocks.
   *                   Must point to at least num_blocks * AmbientDim floats of
   * allocated memory.
   * @param device_constant_state_ids Pointer to GPU device memory containing
   * the indices of state blocks that should remain constant. Can be nullptr if
   * no blocks are constant. The array should contain sorted, unique indices in
   * [0, num_blocks).
   * @param num_blocks The number of state blocks in this batch.
   * @param num_const_state_blocks The number of state blocks listed in
   *                               @p device_constant_state_ids.
   *
   * @note The constant_state_ids array is not copied; this object stores only
   * the pointer. The caller must ensure the array remains valid for the
   * lifetime of this object.
   */
  SizedStateBatch(const float *device_ptr, size_t num_blocks,
                  const int *device_constant_state_ids,
                  size_t num_const_state_blocks)
      : ptr_(device_ptr), num_blocks_(num_blocks),
        constant_state_ids_(device_constant_state_ids),
        num_const_state_blocks_(num_const_state_blocks) {}

  /**
   * @brief Returns the number of state blocks in this batch.
   *
   * @return The total number of state blocks managed by this batch.
   */
  size_t NumStateBlocks() const final { return num_blocks_; }

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
   * The ambient space dimension determines the storage size of each state
   * block. This is a compile-time constant equal to AmbientDim.
   *
   * @return The ambient space dimension (AmbientDim).
   */
  size_t AmbientSize() const final { return AmbientDim; };

  /**
   * @brief Returns a mutable device pointer to a specific state block.
   *
   * Computes the device memory address of the state block at the given index.
   * The pointer can be used to read or modify the state block data on the GPU.
   *
   * @param state_block_idx The zero-based index of the state block.
   *                        Must be in the range [0, NumStateBlocks()).
   * @return Device pointer to the state block data (AmbientDim floats).
   *         Returns nullptr if state_block_idx is out of bounds.
   *
   * @note The returned pointer points to GPU device memory. Use CUDA memory
   * operations or kernels to access/modify the data.
   */
  float *StateBlockDevicePtr(size_t state_block_idx) final {
    if (state_block_idx >= num_blocks_) {
      return nullptr;
    }

    return const_cast<float *>(ptr_ + state_block_idx * AmbientDim);
  }

  /**
   * @brief Returns a const device pointer to a specific state block.
   *
   * Computes the device memory address of the state block at the given index.
   * The pointer provides read-only access to the state block data on the GPU.
   *
   * @param state_block_idx The zero-based index of the state block.
   *                        Must be in the range [0, NumStateBlocks()).
   * @return Const device pointer to the state block data (AmbientDim floats).
   *         Returns nullptr if state_block_idx is out of bounds.
   *
   * @note The returned pointer points to GPU device memory. Use CUDA memory
   * operations or kernels to read the data.
   */
  const float *StateBlockDevicePtr(size_t state_block_idx) const final {
    if (state_block_idx >= num_blocks_) {
      return nullptr;
    }

    return ptr_ + state_block_idx * AmbientDim;
  }

  /**
   * @brief Returns a pointer to the array of constant state block indices.
   *
   * Returns the device pointer to the array containing indices of state blocks
   * that should remain constant (not optimized) during the optimization
   * process.
   *
   * @return Device pointer to an array of integer indices, or nullptr if no
   * blocks are marked as constant. The array should contain sorted, unique
   * indices in the range [0, NumStateBlocks()).
   *
   * @note The returned pointer is valid only if the batch was constructed with
   *       constant_state_ids. Otherwise, it may be nullptr or uninitialized.
   */
  const int *ConstStateIds() const final { return constant_state_ids_; }

  /**
   * @brief Returns the number of state blocks marked as constant.
   * @return The number of constant (non-optimized) state blocks.
   */
  size_t NumConstStateBlocks() const final { return num_const_state_blocks_; }

protected:
  /**
   * @brief Device pointer to the contiguous array of state blocks.
   *
   * Points to GPU device memory containing num_blocks_ state blocks stored
   * sequentially. Each state block occupies AmbientDim floats.
   *
   * Memory layout: [block0: AmbientDim floats][block1: AmbientDim floats]...
   *                [blockN-1: AmbientDim floats]
   *
   * Total memory size: num_blocks_ * AmbientDim * sizeof(float) bytes.
   */
  const float *ptr_;

  /**
   * @brief The number of state blocks in this batch.
   *
   * This value determines the total number of state blocks managed by this
   * batch and is used for bounds checking when accessing individual blocks.
   */
  size_t num_blocks_;

  /**
   * @brief Device pointer to array of constant state block indices.
   *
   * Points to GPU device memory containing the indices of state blocks that
   * should remain constant during optimization. The array should contain
   * sorted, unique indices in the range [0, num_blocks_).
   *
   * Can be nullptr if no state blocks are marked as constant.
   *
   * @note This pointer is not owned by this object; the caller is responsible
   *       for managing the lifetime of the array.
   */
  const int *constant_state_ids_ = nullptr;

  /** @brief Number of state blocks that are held constant during optimization.
   */
  size_t num_const_state_blocks_ = 0;
};

} // namespace cunls
