/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "sized_state_batch.h"

namespace cunls {

/**
 * @brief Computes element-wise addition of vectors: x_plus_delta = x + delta.
 *
 * @param x Pointer to input vectors on GPU.
 * @param delta Pointer to delta vectors on GPU.
 * @param x_plus_delta Pointer to output vectors on GPU.
 * @param num_params Number of state blocks.
 * @param dim Dimension of each vector.
 * @param stream CUDA stream for async execution.
 */
void CalculateVectorPlus(const float *x, const float *delta, float *x_plus_delta,
                         size_t num_params, int dim, cudaStream_t stream);

/**
 * @brief Batch of Euclidean vector state blocks with compile-time dimension.
 *
 * For Euclidean state blocks, the tangent and ambient spaces are identical
 * (both have dimension Dim), and the Plus operation reduces to element-wise
 * vector addition: x_plus_delta = x + delta.
 *
 * @tparam Dim The dimension of each vector state block.
 */
template <int Dim>
class VectorStateBatch : public SizedStateBatch<Dim, Dim> {
 public:
  using Base = SizedStateBatch<Dim, Dim>;

  /**
   * @brief Constructs a batch of vector state blocks.
   *
   * @param device_ptr Pointer to GPU device memory containing the vectors.
   *                   Must point to at least num_blocks * Dim floats of allocated memory.
   * @param num_blocks The number of vector state blocks in this batch.
   */
  VectorStateBatch(const float *device_ptr, size_t num_blocks)
      : Base(device_ptr, num_blocks) {}

  /**
   * @brief Constructs a batch of vector state blocks with constant state constraints.
   *
   * @param device_ptr Pointer to GPU device memory containing the vectors.
   *                   Must point to at least num_blocks * Dim floats of allocated memory.
   * @param num_blocks The number of vector state blocks in this batch.
   * @param device_constant_state_ids Pointer to GPU device memory containing the indices
   *                                       of state blocks that should remain constant.
   * @param num_const_state_blocks The number of constant state blocks.
   */
  VectorStateBatch(const float *device_ptr, size_t num_blocks,
                       const int *device_constant_state_ids,
                       size_t num_const_state_blocks)
      : Base(device_ptr, num_blocks, device_constant_state_ids,
             num_const_state_blocks) {}

  /**
   * @brief Computes x_plus_delta = x + delta element-wise for all blocks.
   *
   * @param x             Device pointer to current state values.
   * @param delta         Device pointer to tangent-space updates.
   * @param x_plus_delta  Device pointer to output state values.
   * @param stream        CUDA stream for asynchronous execution.
   */
  void Plus(const float *x, const float *delta, float *x_plus_delta,
            cudaStream_t stream) override {
    CalculateVectorPlus(x, delta, x_plus_delta, this->num_blocks_, Dim, stream);
  }

 private:
  /** @brief Default constructor (private, not for external use). */
  VectorStateBatch() = default;
};

}  // namespace cunls
