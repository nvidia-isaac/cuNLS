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

#include "cunls/common/cublas_helper.h"
#include "cunls/common/types.h"
#include "cunls/state/sized_state_batch.h"

namespace cunls {

/**
 * @brief Batch processing for SE(3) Lie group state blocks.
 *
 * This class implements the Plus operation for the SE(3) Lie group,
 * which represents rigid body transformations (rotations + translations) in
 * 3D. The tangent space has dimension 6 (3 for rotation, 3 for
 * translation), while the ambient space has dimension 16 (4x4 matrix).
 *
 * The class uses GPU-accelerated operations via CUDA kernels and cuBLAS
 * for efficient batch processing of multiple transformations.
 */
class SE3StateBatch : public SizedStateBatch<16, 6> {
 public:
  using Base = SizedStateBatch<16, 6>;

  /**
   * @brief Constructs a batch of SE(3) state blocks.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param device_ptr Pointer to GPU device memory containing the SE(3)
   * transforms. Must point to at least num_blocks * 16 floats of allocated
   * memory.
   * @param num_blocks The number of SE(3) state blocks in this batch.
   */
  SE3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                size_t num_blocks)
      : Base(device_ptr, num_blocks), cublas_handle_(cublas_handle) {}

  /**
   * @brief Constructs a batch of SE(3) state blocks with constant state
   * constraints.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param device_ptr Pointer to GPU device memory containing the SE(3)
   * transforms. Must point to at least num_blocks * 16 floats of allocated
   * memory.
   * @param num_blocks The number of SE(3) state blocks in this batch.
   * @param device_constant_state_ids Pointer to GPU device memory
   * containing the indices of state blocks that should remain constant.
   * @param num_const_state_blocks The number of constant state blocks.
   */
  SE3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                size_t num_blocks, const int* device_constant_state_ids,
                size_t num_const_state_blocks)
      : Base(device_ptr, num_blocks, device_constant_state_ids,
             num_const_state_blocks),
        cublas_handle_(cublas_handle) {}

  /**
   * @brief Performs the Plus operation: x_plus_delta = x * Exp(skew(delta))
   *
   * Applies a right-multiplication update to the transformation matrix x
   * using the exponential map of the Lie algebra element delta.
   *
   * @param x Input transformation matrices (device pointer)
   * @param delta Tangent space updates (6D twist vectors, device pointer)
   * @param x_plus_delta Output transformation matrices (device pointer)
   * @param stream CUDA stream for asynchronous execution
   */
  void Plus(const float* x, const float* delta, float* x_plus_delta,
            cudaStream_t stream) override;

 private:
  cuBLASHandle& cublas_handle_;  ///< cuBLAS handle for matrix operations

  /// Preallocated workspace for delta transform matrices. Only reallocated
  /// if num_transforms exceeds current capacity.
  mutable dvector<SE3Transform> delta_transforms_;

  /// Preallocated workspace for twist vectors. Only reallocated if
  /// num_transforms exceeds current capacity.
  mutable dvector<float> twists_;
};
}  // namespace cunls
