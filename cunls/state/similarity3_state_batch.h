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

#include "cunls/common/cublas_helper.h"
#include "cunls/state/sized_state_batch.h"

namespace cunls {

/**
 * @brief Batch processing for Sim(3) Lie group state blocks.
 *
 * This class implements the Plus operation for the Sim(3) Lie group,
 * which represents similarity transformations (rotation + translation + scale)
 * in 3D. The tangent space has dimension 7 (3 for rotation, 3 for translation,
 * 1 for log-scale), while the ambient space has dimension 16 (4x4 matrix).
 *
 * A Sim(3) transformation matrix is stored in row-major order as 16 floats:
 * [R00 R01 R02 tx]
 * [R10 R11 R12 ty]
 * [R20 R21 R22 tz]
 * [0   0   0   1/s]
 *
 * The tangent vector convention is [w1, w2, w3, u1, u2, u3, lambda] where
 * w is the rotation axis-angle, u is the translation component, and
 * lambda = log(s) is the log-scale.
 *
 * The class uses GPU-accelerated operations via CUDA kernels and cuBLAS
 * for efficient batch processing of multiple transformations.
 */
class Similarity3StateBatch
    : public SizedStateBatch<16, 7> {
 public:
  using Base = SizedStateBatch<16, 7>;

  /**
   * @brief Constructs a batch of Sim(3) state blocks.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param device_ptr Pointer to GPU device memory containing the Sim(3) transforms.
   *                   Must point to at least num_blocks * 16 floats of allocated memory.
   * @param num_blocks The number of Sim(3) state blocks in this batch.
   */
  Similarity3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                        size_t num_blocks)
      : Base(device_ptr, num_blocks), cublas_handle_(cublas_handle) {}

  /**
   * @brief Constructs a batch of Sim(3) state blocks with constant state constraints.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param device_ptr Pointer to GPU device memory containing the Sim(3) transforms.
   *                   Must point to at least num_blocks * 16 floats of allocated memory.
   * @param num_blocks The number of Sim(3) state blocks in this batch.
   * @param device_constant_state_ids Pointer to GPU device memory containing the indices
   *                                       of state blocks that should remain constant.
   * @param num_const_state_blocks The number of constant state blocks.
   */
  Similarity3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                        size_t num_blocks,
                        const int* device_constant_state_ids,
                        size_t num_const_state_blocks)
      : Base(device_ptr, num_blocks, device_constant_state_ids,
             num_const_state_blocks),
        cublas_handle_(cublas_handle) {}

  /**
   * @brief Performs the Plus operation: x_plus_delta = x * Exp(delta)
   *
   * Applies a right-multiplication update to the transformation matrix x
   * using the exponential map of the Lie algebra element delta.
   *
   * @param x Input transformation matrices (device pointer)
   * @param delta Tangent space updates (7D vectors [w, u, lambda], device pointer)
   * @param x_plus_delta Output transformation matrices (device pointer)
   * @param stream CUDA stream for asynchronous execution
   */
  void Plus(const float* x, const float* delta, float* x_plus_delta,
            cudaStream_t stream) override;

 private:
  cuBLASHandle& cublas_handle_;  ///< cuBLAS handle for matrix operations

  /**
   * @brief Applies a Sim(3) update: result = x * Exp(delta) or
   * result = x * Exp(-delta)
   *
   * @param x Input transformation matrices (device pointer, row-major)
   * @param delta Tangent space updates (7D vectors, device pointer)
   * @param result Output transformation matrices (device pointer, row-major)
   * @param invert_delta If true, compute Exp(-delta), otherwise Exp(delta)
   * @param stream CUDA stream for asynchronous execution
   */
  void ApplyUpdate(const float* x, const float* delta, float* result,
                   bool invert_delta, cudaStream_t stream);
};
}  // namespace cunls
