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
 * @brief Batch processing for Sim(2) Lie group state blocks.
 *
 * This class implements the Plus operation for the Sim(2) Lie group,
 * which represents similarity transformations (rotation + translation + scale)
 * in 2D. The tangent space has dimension 4 (2 for translation, 1 for rotation,
 * 1 for log-scale), while the ambient space has dimension 9 (3x3 matrix).
 *
 * A Sim(2) transformation matrix is stored in row-major order as 9 floats:
 * [cos(theta), -sin(theta), tx, sin(theta), cos(theta), ty, 0, 0, 1/s]
 *
 * The tangent vector convention is [u_x, u_y, theta, lambda] where u_x, u_y are
 * translational components, theta is the rotation angle, and lambda = log(s)
 * is the log-scale.
 *
 * The exponential map for Sim(2) maps a tangent vector [u_x, u_y, theta,
 * lambda] to: Exp([u_x, u_y, theta, lambda]) = [[cos(theta), -sin(theta), tx],
 *                                      [sin(theta),  cos(theta), ty],
 *                                      [    0,           0,     e^{-lambda}]]
 * where [tx, ty] = V(theta, lambda) * [u_x, u_y] with V being the Sim(2)
 * V-matrix as described in Eade's "Lie Groups for 2D and 3D Transformations".
 *
 * The class uses GPU-accelerated operations via CUDA kernels and cuBLAS
 * for efficient batch processing of multiple transformations.
 */
class Similarity2StateBatch : public SizedStateBatch<9, 4> {
public:
  using Base = SizedStateBatch<9, 4>;

  /**
   * @brief Constructs a batch of Sim(2) state blocks.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param device_ptr Pointer to GPU device memory containing the Sim(2)
   * transforms. Must point to at least num_blocks * 9 floats of allocated
   * memory.
   * @param num_blocks The number of Sim(2) state blocks in this batch.
   */
  Similarity2StateBatch(cuBLASHandle &cublas_handle, const float *device_ptr,
                        size_t num_blocks);

  /**
   * @brief Constructs a batch of Sim(2) state blocks with constant state
   * constraints.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param device_ptr Pointer to GPU device memory containing the Sim(2)
   * transforms. Must point to at least num_blocks * 9 floats of allocated
   * memory.
   * @param num_blocks The number of Sim(2) state blocks in this batch.
   * @param device_constant_state_ids Pointer to GPU device memory containing
   * the indices of state blocks that should remain constant.
   * @param num_const_state_blocks The number of constant state blocks.
   */
  Similarity2StateBatch(cuBLASHandle &cublas_handle, const float *device_ptr,
                        size_t num_blocks, const int *device_constant_state_ids,
                        size_t num_const_state_blocks);

  /**
   * @brief Performs the Plus operation: x_plus_delta = x * Exp(delta)
   *
   * Applies a right-multiplication update to the transformation matrix x
   * using the exponential map of the Lie algebra element delta.
   *
   * @param x Input transformation matrices (device pointer)
   * @param delta Tangent space updates (4D vectors [u_x, u_y, theta, lambda],
   * device pointer)
   * @param x_plus_delta Output transformation matrices (device pointer)
   * @param stream CUDA stream for asynchronous execution
   */
  void Plus(const float *x, const float *delta, float *x_plus_delta,
            cudaStream_t stream) override;

private:
  cuBLASHandle &cublas_handle_; ///< cuBLAS handle for matrix operations

  mutable dvector<Matrix<3>> delta_transforms_;
  mutable dvector<float> tangents_;

  /**
   * @brief Applies a Sim(2) update: result = x * Exp(delta) or
   * result = x * Exp(-delta)
   *
   * Computes the right-multiplication update for Sim(2) transformations.
   * First computes the update matrix Exp(delta) or Exp(-delta)
   * using the Sim(2) exponential map, then performs batched matrix
   * multiplication using cuBLAS.
   *
   * This is a helper function used by Plus (invert_delta=false).
   *
   * Note: cuBLAS uses column-major storage, but our matrices are row-major.
   * For right-multiplication (x * update), we use CUBLAS_OP_N for both
   * operands. cuBLAS interprets the matrices as column-major, so this computes
   * the equivalent of the desired row-major result.
   *
   * @param x Input transformation matrices (device pointer, row-major)
   * @param delta Tangent space updates (4D vectors [u_x, u_y, theta, lambda],
   * device pointer)
   * @param result Output transformation matrices (device pointer, row-major)
   * @param invert_delta If true, compute Exp(-delta), otherwise Exp(delta)
   * @param stream CUDA stream for asynchronous execution
   */
  void ApplyUpdate(const float *x, const float *delta, float *result,
                   bool invert_delta, cudaStream_t stream);
};
} // namespace cunls
